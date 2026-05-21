"""Chat completion endpoints (streaming and non-streaming)."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from models import ChatCompletionRequest
from providers import (
    is_available,
    resolve_routes,
    route_states,
    try_provider,
    try_provider_stream,
)

log = logging.getLogger("proxy")

router = APIRouter()


def get_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, client: httpx.AsyncClient = Depends(get_client)):
    model_name = req.model
    routes = resolve_routes(model_name)

    body = req.model_dump(exclude_none=True)
    body.pop("stream", None)
    body.pop("n", None)

    available = [(p, m) for p, m in routes if is_available(p)]
    if not available:
        log.warning("All routes in cooldown for %s, trying anyway", model_name)
        for p, _ in routes:
            route_states[p].retry_after = 0
            route_states[p].cooldown_until = 0
        available = list(routes)

    if req.stream:
        return StreamingResponse(
            _stream_from_routes(client, available, body, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    errors: list[str] = []
    for provider, upstream_model in available:
        log.info("Trying %s for %s", provider, model_name)
        body_for_route = {**body, "model": upstream_model}
        resp = await try_provider(client, provider, body_for_route)
        if resp is not None:
            try:
                data = resp.json()
            except Exception:
                errors.append(f"{provider}: invalid JSON response")
                continue
            if not data.get("choices"):
                err_msg = data.get("error", {}).get("message", str(data))
                if resp.status_code in (401, 403):
                    errors.append(f"{provider}: auth error - {err_msg}")
                    continue
                if not err_msg:
                    errors.append(f"{provider}: no choices in response")
                    continue
                if "rate" in err_msg.lower() or "limit" in err_msg.lower():
                    errors.append(f"{provider}: {err_msg}")
                    continue
                errors.append(f"{provider}: {err_msg}")
                continue
            data["model"] = model_name
            if "object" in data:
                data["object"] = "chat.completion"
            return JSONResponse(content=data, status_code=resp.status_code)
        errors.append(f"{provider}: unavailable")

    log.warning("All routes exhausted for %s: %s", model_name, errors)
    raise HTTPException(429, {
        "error": {"message": f"All routes exhausted. Errors: {'; '.join(errors[-5:])}", "type": "rate_limit_exceeded"},
    })


async def _stream_from_routes(
    client: httpx.AsyncClient,
    routes: list[tuple[str, str]],
    body: dict[str, Any],
    original_model: str,
) -> AsyncIterator[str]:
    errors: list[str] = []
    for provider, upstream_model in routes:
        log.info("Stream trying %s for %s", provider, body.get("model", "?"))
        body_for_route = {**body, "model": upstream_model}
        resp, error = await try_provider_stream(client, provider, body_for_route)
        if resp is not None:
            if resp.status_code == 200:
                try:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            payload = line[6:]
                            if payload.strip():
                                try:
                                    obj = json.loads(payload)
                                    if "model" in obj or "object" in obj:
                                        obj["model"] = original_model
                                        if "object" in obj and "chat" in obj.get("object", ""):
                                            obj["object"] = "chat.completion.chunk"
                                        yield f"data: {json.dumps(obj)}\n\n"
                                    else:
                                        yield line + "\n"
                                except json.JSONDecodeError:
                                    yield line + "\n"
                            else:
                                yield line + "\n"
                        elif line.strip():
                            yield f"data: {line}\n"
                        else:
                            yield "\n"
                    yield "data: [DONE]\n\n"
                finally:
                    await resp.aclose()
                return
            try:
                err_body = await resp.aread()
                err_data = json.loads(err_body) if err_body else {}
                err_msg = err_data.get("error", {}).get("message", str(resp.status_code))
                errors.append(f"{provider}: {err_msg}")
            except Exception:
                pass
            await resp.aclose()
            continue
        errors.append(f"{provider}: {error}")

    log.warning("Stream exhausted for %s: %s", original_model, errors[:3])
    err_msg = {"error": {"message": f"All routes exhausted: {'; '.join(errors[-3:])}", "type": "rate_limit_exceeded"}}
    yield f"data: {json.dumps(err_msg)}\n\n"
    yield "data: [DONE]\n\n"
