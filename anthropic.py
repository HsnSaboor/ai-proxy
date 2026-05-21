"""Anthropic-compatible /v1/messages endpoint.

Translates Anthropic Messages API → OpenAI Chat Completions API
so Claude Code can use any model our proxy supports.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from providers import is_available, resolve_routes, route_states, try_provider, try_provider_stream

log = logging.getLogger("proxy")

router = APIRouter()

ROLE_MAP = {"user": "user", "assistant": "assistant"}
STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
CONTENT_TYPE_MAP = {"text": "text", "tool_use": "tool_use", "tool_result": "tool_result"}


def _get_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def _translate_messages(messages: list[dict], system: str | None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = ROLE_MAP.get(m.get("role", "user"), "user")
        content = m.get("content", "")
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        img = block.get("source", {})
                        if img.get("type") == "base64":
                            texts.append(f"data:{img.get('media_type','image/png')};base64,{img.get('data','')}")
                        elif img.get("type") == "url":
                            texts.append(f"![image]({img.get('url','')})")
                    elif block.get("type") == "tool_use":
                        texts.append(json.dumps({"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": block.get("input", {})}))
                    elif block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            tool_content = " ".join(c.get("text", "") for c in tool_content if isinstance(c, dict))
                        texts.append(json.dumps({"type": "tool_result", "tool_use_id": block.get("tool_use_id"), "content": tool_content}))
            content = "\n".join(texts)
        out.append({"role": role, "content": content})
    return out


def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        if t.get("type") == "custom":
            fn = t.get("function", {})
            out.append({
                "type": "function",
                "function": {"name": fn.get("name", t.get("name", "")), "description": fn.get("description", ""), "parameters": fn.get("parameters", {})},
            })
        else:
            out.append(t)
    return out


def _build_openai_body(anthropic_body: dict) -> dict:
    messages = _translate_messages(anthropic_body.get("messages", []), anthropic_body.get("system"))
    tools = _translate_tools(anthropic_body.get("tools"))
    body = {
        "model": anthropic_body.get("model", "gpt-5.4"),
        "messages": messages,
        "max_tokens": anthropic_body.get("max_tokens", 8192),
        "stream": anthropic_body.get("stream", False),
    }
    for key in ("temperature", "top_p", "stop"):
        if key in anthropic_body:
            body[key] = anthropic_body[key]
    if tools:
        body["tools"] = tools
    return body


def _openai_to_anthropic(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content_blocks: list[dict] = []
    msg_content = message.get("content", "")
    if msg_content:
        content_blocks.append({"type": "text", "text": msg_content})
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                parsed = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            content_blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": parsed})
    usage = data.get("usage", {})
    return {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": STOP_MAP.get(choice.get("finish_reason", ""), choice.get("finish_reason")),
        "stop_sequence": choice.get("stop_sequence"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


@router.post("/v1/messages")
async def messages(request: Request, client: httpx.AsyncClient = Depends(_get_client)):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, {"error": {"message": f"Invalid JSON: {e}"}})

    openai_body = _build_openai_body(body)
    model_name = openai_body["model"]
    routes = resolve_routes(model_name)
    available = [(p, m) for p, m in routes if is_available(p)]
    if not available:
        for p, _ in routes:
            route_states[p].retry_after = 0
            route_states[p].cooldown_until = 0
        available = list(routes)

    if body.get("stream"):
        return StreamingResponse(
            _anthropic_stream(client, available, openai_body, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    errors: list[str] = []
    for provider, upstream_model in available:
        log.info("Anthropic trying %s for %s", provider, model_name)
        body_for_route = {**openai_body, "model": upstream_model, "stream": False}
        resp = await try_provider(client, provider, body_for_route)
        if resp is not None:
            try:
                data = resp.json()
            except Exception:
                errors.append(f"{provider}: invalid JSON")
                continue
            if not data.get("choices"):
                errors.append(f"{provider}: no choices")
                continue
            anthropic_resp = _openai_to_anthropic(data, model_name)
            return JSONResponse(content=anthropic_resp, status_code=resp.status_code)
        errors.append(f"{provider}: unavailable")

    raise HTTPException(429, {"error": {"message": f"All routes exhausted: {'; '.join(errors[-3:])}", "type": "rate_limit"}})


async def _anthropic_stream(
    client: httpx.AsyncClient,
    routes: list[tuple[str, str]],
    body: dict[str, Any],
    original_model: str,
) -> AsyncIterator[str]:
    errors: list[str] = []
    for provider, upstream_model in routes:
        log.info("Anthropic stream trying %s for %s", provider, original_model)
        body_for_route = {**body, "model": upstream_model}
        resp, error = await try_provider_stream(client, provider, body_for_route)
        if resp is not None and resp.status_code == 200:
            msg_id = f"msg_{int(time.time())}"
            yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':original_model,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':1,'cache_creation_input_tokens':0,'cache_read_input_tokens':0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
            ended = False
            try:
                async for line in resp.aiter_lines():
                    if ended:
                        break
                    if line.startswith("data: ") and line != "data: [DONE]":
                        payload = line[6:].strip()
                        if not payload:
                            continue
                        try:
                            obj = json.loads(payload)
                            choices = obj.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                content = delta.get("content", "")
                                if content:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':content}})}\n\n"
                                finish = choices[0].get("finish_reason")
                                if finish:
                                    ended = True
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
                                    usage = obj.get("usage", {})
                                    ot = max(usage.get("completion_tokens", 0), 1)
                                    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':STOP_MAP.get(finish,finish),'stop_sequence':None},'usage':{'output_tokens':ot,'cache_creation_input_tokens':0,'cache_read_input_tokens':0}})}\n\n"
                                    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
                        except json.JSONDecodeError:
                            pass
            finally:
                await resp.aclose()
            return
        if resp is not None:
            await resp.aclose()
        errors.append(error or f"{provider}: HTTP {resp.status_code if resp else '?'}")

    msg = f"All routes exhausted: {'; '.join(errors[-3:])}"
    yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message':msg}})}\n\n"
