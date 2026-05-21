"""AI Proxy – high-performance OpenAI-compatible API proxy.

Routes chat, image, audio, and video requests to PollinationsAI and DeepInfra.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import MODEL_ALIASES, PORT, PROVIDERS
from providers import route_states

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(180),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="AI Proxy", version="0.3.0", lifespan=lifespan)

from anthropic import router as anthropic_router
from chat import router as chat_router
from media import router as media_router

app.include_router(chat_router)
app.include_router(media_router)
app.include_router(anthropic_router)


@app.get("/v1/models")
async def list_models(request: Request):
    client = request.app.state.http_client
    all_models: list[dict] = []
    seen: set[str] = set()

    for prov_key, info in PROVIDERS.items():
        try:
            resp = await client.get(f"{info['base_url']}/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                model_list = data.get("data", data.get("models", data.get("result", [])))
                if isinstance(model_list, list):
                    for m in model_list:
                        mid = (m.get("id", m.get("name", "")) if isinstance(m, dict) else str(m))
                        if mid and mid not in seen:
                            seen.add(mid)
                            all_models.append({
                                "id": mid, "object": "model",
                                "created": int(time.time()), "owned_by": prov_key,
                            })
        except Exception as e:
            log.debug("Failed to fetch models from %s: %s", prov_key, e)

    for alias in MODEL_ALIASES:
        if alias not in seen:
            seen.add(alias)
            all_models.append({
                "id": alias, "object": "model",
                "created": int(time.time()), "owned_by": "alias",
            })

    return {"object": "list", "data": all_models}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": list(PROVIDERS.keys()),
    }


@app.get("/providers")
async def list_providers():
    return {
        "providers": list(PROVIDERS.keys()),
        "aliases": len(MODEL_ALIASES),
        "route_states": {
            k: {
                "retry_after": max(0, v.retry_after - time.time()),
                "cooldown": max(0, v.cooldown_until - time.time()),
                "errors": v.consecutive_errors,
            }
            for k, v in route_states.items()
        },
    }


@app.get("/")
async def root():
    return {
        "name": "AI Proxy",
        "version": "0.3.0",
        "providers": list(PROVIDERS.keys()),
        "usage": {
            "list_models": "GET /v1/models",
            "chat": "POST /v1/chat/completions",
            "streaming": "POST /v1/chat/completions (stream: true)",
            "completions": "POST /v1/completions",
            "images": "POST/GET /v1/images/generations",
            "image_edits": "POST /v1/images/edits",
            "audio_transcriptions": "POST /v1/audio/transcriptions",
            "audio_speech": "GET /v1/audio/speech",
            "video": "GET /v1/video/generations",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
