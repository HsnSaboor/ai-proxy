"""Image, audio, and video generation endpoints."""

from __future__ import annotations

import base64
import logging
import time
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import Response as FastResponse
import httpx

from config import PROVIDERS
from models import ImageGenerationRequest, ImageEditRequest

log = logging.getLogger("proxy")

router = APIRouter()


def get_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


@router.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest, client: httpx.AsyncClient = Depends(get_client)):
    errors: list[str] = []
    base_url = PROVIDERS["pollinations"]["base_url"]
    api_key = PROVIDERS["pollinations"]["api_key"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = req.model_dump(exclude_none=True)

    for attempt in range(2):
        try:
            resp = await client.post(f"{base_url}/images/generations", json=body, headers=headers, timeout=60)
            if resp.status_code == 200:
                return JSONResponse(content=resp.json(), status_code=200)
            err_data = resp.json()
            errors.append(f"pollinations: {err_data.get('error', {}).get('message', str(resp.status_code))}")
        except Exception as e:
            errors.append(f"pollinations: {str(e)}")

        if attempt == 0:
            legacy_url = f"https://image.pollinations.ai/prompt/{quote(req.prompt)}"
            w, h = req.size.split("x") if "x" in req.size else (1024, 1024)
            params = {"width": w, "height": h}
            if req.model != "flux":
                params["model"] = req.model
            try:
                legacy_resp = await client.get(legacy_url, params=params, timeout=60)
                if legacy_resp.status_code == 200:
                    b64 = base64.b64encode(legacy_resp.content).decode()
                    return JSONResponse(content={
                        "created": int(time.time()),
                        "data": [{"b64_json": b64, "revised_prompt": req.prompt}],
                    })
            except Exception:
                pass

    log.warning("Image generation failed: %s", errors)
    raise HTTPException(502, {
        "error": {"message": f"Image generation failed: {'; '.join(errors[-3:])}", "type": "upstream_error"},
    })


@router.get("/v1/images/generations")
async def image_generations_get(
    prompt: str,
    client: httpx.AsyncClient = Depends(get_client),
    model: str = "flux", width: int = 1024, height: int = 1024,
):
    try:
        resp = await client.get(
            f"https://image.pollinations.ai/prompt/{quote(prompt)}",
            params={"width": width, "height": height, "model": model},
            timeout=60,
        )
        if resp.status_code == 200:
            return FastResponse(
                content=resp.content,
                media_type="image/jpeg",
                headers={"X-Model-Used": resp.headers.get("x-model-used", model)},
            )
    except Exception as e:
        raise HTTPException(500, {"error": {"message": str(e)}})
    raise HTTPException(500, {"error": {"message": "Image generation failed"}})


@router.post("/v1/images/edits")
async def image_edits(req: ImageEditRequest, client: httpx.AsyncClient = Depends(get_client)):
    w, h = req.size.split("x") if "x" in req.size else (1024, 1024)
    url = f"https://image.pollinations.ai/prompt/{quote(req.prompt)}"
    params: dict[str, str | int] = {"width": w, "height": h, "model": req.model}
    if req.image:
        params["image"] = req.image
    try:
        resp = await client.get(url, params=params, timeout=120)
        if resp.status_code == 200:
            b64 = base64.b64encode(resp.content).decode()
            return JSONResponse(content={
                "created": int(time.time()),
                "data": [{"b64_json": b64, "revised_prompt": req.prompt}],
            })
    except Exception as e:
        raise HTTPException(500, {"error": {"message": str(e)}})
    raise HTTPException(500, {"error": {"message": "Image edit failed"}})


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper"),
    language: str | None = Form(None),
):
    client: httpx.AsyncClient = request.app.state.http_client
    content = await file.read()
    files = {"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")}
    data: dict[str, str] = {"model": model}
    if language:
        data["language"] = language
    headers = {"Authorization": f"Bearer {PROVIDERS['pollinations']['api_key']}"}
    try:
        resp = await client.post(
            f"{PROVIDERS['pollinations']['base_url']}/audio/transcriptions",
            files=files, data=data, headers=headers, timeout=120,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(502, {"error": {"message": str(e)}})


@router.get("/v1/audio/speech")
async def audio_speech(
    request: Request,
    text: str, voice: str = "alloy",
):
    client: httpx.AsyncClient = request.app.state.http_client
    errors: list[str] = []

    # Try Pollinations first
    poll_headers = {"Authorization": f"Bearer {PROVIDERS['pollinations']['api_key']}"}
    try:
        poll_resp = await client.get(
            f"https://gen.pollinations.ai/audio/{quote(text)}",
            params={"voice": voice}, headers=poll_headers, timeout=60,
        )
        if poll_resp.status_code == 200:
            return FastResponse(
                content=poll_resp.content,
                media_type=poll_resp.headers.get("content-type", "audio/mpeg"),
            )
        errors.append(f"pollinations: HTTP {poll_resp.status_code}")
    except Exception as e:
        errors.append(f"pollinations: {str(e)}")

    # Fallback to DeepInfra TTS (no auth needed)
    try:
        deepinfra_body = {"model": "Qwen/Qwen3-TTS-VoiceDesign", "input": text, "voice": "default"}
        deep_resp = await client.post(
            f"{PROVIDERS['deepinfra']['base_url']}/audio/speech",
            json=deepinfra_body,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if deep_resp.status_code == 200:
            return FastResponse(
                content=deep_resp.content,
                media_type=deep_resp.headers.get("content-type", "audio/wav"),
            )
        errors.append(f"deepinfra: HTTP {deep_resp.status_code}")
    except Exception as e:
        errors.append(f"deepinfra: {str(e)}")

    log.warning("TTS failed: %s", errors)
    raise HTTPException(502, {"error": {"message": f"TTS failed: {'; '.join(errors[-3:])}"}})


@router.get("/v1/video/generations")
async def video_generations(
    request: Request,
    prompt: str, model: str = "veo", width: int = 720, height: int = 720, duration: int = 5,
):
    client: httpx.AsyncClient = request.app.state.http_client
    headers = {"Authorization": f"Bearer {PROVIDERS['pollinations']['api_key']}"}
    try:
        resp = await client.get(
            f"https://gen.pollinations.ai/video/{quote(prompt)}",
            params={"model": model, "width": width, "height": height, "duration": duration},
            headers=headers, timeout=300,
        )
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("video/"):
            return FastResponse(content=resp.content, media_type=resp.headers["content-type"])
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(502, {"error": {"message": str(e)}})
