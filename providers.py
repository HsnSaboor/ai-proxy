"""Provider routing and request dispatch to upstream APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import httpx

from config import MODEL_ALIASES, MODEL_PROVIDER_MAP, PROVIDERS

log = logging.getLogger("proxy")


@dataclass
class RouteState:
    retry_after: float = 0
    last_attempt: float = 0
    consecutive_errors: int = 0
    cooldown_until: float = 0


route_states: dict[str, RouteState] = defaultdict(RouteState)
_route_lock = asyncio.Lock()


def is_available(provider: str) -> bool:
    state = route_states[provider]
    now = time.time()
    return state.cooldown_until <= now and state.retry_after <= now


async def mark_success(provider: str):
    async with _route_lock:
        state = route_states[provider]
        state.consecutive_errors = 0
        state.retry_after = 0


async def mark_failure(provider: str, retry_after: float = 0):
    async with _route_lock:
        state = route_states[provider]
        state.consecutive_errors += 1
        state.last_attempt = time.time()
        if retry_after > 0:
            state.retry_after = time.time() + retry_after
        if state.consecutive_errors >= 3:
            state.cooldown_until = time.time() + 60 * min(state.consecutive_errors, 10)


def resolve_routes(model_name: str) -> list[tuple[str, str]]:
    """Resolve model name to prioritized list of (provider, upstream_model)."""
    stripped = model_name
    if ":" in model_name:
        stripped = model_name.split(":", 1)[1]
    alias = MODEL_ALIASES.get(stripped) or MODEL_ALIASES.get(model_name.lower())
    providers = MODEL_PROVIDER_MAP.get(alias or stripped, ["pollinations"])
    routes: list[tuple[str, str]] = []
    for prov in providers:
        upstream = get_upstream_model(stripped, prov)
        routes.append((prov, upstream))
    if not any(p == "pollinations" for p, _ in routes):
        routes.append(("pollinations", "openai"))
    return routes


def get_upstream_model(model_name: str, provider: str) -> str:
    """Map our model name to what the upstream provider expects."""
    stripped = model_name
    if ":" in model_name:
        stripped = model_name.split(":", 1)[1]
    if provider == "pollinations":
        if stripped in ("gpt-5.4", "gpt-5.4-reasoning", "gpt-5.2", "gpt-5.2-reasoning", "openai-reasoning"):
            return "openai-large"
        if stripped == "openai-large":
            return "openai-large"
        if stripped in ("gpt-5.5", "gpt-5.5-reasoning"):
            return "gpt-5.5"
        if "nano" in stripped:
            return "openai"
        if "mini" in stripped:
            return "gpt-5.4-mini"
        if "5" in stripped and "gpt" in stripped:
            return "openai-large"
        passthrough = {
            "kimi-k2.6", "mistral", "llama", "llama-scout",
            "qwen-coder", "qwen-large",
            "perplexity-fast", "perplexity-reasoning",
            "openai-audio", "openai-audio-large",
            "minimax",
        }
        if stripped in passthrough:
            return stripped
        family_map = {
            "deepseek": "deepseek", "claude": "claude", "gpt": "openai",
            "llama": "llama", "kimi": "kimi", "mistral": "mistral",
            "qwen": "qwen-coder",
        }
        for key, val in family_map.items():
            if key in stripped.lower():
                return val
        return "openai"
    if provider == "deepinfra":
        exact = {
            "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek-r1": "deepseek-ai/DeepSeek-R1-0528",
            "qwen-3.5-122b": "Qwen/Qwen3.5-122B-A10B",
            "qwen3.5-122b": "Qwen/Qwen3.5-122B-A10B",
            "kimi-k2.6": "moonshotai/Kimi-K2.6",
            "nemotron-super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
            "nemotron-3-super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
            "step-3.5": "stepfun-ai/Step-3.5-Flash",
            "step-3.5-flash": "stepfun-ai/Step-3.5-Flash",
        }
        for key, val in exact.items():
            if key == stripped.lower():
                return val
        family_map = {
            "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
            "llama": "meta-llama/Llama-3.3-70B-Instruct",
            "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
            "qwen3.5": "Qwen/Qwen3.5-122B-A10B",
            "kimi": "moonshotai/Kimi-K2.6",
            "nemotron": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
            "step": "stepfun-ai/Step-3.5-Flash",
        }
        for key, val in family_map.items():
            if key in stripped.lower():
                return val
        return stripped.replace("models/", "").replace("openai/", "")
    return stripped


def _build_headers(provider: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = PROVIDERS.get(provider, {}).get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_url(provider: str) -> str | None:
    info = PROVIDERS.get(provider)
    if not info:
        return None
    return f"{info['base_url']}/chat/completions"


async def try_provider(
    client: httpx.AsyncClient,
    provider: str,
    body: dict[str, Any],
) -> httpx.Response | None:
    """Try a single provider, return response or None."""
    url = _build_url(provider)
    if not url:
        return None

    headers = _build_headers(provider)
    body_copy = {**body}

    for attempt in range(3):
        try:
            resp = await client.post(url, json=body_copy, headers=headers)
            if resp.status_code == 429:
                await mark_failure(provider, 5.0)
                log.warning("429 on %s, failover", provider)
                return None
            if resp.status_code in (401, 403):
                log.warning("Auth error on %s: %d", provider, resp.status_code)
                await mark_failure(provider)
                return resp
            if resp.status_code >= 500:
                log.warning("5xx on %s: %d", provider, resp.status_code)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return resp
            await mark_success(provider)
            return resp
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            log.warning("Error on %s: %s", provider, str(e))
            await mark_failure(provider)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return None
        except Exception as e:
            log.warning("Unexpected error on %s: %s", provider, str(e))
            await mark_failure(provider)
            return None
    return None


async def try_provider_stream(
    client: httpx.AsyncClient,
    provider: str,
    body: dict[str, Any],
) -> tuple[httpx.Response | None, str | None]:
    """Try a single streaming provider, return (response, error)."""
    url = _build_url(provider)
    if not url:
        return None, f"No endpoint for {provider}"

    headers = _build_headers(provider)
    body_copy = {**body, "stream": True}

    for attempt in range(3):
        try:
            req = client.build_request("POST", url, json=body_copy, headers=headers)
            resp = await client.send(req, stream=True)
            if resp.status_code in (401, 403):
                msg = f"HTTP {resp.status_code}"
                try:
                    err_body = await resp.aread()
                    err = json.loads(err_body) if err_body else {}
                    msg = err.get("error", {}).get("message", msg)
                except Exception:
                    pass
                await mark_failure(provider)
                return resp, msg
            if resp.status_code == 429:
                await mark_failure(provider, 5.0)
                log.warning("429 on %s (stream), failover", provider)
                return resp, f"Rate limited on {provider}"
            if resp.status_code >= 500:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return resp, f"Server error on {provider}: {resp.status_code}"
            await mark_success(provider)
            return resp, None
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            await mark_failure(provider)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return None, str(e)
        except Exception as e:
            await mark_failure(provider)
            return None, str(e)
    return None, "All retries exhausted"
