"""Pydantic models for the proxy API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    modalities: list[str] | None = None
    audio: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    reasoning: str | bool | None = None


class ImageGenerationRequest(BaseModel):
    model: str = "flux"
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    quality: str | None = None
    seed: int | None = None
    enhance: bool | None = None
    safe: bool | None = None
    image: str | None = None


class ImageEditRequest(BaseModel):
    prompt: str = ""
    model: str = "flux"
    size: str = "1024x1024"
    response_format: str = "b64_json"
    image: str = ""
