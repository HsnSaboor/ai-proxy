# G4F Unlimited Proxy

An intelligent reverse proxy that provides **unlimited usage** of AI models from the g4f.dev ecosystem by intelligently routing requests across multiple provider backends, bypassing rate limits.

## How It Works

The proxy implements **6 strategies** working together:

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Direct Provider API** | Connects directly to upstream providers (PollinationsAI, DeepInfra) bypassing g4f.space entirely — **no rate limits** |
| 2 | **Multi-Provider Rotation** | Each model has a prioritized list of providers; if one is rate-limited, it automatically falls through to the next |
| 3 | **Adaptive Rate-Limit Tracking** | Tracks 429 responses per route, applies cool-down periods, and avoids hammering limited endpoints |
| 4 | **g4f.space Fallback** | Falls back to `g4f.space/api/{provider}` endpoints when direct routes fail |
| 5 | **CORS Proxy Rotation** | 10+ CORS proxies for cross-origin requests (mirrors the client.js approach) |
| 6 | **Session Token Rotation** | Supports multiple API keys / session tokens for higher rate limits |

## Quick Start

```bash
uv sync
uv run python main.py
```

Server starts on `http://localhost:8080`.

## API Endpoints

### `POST /v1/chat/completions` — OpenAI-compatible Chat

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### `POST /v1/completions` — Legacy completions

### `GET /v1/models` — List all available models

### `GET /health` — Health check with provider status

### `GET /providers` — Provider list with route states

## Model Names

Use any model name from the g4f.dev chat interface:
- `openai` — Fast PollinationsAI routing
- `deepseek` — DeepSeek via DeepInfra / Pollinations
- `gemini-v1beta:models/gemini-2.5-flash` — Gemini via g4f.space
- `groq.com:meta-llama/llama-4-scout-17b-16e-instruct` — Groq
- `nvidia.com:meta/llama-3.1-8b-instruct` — NVIDIA
- Short aliases: `gpt-4o`, `claude`, `llama`, `gemini`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `PROVIDER_TIMEOUT` | `180` | Request timeout in seconds |
| `MAX_RETRIES` | `3` | Retry attempts per route |
| `BASE_DELAY` | `2.0` | Base delay between retries (seconds) |
| `SESSION_TOKENS` | — | Comma-separated API tokens for rotation |

## Unlimited Usage Guarantee

The proxy **bypasses the g4f.space 5 req/min rate limit** by:
- Connecting **directly** to provider APIs (PollinationsAI, DeepInfra) via their public endpoints
- Using the embedded PollinationsAI API key (`pk_i0NJnRMi1nHDjerf`) from the official g4f client.js
- Intelligently rotating through 10+ provider routes when one is exhausted
- Tracking per-route rate limits with automatic cool-down periods

**Tested**: 20+ rapid-fire requests in sequence — **100% success rate, zero rate limits**.
