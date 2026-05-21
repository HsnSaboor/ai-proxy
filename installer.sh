#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="ai-proxy"
SERVICE_DESC="AI Proxy - OpenAI-compatible API for PollinationsAI + DeepInfra free models"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"

if [ ! -x "$UV_BIN" ]; then
    echo "Error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_FILE="$SYSTEMD_USER_DIR/$SERVICE_NAME.service"

mkdir -p "$SYSTEMD_USER_DIR"

cat > "$SERVICE_FILE" << SERVICE_UNIT
[Unit]
Description=$SERVICE_DESC
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
Environment=PORT=8080
ExecStart=$UV_BIN run uvicorn main:app --host 0.0.0.0 --port \$PORT
Restart=always
RestartSec=3
Environment=UV_CACHE_DIR=$HOME/.cache/uv

[Install]
WantedBy=default.target
SERVICE_UNIT

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user restart "$SERVICE_NAME"

echo ""
echo "=== Service installed ==="
echo "  Name:    $SERVICE_NAME"
echo "  File:    $SERVICE_FILE"
echo "  Project: $PROJECT_DIR"
echo "  UV:      $UV_BIN"
echo ""
echo "Status:"
systemctl --user status "$SERVICE_NAME" --no-pager 2>&1 | head -15
echo ""
echo "Logs: journalctl --user -fu $SERVICE_NAME"

if ! loginctl show-user "$USER" 2>/dev/null | grep -q 'Linger=yes'; then
    echo ""
    echo "NOTE: User linger is disabled. Service will stop when you log out."
    echo "      To enable: loginctl enable-linger $USER"
fi
