#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
client_python="${OPENPI_CLIENT_PYTHON:-}"
if [[ -z "$client_python" ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "请先激活客户端环境：source .venv-real/bin/activate" >&2
    exit 1
  fi
  client_python="$VIRTUAL_ENV/bin/python"
fi
if [[ ! -x "$client_python" ]]; then
  echo "找不到 Python：$client_python；请检查已激活的环境。" >&2
  exit 1
fi
exec "$client_python" "$project_root/scripts/interactive/launcher.py" client "$@"
