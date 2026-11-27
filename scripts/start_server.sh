#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_python="${OPENPI_SERVER_PYTHON:-}"
if [[ -z "$server_python" ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "请先激活服务端环境：source .venv-server/bin/activate" >&2
    exit 1
  fi
  server_python="$VIRTUAL_ENV/bin/python"
fi
if [[ ! -x "$server_python" ]]; then
  echo "找不到 Python：$server_python；请检查已激活的环境。" >&2
  exit 1
fi
exec "$server_python" "$project_root/scripts/interactive/launcher.py" server "$@"
