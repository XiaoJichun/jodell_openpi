#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_python="${OPENPI_SERVER_PYTHON:-/tmp/openpi-server-venv/bin/python}"
if [[ ! -x "$server_python" ]]; then
  echo "找不到 Python：$server_python；请通过 OPENPI_SERVER_PYTHON 指定服务端环境。" >&2
  exit 1
fi
exec "$server_python" "$project_root/scripts/interactive/launcher.py" server "$@"
