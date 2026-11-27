#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
client_python="${OPENPI_CLIENT_PYTHON:-/tmp/openpi-bspline-venv/bin/python}"
if [[ ! -x "$client_python" ]]; then
  echo "找不到 Python：$client_python；请通过 OPENPI_CLIENT_PYTHON 指定客户端环境。" >&2
  exit 1
fi
exec "$client_python" "$project_root/scripts/interactive/launcher.py" client "$@"
