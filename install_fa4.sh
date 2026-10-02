#!/usr/bin/env sh
# Installs Veda's FA4 kernels (NVIDIA) or MLX (Apple) into the Python that
# runs ComfyUI. Usage: ./install_fa4.sh [--check]; set PYTHON to override.
here="$(cd "$(dirname "$0")" && pwd)"
py="${PYTHON:-}"
for candidate in "$here/../../.venv/bin/python" "$here/../../venv/bin/python"; do
    if [ -z "$py" ] && [ -x "$candidate" ]; then py="$candidate"; fi
done
if [ -z "$py" ] && [ -n "$VIRTUAL_ENV" ]; then py="$VIRTUAL_ENV/bin/python"; fi
if [ -z "$py" ]; then py="$(command -v python3 || command -v python)"; fi
echo "Using $py"
exec "$py" "$here/install_fa4.py" "$@"
