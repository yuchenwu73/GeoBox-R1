#!/usr/bin/env bash
# Launch the GeoBox-R1 demo.
#   bash run_demo.sh                          # GPU 0, port 7860
#   GEOBOX_GPU=1 GEOBOX_PORT=8000 bash run_demo.sh
#   GEOBOX_SHARE=1 bash run_demo.sh           # temporary gradio.live link
#   GEOBOX_MODEL=/path/to/GeoBox-R1 bash run_demo.sh
#   GEOBOX_ATTN=sdpa bash run_demo.sh         # without flash-attn
#   GEOBOX_PYTHON=/path/to/envs/geobox-r1/bin/python bash run_demo.sh
set -e
cd "$(dirname "$0")"

PY="${GEOBOX_PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python executable not found: $PY (set GEOBOX_PYTHON to override)" >&2
  exit 1
fi

# Keep local Gradio traffic outside configured proxies.
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"

export CUDA_VISIBLE_DEVICES="${GEOBOX_GPU:-0}"
export GEOBOX_PORT="${GEOBOX_PORT:-7860}"
export GEOBOX_SHARE="${GEOBOX_SHARE:-0}"

echo "=============================================="
echo " GeoBox-R1 visual grounding demo"
echo " GPU  : ${CUDA_VISIBLE_DEVICES}"
echo " PORT : ${GEOBOX_PORT}"
echo " URL  : http://localhost:${GEOBOX_PORT}"
echo "----------------------------------------------"
echo " For remote servers, forward the port over SSH:"
echo "   ssh -N -L ${GEOBOX_PORT}:localhost:${GEOBOX_PORT} <user>@<server>"
echo " Set GEOBOX_SHARE=1 for a temporary public URL."
echo "=============================================="

exec "$PY" app.py
