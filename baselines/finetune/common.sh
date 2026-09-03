#!/usr/bin/env bash
# Shared settings for the baseline LoRA launchers (run_internvl3.sh, run_llava_ov15.sh,
# run_geochat.sh). Source this file; do not execute it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Every baseline trains on the same curriculum SFT file as GeoBox-R1; the prepare scripts
# only rewrite the box numbers into each backbone's native coordinate format.
SOURCE_DATA="${SOURCE_DATA:-${REPO_ROOT}/data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/data/refGeo/images}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
HF_DATA_NORM1000="${DATA_DIR}/refgeo_norm1000.jsonl"          # InternVL3
HF_DATA_NORM1="${DATA_DIR}/refgeo_norm1.jsonl"                # LLaVA-OV-1.5
GEOCHAT_DATA="${DATA_DIR}/refgeo_geochat_native_llava.json"   # GeoChat

# Materialize the HF-trainer files on first use (both variants come from one pass).
prepare_data() {
    if [[ ! -s "${HF_DATA_NORM1000}" || ! -s "${HF_DATA_NORM1}" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_hf_data.py" \
            --input "${SOURCE_DATA}" --image-root "${IMAGE_ROOT}" --output-dir "${DATA_DIR}"
    fi
}

prepare_geochat_data() {
    if [[ ! -s "${GEOCHAT_DATA}" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_geochat_data.py" \
            --input "${SOURCE_DATA}" --image-root "${IMAGE_ROOT}" --output-dir "${DATA_DIR}"
    fi
}

# First free TCP port at or above $1 (default 29500), for torchrun's rendezvous.
find_free_port() {
    "${PYTHON_BIN}" - "${1:-29500}" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, start + 100):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit("no free port found")
PY
}

# The torchrun entry point resolves "python" through PATH; calling the module through an
# explicit interpreter guarantees the intended environment is used.
torchrun_cmd() {
    local python_bin="$1"; shift
    "${python_bin}" -m torch.distributed.run "$@"
}
