#!/usr/bin/env bash
# Full run: run boltz predict on all PXR compounds returned by the input builder
# (currently 4652 for the published 4139-train + 513-test snapshot)
# with the R1 settings validated by the smoke test.
#
# This is expected to take roughly 4 days on an RTX 5080. Boltz CLI
# resumes automatically by skipping compound directories that already
# contain predictions, so the run is safe to interrupt and re-launch.
#
# Outputs land in structures/boltz2/outputs/. Logs go to logs/boltz2_full.log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# torch 2.11+cu130 inside the boltz uv tool venv ships its CUDA NVRTC libs
# under nvidia/cu13/lib/. dlopen does not look there by default, so the JIT
# kernel compilation in triton/cuequivariance fails with
# "failed to open libnvrtc-builtins.so.13.0". Adding the directory to
# LD_LIBRARY_PATH lets dlopen resolve it.
NVIDIA_CU13_LIB_DIR="$HOME/.local/share/uv/tools/boltz/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$NVIDIA_CU13_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

INPUT_DIR="structures/boltz2/inputs"
OUTPUT_DIR="structures/boltz2/outputs"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/boltz2_full.log"
USE_MSA_SERVER="${USE_MSA_SERVER:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[boltz2_full] start: $(date -Is)" | tee "$LOG_FILE"
echo "[boltz2_full] input_dir : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_full] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_full] yaml count: $(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' | wc -l)" | tee -a "$LOG_FILE"
echo "[boltz2_full] boltz: $(boltz --help 2>&1 | head -1)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# R1 settings (validated by smoke test, see commit 175aa25)
msa_args=()
if [[ "$USE_MSA_SERVER" == "1" ]]; then
    msa_args+=(--use_msa_server)
fi

boltz predict "$INPUT_DIR" \
    "${msa_args[@]}" \
    --out_dir "$OUTPUT_DIR" \
    --use_potentials \
    --diffusion_samples 1 \
    --recycling_steps 3 \
    --output_format mmcif \
    --accelerator gpu \
    --devices 1 \
    --num_workers 2 \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[boltz2_full] done: $(date -Is)" | tee -a "$LOG_FILE"
