#!/usr/bin/env bash
# Held-out judge-correlation gate: scores the QLoRA adapter and the AWQ base on
# the SFT val split (case_ids the adapter never trained on), writing to
# non-baseline paths so the committed study.json is untouched.
# Run under secret-run for the Sonnet canonical re-judge:
#   ./secret-run bash scripts/gate_judge_qlora.sh
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8001/v1}"
OUTDIR=eval/results/judge_correlation
COMMON=(
  --run-dir eval/results/campaign-critic
  --only-cases eval/datasets/judge_sft/val_case_ids.txt
)

echo ">>> fine-tune (judge-qlora) on held-out" >&2
VLLM_URL="$VLLM_URL" uv run python scripts/run_judge_correlation.py \
  "${COMMON[@]}" \
  --vllm-model judge-qlora \
  --out "$OUTDIR/qlora_heldout_config.yaml" \
  --pairs-out "$OUTDIR/qlora_heldout.json"

echo ">>> base (AWQ, no adapter) on held-out" >&2
VLLM_URL="$VLLM_URL" uv run python scripts/run_judge_correlation.py \
  "${COMMON[@]}" \
  --out "$OUTDIR/base_heldout_config.yaml" \
  --pairs-out "$OUTDIR/base_heldout.json"

echo ">>> done. gate configs:" >&2
echo "    $OUTDIR/qlora_heldout_config.yaml" >&2
echo "    $OUTDIR/base_heldout_config.yaml" >&2
