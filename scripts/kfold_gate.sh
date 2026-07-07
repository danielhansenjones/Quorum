#!/usr/bin/env bash
# Cross-validated gate: score each fold's held-out questions with that fold's
# adapter, then aggregate. vLLM must already serve the K adapters
# (judge-fold-0 .. judge-fold-K-1) via --enable-lora. Sonnet re-judges the qual
# citations, so run under secret-run:
#   ./secret-run bash scripts/kfold_gate.sh 5
#
# Quality is read from the committed Sonnet scores (no new Sonnet calls); only
# the qual-faithfulness half spends, and each claim is judged once across all
# folds, so the Sonnet cost is a fraction of a dollar.
set -euo pipefail

K="${1:-5}"
VLLM_URL="${VLLM_URL:-http://localhost:8001/v1}"
DATA=eval/datasets/judge_sft
OUTDIR=eval/results/judge_correlation/kfold

mkdir -p "$OUTDIR"
for i in $(seq 0 $((K - 1))); do
  echo ">>> gating fold $i (adapter judge-fold-$i)" >&2
  VLLM_URL="$VLLM_URL" uv run python scripts/run_judge_correlation.py \
    --run-dirs eval/results/campaign-critic \
    --only-cases "$DATA/fold_$i/val_case_ids.txt" \
    --vllm-model "judge-fold-$i" \
    --out "$OUTDIR/fold_${i}_config.yaml" \
    --pairs-out "$OUTDIR/fold_${i}_study.json"
done

echo ">>> aggregating $K folds" >&2
uv run python scripts/aggregate_kfold.py --kfold "$K" --dir "$OUTDIR"
