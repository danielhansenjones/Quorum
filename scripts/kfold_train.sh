#!/usr/bin/env bash
# Cross-validation training: one adapter per fold. Local GPU only, no API.
# Each fold trains on every case_id except its own held-out slice, so when the
# folds are gated together (kfold_gate.sh) every question is scored by an adapter
# that never trained on it - turning the 7-question val slice into full-corpus
# coverage without generating a single new case.
#
#   uv run python scripts/build_judge_sft.py --kfold 5   # writes fold_0..fold_4
#   bash scripts/kfold_train.sh 5
set -euo pipefail

K="${1:-5}"
DATA=eval/datasets/judge_sft
MODELS=eval/models

for i in $(seq 0 $((K - 1))); do
  echo ">>> training fold $i / $((K - 1))" >&2
  uv run python scripts/train_judge_qlora.py \
    --data "$DATA/fold_$i" \
    --out "$MODELS/judge-fold-$i"
done
echo ">>> done. adapters in $MODELS/judge-fold-0 .. judge-fold-$((K - 1))" >&2
echo ">>> serve them on vLLM (--enable-lora), then: ./secret-run bash scripts/kfold_gate.sh $K" >&2
