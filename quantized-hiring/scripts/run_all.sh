#!/usr/bin/env bash
# Sequential per-model pipeline (one GPU): quantize+perplexity -> sanity -> inference -> determinism -> cleanup of quantized files.
# Usage: bash scripts/run_all.sh llama32_3b qwen25_3b phi35_mini
cd "$(dirname "$0")/.."
mkdir results/.run_all.lock 2>/dev/null || { echo "another run_all instance holds results/.run_all.lock"; exit 1; }
trap "rmdir results/.run_all.lock" EXIT
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
for KEY in "$@"; do
  echo "=== $KEY: waiting for source ==="
  while [ ! -f models/$KEY/$KEY-F16.gguf ]; do
    [ "$KEY" = phi35_mini ] && grep -q "\[phi35_mini\] done" results/dl_phi.log 2>/dev/null && break
    sleep 30
  done
  sleep 5
  $PY scripts/01_quantize.py --models $KEY --stage all || { echo "quantize stage failed for $KEY"; continue; }
  $PY scripts/03_run_inference.py --sanity --models $KEY | tee results/logs/sanity_$KEY.log
  if grep -q "SANITY FAILED" results/logs/sanity_$KEY.log; then echo "SANITY FAILED for $KEY - stopping"; exit 1; fi
  $PY scripts/03_run_inference.py --models $KEY
  $PY scripts/03_run_inference.py --determinism --models $KEY
  $PY scripts/08_cleanup.py $KEY
done
echo ALL DONE
