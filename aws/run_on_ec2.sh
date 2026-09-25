#!/usr/bin/env bash
# Run on the EC2 instance (Ubuntu 24.04) from ~/Amazon_ml after unpacking upload.tgz:
#     bash aws/run_on_ec2.sh            # full: train on all training data, predict test, validate
#     bash aws/run_on_ec2.sh predict    # only predict with an existing trained model
#     bash aws/run_on_ec2.sh bench      # train OLD code (aws/baseline_src) and NEW code on the same
#                                       # sample, paired-bootstrap the difference, then predict
# Everything is logged to ~/Amazon_ml/run.log. Safe to run inside tmux (recommended).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
MODE=${1:-all}
LOG="$ROOT/run.log"

# ---- system library LightGBM needs (OpenMP); minimal Ubuntu images lack it
if ! ldconfig -p | grep -q libgomp.so.1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq libgomp1
fi

# ---- environment: Python 3.13 via uv (matches the pinned requirements)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    uv venv --python 3.13 "$ROOT/.venv"
    uv pip install --python "$ROOT/.venv/bin/python" -r code/business_entity_resolution/requirements.txt
fi
PY="$ROOT/.venv/bin/python"

# ---- use the whole machine: normalisation processes = vCPUs
export ER_WORKERS=$(nproc)
export PYTHONUNBUFFERED=1
echo "vCPUs: $(nproc), RAM: $(free -g | awk '/Mem:/{print $2}') GB" | tee -a "$LOG"

cd code/business_entity_resolution
if [ "$MODE" = "bench" ]; then
    # Old code (git HEAD, aws/baseline_src, reporting-only patch) vs new code on the SAME seeded
    # 400k-entity sample, then a paired bootstrap of per-entity OOF F0.5 (see pipeline --baseline).
    BASE_ART="$ROOT/aws/baseline_artifacts"
    if [ ! -f "$BASE_ART/oof_entity_f05.parquet" ]; then
        echo "==== BASELINE (old code) ====" | tee -a "$LOG"
        "$PY" "$ROOT/aws/baseline_src/pipeline.py" train --sample 400000 --min-free-gb 0 \
            --data-dir "$ROOT/student_resource/dataset" --cache-dir "$ROOT/code/business_entity_resolution/cache" \
            --artifact-dir "$BASE_ART" --output-dir "$ROOT/aws/baseline_output" 2>&1 | tee -a "$LOG"
    fi
    echo "==== NEW CODE ====" | tee -a "$LOG"
    "$PY" src/pipeline.py train --sample 400000 --dump-errors --min-free-gb 0 \
        --baseline "$BASE_ART" 2>&1 | tee -a "$LOG"
elif [ "$MODE" = "all" ]; then
    # Full training data: stage-2 sample of 400k S1 (the laptop default is 200k).
    "$PY" src/pipeline.py train --sample 400000 --dump-errors --min-free-gb 0 2>&1 | tee -a "$LOG"
fi
"$PY" src/pipeline.py predict --min-free-gb 0 2>&1 | tee -a "$LOG"

# ---- official format check
cd "$ROOT/student_resource"
"$PY" utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids 2>&1 | tee -a "$LOG"
echo "DONE. Outputs in $ROOT/output/" | tee -a "$LOG"
