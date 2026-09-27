#!/usr/bin/env bash
# Run on the EC2 instance (Ubuntu 24.04) from ~/Amazon_ml after unpacking upload.tgz / code.tgz:
#     bash aws/run_on_ec2.sh            # full: train (writes the model-input frames), predict test, validate
#     bash aws/run_on_ec2.sh predict    # only predict with an existing trained model
#     bash aws/run_on_ec2.sh bench      # train OLD code (aws/baseline_src) and NEW code on the same
#                                       # sample, paired-bootstrap the difference, then predict
#     bash aws/run_on_ec2.sh ce         # GPU: train the cross-encoder on sample C, score the train/test frames
#     bash aws/run_on_ec2.sh stack      # level-1 ensemble + level-2 meta-model on the cached frames,
#                                       # ablation with bootstrap, then predict from the test frame and validate
#     bash aws/run_on_ec2.sh gpu        # ce, then stack
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
fi
uv pip install --python "$ROOT/.venv/bin/python" -r code/business_entity_resolution/requirements.txt  # no-op when current
PY="$ROOT/.venv/bin/python"

# ---- use the whole machine: normalisation processes = vCPUs
export ER_WORKERS=$(nproc)
export PYTHONUNBUFFERED=1
echo "vCPUs: $(nproc), RAM: $(free -g | awk '/Mem:/{print $2}') GB" | tee -a "$LOG"

cd code/business_entity_resolution
validate() {
    (cd "$ROOT/student_resource" && "$PY" utils/validate_submission.py         --matching ../output/matching_results.tsv         --candidate ../output/candidate_pairs.tsv         --test-dir dataset/test --check-ids 2>&1 | tee -a "$LOG")
}
if [ "$MODE" = "bench" ]; then
    # Old code (git HEAD, aws/baseline_src, reporting-only patch) vs new code on the SAME seeded
    # 400k-entity sample, then a paired bootstrap of per-entity OOF F0.5 (see pipeline --baseline).
    BASE_ART="$ROOT/aws/baseline_artifacts"
    if [ ! -f "$BASE_ART/oof_entity_f05.parquet" ]; then
        echo "==== BASELINE (old code) ====" | tee -a "$LOG"
        "$PY" "$ROOT/aws/baseline_src/pipeline.py" train --sample 400000 --min-free-gb 0             --data-dir "$ROOT/student_resource/dataset" --cache-dir "$ROOT/code/business_entity_resolution/cache"             --artifact-dir "$BASE_ART" --output-dir "$ROOT/aws/baseline_output" 2>&1 | tee -a "$LOG"
    fi
    echo "==== NEW CODE ====" | tee -a "$LOG"
    "$PY" src/pipeline.py train --sample 400000 --dump-errors --min-free-gb 0         --baseline "$BASE_ART" 2>&1 | tee -a "$LOG"
    "$PY" src/pipeline.py predict --min-free-gb 0 2>&1 | tee -a "$LOG"
    validate
elif [ "$MODE" = "all" ]; then
    # Full training data: stage-2 sample of 400k S1 (the laptop default is 200k). The previous
    # run's artifacts are kept once in aws/prev_artifacts and used as the bootstrap baseline.
    PREV="$ROOT/aws/prev_artifacts"
    if [ -f artifacts/oof_entity_f05.parquet ] && [ ! -d "$PREV" ]; then cp -r artifacts "$PREV"; fi
    BASE_OPT=()
    if [ -f "$PREV/oof_entity_f05.parquet" ]; then BASE_OPT=(--baseline "$PREV"); fi
    "$PY" src/pipeline.py train --sample 400000 --dump-errors --min-free-gb 0 "${BASE_OPT[@]}" 2>&1 | tee -a "$LOG"
    "$PY" src/pipeline.py predict --min-free-gb 0 2>&1 | tee -a "$LOG"
    validate
elif [ "$MODE" = "predict" ]; then
    "$PY" src/pipeline.py predict --min-free-gb 0 2>&1 | tee -a "$LOG"
    validate
fi
if [ "$MODE" = "ce" ] || [ "$MODE" = "gpu" ]; then
    nvidia-smi 2>&1 | head -15 | tee -a "$LOG" || true
    "$PY" src/pipeline.py ce-train --min-free-gb 0 2>&1 | tee -a "$LOG"
    "$PY" src/pipeline.py ce-score --min-free-gb 0 2>&1 | tee -a "$LOG"
fi
if [ "$MODE" = "stack" ] || [ "$MODE" = "gpu" ]; then
    "$PY" src/pipeline.py stack --min-free-gb 0 2>&1 | tee -a "$LOG"
    "$PY" src/pipeline.py stack-predict --min-free-gb 0 2>&1 | tee -a "$LOG"
    validate
fi
echo "DONE ($MODE). Outputs in $ROOT/output/" | tee -a "$LOG"
