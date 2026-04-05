#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="$1"
BASE_URL="${2:-http://localhost:8000/v1}"
OUTPUT_DIR="${3:-results}"
IS_CHAT="${4:-True}"
MAX_TOKENS="${5:-1024}"

MODEL_SHORT="${MODEL_NAME##*/}"
NONPED_TASKS="problem_solving.yaml,socratic_questioning.yaml,student_solution_correctness.yaml,mistake_location.yaml,mistake_correction.yaml"
PED_TASKS=(
  "scaffolding_generation.yaml"
  "pedagogy_following.yaml"
  "scaffolding_generation_hard.yaml"
  "pedagogy_following_hard.yaml"
)

mkdir -p "$OUTPUT_DIR"
START_TS=$(date +%s)

python - "$BASE_URL" "$MODEL_NAME" <<'PY'
import sys, requests
base_url = sys.argv[1].rstrip('/')
want = sys.argv[2]
resp = requests.get(base_url + '/models', timeout=30)
resp.raise_for_status()
ids = [m.get('id') for m in resp.json().get('data', [])]
if want not in ids:
    raise SystemExit(f'Served model name mismatch. Wanted {want}. Available: {ids}')
print('Verified served model name:', want)
PY

rm -f "$OUTPUT_DIR/results-${MODEL_SHORT}.yaml"
rm -f "$OUTPUT_DIR/benchmark_meta-${MODEL_SHORT}.json"

python main.py \
  --tasks "$NONPED_TASKS" \
  --provider completion_api \
  --model_args "base_url=${BASE_URL},model=${MODEL_NAME},is_chat=${IS_CHAT},temperature=0.0,max_tokens=${MAX_TOKENS}" \
  --output "$OUTPUT_DIR"

for TASK in "${PED_TASKS[@]}"; do
  python main.py \
    --tasks "$TASK" \
    --provider completion_api \
    --model_args "base_url=${BASE_URL},model=${MODEL_NAME},is_chat=${IS_CHAT},temperature=0.0,max_tokens=${MAX_TOKENS}" \
    --output "$OUTPUT_DIR"

  TASK_STEM="${TASK%.yaml}"
  python reward_model/compute_scaffolding_score.py \
    --data_path "$OUTPUT_DIR/generations-${MODEL_SHORT}-${TASK_STEM}.json"
done

END_TS=$(date +%s)
ELAPSED_MINUTES=$(python - <<PY
start_ts = $START_TS
end_ts = $END_TS
print(round((end_ts - start_ts) / 60.0, 4))
PY
)

python - "$OUTPUT_DIR" "$MODEL_SHORT" "$ELAPSED_MINUTES" <<'PY'
import json, os, sys
out_dir, model_short, elapsed = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(os.path.join(out_dir, f'benchmark_meta-{model_short}.json'), 'w') as f:
    json.dump({'model': model_short, 'elapsed_minutes': elapsed}, f, indent=2)
print('Wrote benchmark meta for', model_short, 'elapsed_minutes=', elapsed)
PY
