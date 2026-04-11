# PedagogicalRL

This repository is the working root for the TutorRM + GDPO tutoring-RL project.
The goal is to compare stock PedagogicalRL training against TutorRM-augmented GRPO and GDPO, then evaluate the resulting models on MathTutorBench and internal PedagogicalRL metrics.

## Where The Instructions Live

- `AGENTS.md` is the current source of truth for the full handoff.
- The helper scripts in this root support setup, benchmark execution, and result aggregation.

## Helper Files

- `constraints_tutor_gdpo.txt` - version and dependency constraints for the shared environment.
- `setup_env.sh` - creates the Python 3.11 environment and installs the stack.
- `run_mathtutorbench_suite.sh` - runs the benchmark suite against a served model.
- `aggregate_results.py` - builds the final CSV summary tables.
-  `.env.example` - template for optional secrets and recommended environment defaults.

## Quick Start

1. Read `AGENTS.md`.
2. Create the environment with `setup_env.sh`.
3. Clone or place the three repos described in the handoff.
4. Patch PedagogicalRL as instructed.
5. Run the baselines, smoke tests, and final A1/A2/A3 jobs.
6. Use `aggregate_results.py` to assemble the final result tables.

## High-Level Plan

- Start from the released `eth-nlped/TutorRL-7B` checkpoint.
- Run A0 baselines on MathTutorBench.
- Train A1 with TutorRM + stock GRPO.
- Train A2 with TutorRM + GDPO.
- Evaluate both, pick the winner, and run the LightEval sanity check once.
- Optionally run A3 with `tutor_rm_mode=all_teacher_turns_mean` if time remains.

## Expected Outputs

- Training logs and checkpoints under `PedagogicalRL/outputs/` and `PedagogicalRL/logs/`
- MathTutorBench generations and scores under `mathtutorbench/results/`
- Summary CSVs under `summary_tables/`

For exact command blocks, file edits, and output contracts, follow `AGENTS.md`.

## Full GPU Run (Step-by-step, A100 or similar)

Assumes your root is `$ROOT` and the working project lives in `tutor_gdpo_project/`.

### 0) Setup both environments (one-time)

Core env (training + benchmarks + internal evals):

```bash
export ROOT=$ROOT
cd "$ROOT"
bash setup_env.sh "$ROOT/tutor_gdpo_project" "$ROOT/constraints_tutor_gdpo.txt"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
```

LightEval env (only used for LightEval):

```bash
cd "$ROOT"
bash setup_lighteval_env.sh "$ROOT/tutor_gdpo_project" "$ROOT/constraints_lighteval.txt"
source "$ROOT/tutor_gdpo_project/.venv_lighteval/bin/activate"
```

When to use which env:
- Use `tutor_gdpo_project/.venv` for A0/A1/A2/A3, training, and MathTutorBench runs.
- Use `tutor_gdpo_project/.venv_lighteval` only for the LightEval step.

### 1) Ports and how to change them

- Training server ports (PedagogicalRL vLLM server):
  - A1: `SERVER_PORT=8005`
  - A2: `SERVER_PORT=8006`
  - A3: `SERVER_PORT=8007`
- Benchmark server port (MathTutorBench vLLM serve): default `8000`.

To change ports:
- For training: set `export SERVER_PORT=XXXX` and override `generation.server_port=XXXX` on the command line.
- For MathTutorBench: set `VLLM_PORT=XXXX` and use `BASE_URL=http://localhost:${VLLM_PORT}/v1`.

### 2) Stage A0a baseline (sequential)

Core env required.

```bash
cd "$ROOT/tutor_gdpo_project/mathtutorbench"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
mkdir -p pids

VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}/v1"
vllm serve eth-nlped/TutorRL-7B \
  --served-model-name TutorRL-7B \
  --seed 42 \
  --tensor-parallel-size 1 \
  --port "${VLLM_PORT}" \
  > serve_TutorRL-7B.log 2>&1 &
echo $! > pids/TutorRL-7B.pid

until curl -fsS "${BASE_URL}/models" >/dev/null; do sleep 5; done
./run_mathtutorbench_suite.sh TutorRL-7B "${BASE_URL}" results 2>&1 | tee baseline_tutorrly7b_bench.log

kill "$(cat pids/TutorRL-7B.pid)" || true
sleep 5
kill -9 "$(cat pids/TutorRL-7B.pid)" 2>/dev/null || true
rm -f pids/TutorRL-7B.pid
```

### 3) Stage A0b baseline (sequential, after A0a)

```bash
cd "$ROOT/tutor_gdpo_project/mathtutorbench"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
mkdir -p pids

VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}/v1"
vllm serve eth-nlped/TutorRL-7B-think \
  --served-model-name TutorRL-7B-think \
  --seed 42 \
  --tensor-parallel-size 1 \
  --port "${VLLM_PORT}" \
  > serve_TutorRL-7B-think.log 2>&1 &
echo $! > pids/TutorRL-7B-think.pid

until curl -fsS "${BASE_URL}/models" >/dev/null; do sleep 5; done
./run_mathtutorbench_suite.sh TutorRL-7B-think "${BASE_URL}" results 2>&1 | tee baseline_tutorrly7b_think_bench.log

kill "$(cat pids/TutorRL-7B-think.pid)" || true
sleep 5
kill -9 "$(cat pids/TutorRL-7B-think.pid)" 2>/dev/null || true
rm -f pids/TutorRL-7B-think.pid
```

### 4) Stage A1 and A2 training (parallel if you have 2 GPUs)

Core env required. These can run in parallel on different GPUs and ports.

Terminal A (GPU 0, A1):
```bash
cd "$ROOT/tutor_gdpo_project/PedagogicalRL"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
export SERVER_PORT=8005
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_grpo.yaml \
  -- generation.server_port=8005 \
  2>&1 | tee logs/final_grpo.log
```

Terminal B (GPU 1, A2):
```bash
cd "$ROOT/tutor_gdpo_project/PedagogicalRL"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
export SERVER_PORT=8006
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_gdpo.yaml \
  -- generation.server_port=8006 \
  2>&1 | tee logs/final_gdpo.log
```

If you only have one GPU, run A1 then A2 sequentially.

### 5) Stage A1/A2 internal evals (sequential)

```bash
cd "$ROOT/tutor_gdpo_project/PedagogicalRL"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"

python eval.py \
  --config-name TutorRL.yaml \
  teacher_model.model_name_or_path=outputs/tutorrm_grpo/model \
  logging.save_dir=outputs/tutorrm_grpo \
  logging.wandb=false \
  2>&1 | tee logs/internal_tutorrm_grpo.log

python eval.py \
  --config-name TutorRL.yaml \
  teacher_model.model_name_or_path=outputs/tutorrm_gdpo/model \
  logging.save_dir=outputs/tutorrm_gdpo \
  logging.wandb=false \
  2>&1 | tee logs/internal_tutorrm_gdpo.log
```

### 6) Stage A1/A2 external MathTutorBench (sequential recommended)

Core env required. Run one model at a time on port 8000 (or use different ports if you truly want parallel).

```bash
cd "$ROOT/tutor_gdpo_project/mathtutorbench"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
mkdir -p pids

VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}/v1"
vllm serve "$ROOT/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_grpo/model" \
  --served-model-name tutorrm-grpo \
  --seed 42 \
  --tensor-parallel-size 1 \
  --port "${VLLM_PORT}" \
  > serve_tutorrm_grpo.log 2>&1 &
echo $! > pids/tutorrm-grpo.pid

until curl -fsS "${BASE_URL}/models" >/dev/null; do sleep 5; done
./run_mathtutorbench_suite.sh tutorrm-grpo "${BASE_URL}" results 2>&1 | tee tutorrm_grpo_bench.log

kill "$(cat pids/tutorrm-grpo.pid)" || true
sleep 5
kill -9 "$(cat pids/tutorrm-grpo.pid)" 2>/dev/null || true
rm -f pids/tutorrm-grpo.pid
```

Repeat for GDPO:

```bash
cd "$ROOT/tutor_gdpo_project/mathtutorbench"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
mkdir -p pids

VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}/v1"
vllm serve "$ROOT/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_gdpo/model" \
  --served-model-name tutorrm-gdpo \
  --seed 42 \
  --tensor-parallel-size 1 \
  --port "${VLLM_PORT}" \
  > serve_tutorrm_gdpo.log 2>&1 &
echo $! > pids/tutorrm-gdpo.pid

until curl -fsS "${BASE_URL}/models" >/dev/null; do sleep 5; done
./run_mathtutorbench_suite.sh tutorrm-gdpo "${BASE_URL}" results 2>&1 | tee tutorrm_gdpo_bench.log

kill "$(cat pids/tutorrm-gdpo.pid)" || true
sleep 5
kill -9 "$(cat pids/tutorrm-gdpo.pid)" 2>/dev/null || true
rm -f pids/tutorrm-gdpo.pid
```

### 7) Choose winner and run LightEval (sequential)

Aggregate results (core env), then LightEval with the lighteval env.

```bash
cd "$ROOT/tutor_gdpo_project"
source "$ROOT/tutor_gdpo_project/.venv/bin/activate"
python aggregate_results.py --project-root "$ROOT/tutor_gdpo_project" --output-dir "$ROOT/tutor_gdpo_project/summary_tables"
```

Then LightEval (lighteval env):

```bash
cd "$ROOT/tutor_gdpo_project"
source "$ROOT/tutor_gdpo_project/.venv_lighteval/bin/activate"
# Set WINNER_MODEL based on results_external.csv
lighteval vllm \
  "model_name=${WINNER_MODEL},gpu_memory_utilization=0.85,max_model_length=4096,dtype=bfloat16,generation_parameters={max_new_tokens:2048,temperature:0.0}" \
  "lighteval|math_500|0|0,helm|mmlu|5|0,lighteval|gsm8k|4|0" \
  --use-chat-template
```

### 8) A3 (optional, only after A1/A2 complete)

Run A3 only after A1/A2 training + evals are done and you have picked a winner. Use `SERVER_PORT=8007` and a new config (see `AGENTS.md` for the exact generation step).

---

If anything here conflicts with `AGENTS.md`, follow `AGENTS.md` as the source of truth and adjust the local commands accordingly.
