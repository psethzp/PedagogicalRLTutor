# AGENTS_FINAL.md — single end-to-end handoff for TutorRM + GDPO on PedagogicalRL

## 0) What to give the agent

Give the agent exactly these six files plus the instruction to follow `AGENTS_FINAL.md` as the only source of truth:

- `AGENTS_FINAL.md`
- `constraints_tutor_gdpo.txt`
- `.env.example`
- `setup_env.sh`
- `run_mathtutorbench_suite.sh`
- `aggregate_results.py`


## 1) Fixed project choices

This project is fixed.

- Main training repo: `eth-lre/PedagogicalRL`
- Start checkpoint: `eth-nlped/TutorRL-7B`
- External benchmark repo: `eth-lre/mathtutorbench`
- Tutor reward-model repo: `Kpetyxova/Towards_Reward_Modeling_for_Tutors`
- Released TutorRM model: `kpetyxova/towards-reward-modeling-tutors`
- Main comparison:
  - **A1:** TutorRM + stock PedagogicalRL summed-reward GRPO
  - **A2:** TutorRM + GDPO
- Baselines:
  - **A0a:** `eth-nlped/TutorRL-7B` on MathTutorBench
  - **A0b:** `eth-nlped/TutorRL-7B-think` on MathTutorBench
- Add-on only after A1/A2 are complete:
  - **A3:** `first_teacher_only` vs `all_teacher_turns_mean`
- Hardware assumption: two independent one-GPU runs in parallel
- Training strategy: short continuation from released TutorRL, not full reproduction from raw Qwen

Do not redesign the project. Do not swap repos. Do not retrain TutorRM. Do not attempt the full PedagogicalRL paper budget.

## 2) What each helper file is for

### `constraints_tutor_gdpo.txt`
Shared best-effort version lock for the one-environment setup.

### `.env.example`
Template for optional secrets and recommended environment defaults.

### `setup_env.sh`
Creates the Python 3.11 venv, loads `.env`, exports `HUGGING_FACE_HUB_TOKEN` from `HF_TOKEN` if needed, installs the shared stack, writes `env_versions.json`, and avoids MathTutorBench downgrading the training stack.

### `run_mathtutorbench_suite.sh`
Runs MathTutorBench in the correct split:
- five non-pedagogical tasks together
- four pedagogical generation tasks separately
- `compute_scaffolding_score.py` after each pedagogical task
- stable served-model-name verification
- writes benchmark meta JSON with elapsed minutes

### `aggregate_results.py`
Builds the three final paper tables from local logs and MathTutorBench outputs:
- `results_internal.csv`
- `results_external.csv`
- `results_efficiency.csv`

### `AGENTS_FINAL.md`
The only master instruction file.

## 3) Final folder layout

Create exactly this layout:

```text
~/tutor_gdpo_project/
  AGENTS_FINAL.md
  constraints_tutor_gdpo.txt
  .env.example
  .env
  setup_env.sh
  aggregate_results.py
  PedagogicalRL/
  mathtutorbench/
    run_mathtutorbench_suite.sh
  Towards_Reward_Modeling_for_Tutors/
```

## 4) Locked environment choice

Use this single shared environment:

- Python `3.11`
- PyTorch `2.6.0`
- torchvision `0.21.0`
- torchaudio `2.6.0`
- vLLM `0.8.3`
- flash-attn `2.7.4.post1`
- Transformers `4.50.3`
- TRL `0.18.0`
- Accelerate `1.6.0`
- Datasets `3.1.0`
- Hydra `1.3.2`
- OmegaConf `2.3.0`

Install MathTutorBench with `--no-deps` after the core stack is fixed.

## 5) Clone repos

```bash
mkdir -p ~/tutor_gdpo_project
cd ~/tutor_gdpo_project

git clone https://github.com/eth-lre/PedagogicalRL.git
git clone https://github.com/eth-lre/mathtutorbench.git
git clone https://github.com/Kpetyxova/Towards_Reward_Modeling_for_Tutors.git
```

Checklist:
- `PedagogicalRL/train_rl.py` exists
- `PedagogicalRL/vllm_server.py` exists
- `PedagogicalRL/eval.py` exists
- `mathtutorbench/main.py` exists
- `mathtutorbench/reward_model/compute_scaffolding_score.py` exists
- `Towards_Reward_Modeling_for_Tutors/inference.py` exists

## 6) Copy helper files into place

```bash
cd ~/tutor_gdpo_project
cp /path/to/AGENTS_FINAL.md .
cp /path/to/constraints_tutor_gdpo.txt .
cp /path/to/.env.example .
cp /path/to/setup_env.sh .
cp /path/to/aggregate_results.py .
chmod +x setup_env.sh

cp /path/to/run_mathtutorbench_suite.sh mathtutorbench/
chmod +x mathtutorbench/run_mathtutorbench_suite.sh
```

## 7) Create `.env`

```bash
cd ~/tutor_gdpo_project
cp .env.example .env
```

Rules:
- `HF_TOKEN`: optional but recommended
- `WANDB_API_KEY`: optional
- `OPENROUTER_API_KEY`: optional
- `WANDB_MODE=disabled` by default
- if no `WANDB_API_KEY`, everything must still run and log locally

## 8) Build the environment

```bash
cd ~/tutor_gdpo_project
bash setup_env.sh ~/tutor_gdpo_project ~/tutor_gdpo_project/constraints_tutor_gdpo.txt
source ~/tutor_gdpo_project/.venv/bin/activate
```

After install, run this exact sanity check:

```bash
python - <<'PY'
import torch, transformers, trl, fastapi, uvicorn, datasets
print('torch', torch.__version__)
print('cuda?', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('trl', trl.__version__)
PY
```

Checklist:
- `.venv` exists
- `env_versions.json` exists
- `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- `python -c "import transformers, trl, vllm; print(transformers.__version__, trl.__version__, vllm.__version__)"` prints `4.50.3 0.18.0 0.8.3`

## 9) Patch PedagogicalRL launcher so two runs can coexist

This stage is mandatory.

### Why this stage exists
PedagogicalRL’s stock launcher hardcodes `http://localhost:8005/docs` when waiting for the server, and `stop_vllm_server.sh` uses global `pkill` on `uvicorn`, `multiprocess.spawn`, and `vllm_server.py`. That means two ablations running at once will collide and kill each other.

### 9.1 Replace `PedagogicalRL/stop_vllm_server.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SERVER_PORT="${SERVER_PORT:-8005}"
PID_FILE=".vllm_${SERVER_PORT}.pid"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" || true
    sleep 2
    kill -9 "${PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi
```

Then:

```bash
chmod +x PedagogicalRL/stop_vllm_server.sh
```

### 9.2 Replace `PedagogicalRL/start_vllm_server.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SERVER_PORT="${SERVER_PORT:-8005}"
./stop_vllm_server.sh
exec python vllm_server.py "$@"
```

Then:

```bash
chmod +x PedagogicalRL/start_vllm_server.sh
```

### 9.3 Patch `PedagogicalRL/start_rl_training.sh`

Required changes:
- read `SERVER_PORT="${SERVER_PORT:-8005}"`
- after starting the server, write `.vllm_${SERVER_PORT}.pid`
- replace the hardcoded wait loop URL with `http://localhost:${SERVER_PORT}/docs`
- keep the rest of the accelerate logic intact

Checklist:
- no hardcoded `8005` in the wait loop
- no global `pkill` left anywhere

### 9.4 Validate the launcher patch

From `~/tutor_gdpo_project/PedagogicalRL`, run:

```bash
grep -n "localhost:" start_rl_training.sh
grep -n "pkill" stop_vllm_server.sh || true
```

Expected result:
- `start_rl_training.sh` should reference `localhost:${SERVER_PORT}`
- `stop_vllm_server.sh` should contain **no** `pkill`

## 10) Add exact local-only logging fallback patches

This section is mandatory. Do not implement local logging by interpretation. Apply the exact code changes below.

### 10.1 `PedagogicalRL/train_rl.py`

#### 10.1.1 Add imports at the top
Add these imports if missing:

```python
import json
from pathlib import Path
```

#### 10.1.2 Right after config extraction in `main(cfg: RLModelTrainingConfig)`

Right after:

```python
model_config = cfg.teacher_model
train_config = cfg.train
logging_config = cfg.logging
lora_config = model_config.lora
data_config = cfg.dataset
```

insert:

```python
Path(logging_config.save_dir).mkdir(parents=True, exist_ok=True)
```

#### 10.1.3 Immediately after `accelerator = Accelerator(kwargs_handlers=kwargs)`

Right after:

```python
accelerator = Accelerator(kwargs_handlers=kwargs)
```

insert:

```python
use_wandb = bool(logging_config.wandb and os.getenv("WANDB_API_KEY"))
if logging_config.wandb and not use_wandb and accelerator.is_main_process:
    logger.warning(
        "WANDB requested but WANDB_API_KEY is missing; continuing with local-only logging."
    )

if accelerator.is_main_process:
    OmegaConf.save(cfg, Path(logging_config.save_dir) / "resolved_config.yaml")

accelerator.wait_for_everyone()
```

#### 10.1.4 Replace the stock W&B init block

Replace:

```python
if logging_config.wandb and accelerator.is_main_process:
    wandb.init(
        project=logging_config.wandb_project,
        name=logging_config.wandb_run_name,
        entity=logging_config.wandb_entity,
        group=logging_config.run_group,
        tags=logging_config.wandb_tags,
        config=OmegaConf.to_object(cfg),
    )
```

with:

```python
if use_wandb and accelerator.is_main_process:
    wandb.init(
        project=logging_config.wandb_project,
        name=logging_config.wandb_run_name,
        entity=logging_config.wandb_entity,
        group=logging_config.run_group,
        tags=logging_config.wandb_tags,
        config=OmegaConf.to_object(cfg),
    )
```

#### 10.1.5 Replace the `report_to` line in `ClassroomGRPOConfig(...)`

Replace:

```python
report_to=["wandb"] if logging_config.wandb else [],
```

with:

```python
report_to=["wandb"] if use_wandb else [],
```

#### 10.1.6 Add local train summary at the end

Right after:

```python
trainer.save_model(logging_config.save_dir + "/model")
```

insert:

```python
if accelerator.is_main_process:
    summary = {
        "output_dir": logging_config.save_dir,
        "last_checkpoint": last_ckpt,
        "model_name_or_path": model_config.model_name_or_path,
        "train_results": getattr(train_results, "metrics", str(train_results)),
    }
    with open(Path(logging_config.save_dir) / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
```

### 10.2 `PedagogicalRL/vllm_server.py`

#### 10.2.1 Add imports and globals

Add these imports if missing:

```python
from pathlib import Path
```

Add this global near the existing globals:

```python
use_wandb = False
```

#### 10.2.2 In `main(cfg: RLModelTrainingConfig)`

Right after:

```python
config = cfg
```

insert:

```python
global use_wandb

Path(cfg.logging.save_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.logging.save_dir, "server_batches").mkdir(parents=True, exist_ok=True)

use_wandb = bool(cfg.logging.wandb and os.getenv("WANDB_API_KEY"))
if cfg.logging.wandb and not use_wandb:
    logger.warning(
        "WANDB requested for server logging but WANDB_API_KEY is missing; continuing with local-only server logging."
    )
```

Then replace:

```python
if cfg.logging.wandb:
    wandb.init(
        ...
    )
```

with:

```python
if use_wandb:
    wandb.init(
        ...
    )
```

#### 10.2.3 In `sample_conversations(...)`

Right after all reward columns and `total_reward` are added to `df_table`, but before `df_table = df_table.astype(str)`, insert:

```python
batch_idx = len(classroom.conversation_sets)
batch_path = Path(config.logging.save_dir) / "server_batches" / f"batch_{batch_idx:05d}.csv"
df_table.to_csv(batch_path, index=False)
```

Then replace:

```python
if config.logging.wandb:
    wandb.log(...)
```

with:

```python
if use_wandb:
    wandb.log(...)
```

### 10.3 `PedagogicalRL/eval.py`

#### 10.3.1 Add imports if missing

Add:

```python
import json
from pathlib import Path
```

#### 10.3.2 Right after config merge in `main(cfg: EvalConfig)`

Right after:

```python
cfg = OmegaConf.merge(default_config, cfg)
```

insert:

```python
save_dir = Path(cfg.logging.save_dir)
eval_dir = save_dir / "eval_outputs"
eval_dir.mkdir(parents=True, exist_ok=True)

use_wandb = bool(
    hasattr(cfg, "logging")
    and cfg.logging.get("wandb", False)
    and os.getenv("WANDB_API_KEY")
)

if hasattr(cfg, "logging") and cfg.logging.get("wandb", False) and not use_wandb:
    logger.warning(
        "WANDB requested for eval but WANDB_API_KEY is missing; continuing with local-only eval logging."
    )
```

#### 10.3.3 Replace the stock W&B init block

Replace:

```python
if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
    wandb.init(...)
```

with:

```python
if use_wandb:
    wandb.init(...)
```

#### 10.3.4 After metrics are computed and `df_table` is ready

Right after `df_table` contains the reward columns and before any final W&B conversation logging, insert:

```python
metrics = {
    "delta_mean": delta_mean if cfg.recompute_initial_attempts else 0,
    "initial_rm_rewards_mean": initial_rm_mean if cfg.recompute_initial_attempts else 0,
    "end_rm_rewards_mean": end_rm_mean,
    "leaked_solutions_mean": leaked_mean,
    "rejects_pedagogical_values_mean": does_not_follow_mean,
}

if cfg.score_using_pedagogical_reward:
    metrics["pedagogical_reward_macro_avg"] = pedagogical_reward_macro_avg
    metrics["pedagogical_reward_micro_avg"] = pedagogical_reward_micro_avg

with open(eval_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=float)

df_table.astype(str).to_csv(eval_dir / "conversations.csv", index=False)
```

#### 10.3.5 Replace final W&B logging

Replace:

```python
if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
    wandb.log({...})
...
if cfg.logging.wandb:
    wandb.log({"conversations": wandb.Table(dataframe=df_table)})
    wandb.finish()
```

with:

```python
if use_wandb:
    wandb.log(metrics)
    wandb.log({"conversations": wandb.Table(dataframe=df_table.astype(str))})
    wandb.finish()
```

## 11) Add TutorRM config fields

Edit `PedagogicalRL/config/train_rl_model.py`.

Inside `GenerationConfig`, add:

```python
use_tutor_rm: bool = True
tutor_rm_model_name_or_path: str = "kpetyxova/towards-reward-modeling-tutors"
tutor_rm_max_length: int = 1024
tutor_rm_mode: str = "first_teacher_only"
```

Inside `TrainConfig`, add:

```python
use_gdpo: bool = False
gdpo_eps: float = 1e-4
reward_weights: list[float] = field(
    default_factory=lambda: [1.0, 0.25, 1.0, 1.0, 1.0]
)
```

The reward order is fixed as:

```text
[end_rm_reward, tutor_rm_reward, thinking_reward, end_of_conversation_reward, length_reward]
```

## 12) Add TutorRM implementation

### Target behavior
Add one new reward channel named `tutor_rm_reward`.

It must:
- return `0.0` unless the conversation type is `ATTEMPTED`
- score the tutor’s first reply after the student’s initial attempt
- Support `all_teacher_turns_mean` later for A3
- use the released HF model `kpetyxova/towards-reward-modeling-tutors`
- use the reward model exactly as a sequence classifier, not as a generative model

### 12.1 `PedagogicalRL/src/classroom.py`

Use the author-faithful load path:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
```

In `Classroom.__init__`, add:

```python
self.use_tutor_rm = generation_cfg.use_tutor_rm
self.tutor_rm_mode = generation_cfg.tutor_rm_mode
self.tutor_rm_max_length = generation_cfg.tutor_rm_max_length

if self.use_tutor_rm:
    self.tutor_rm_tokenizer = AutoTokenizer.from_pretrained(
        generation_cfg.tutor_rm_model_name_or_path,
        use_fast=False,
    )
    if self.tutor_rm_tokenizer.pad_token is None:
        self.tutor_rm_tokenizer.pad_token = self.tutor_rm_tokenizer.eos_token
    self.tutor_rm_tokenizer.padding_side = "right"
    self.tutor_rm_tokenizer.truncation_side = "left"

    self.tutor_rm_model = AutoModelForSequenceClassification.from_pretrained(
        generation_cfg.tutor_rm_model_name_or_path
    )
    self.tutor_rm_model.to(self.device)
    self.tutor_rm_model.eval()
```

Add these methods:

```python
def _serialize_tutor_rm_messages(self, conversation, teacher_response):
    hidden = conversation._get_hidden_conversation()
    student_msgs = [m["content"] for m in hidden if m["role"] == "student"]
    if len(student_msgs) == 0:
        return None

    user_content = (
        f"Problem: {conversation.problem}\n\n"
        f"Student attempt: {student_msgs[0]}\n\n"
        f"Gold solution: {conversation.answer}"
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": teacher_response},
    ]


def _score_single_tutor_rm(self, conversation) -> float:
    if conversation.type != ConversationType.ATTEMPTED:
        return 0.0

    hidden = conversation._get_hidden_conversation()
    teacher_msgs = [m["content"] for m in hidden if m["role"] == "teacher"]
    if len(teacher_msgs) == 0:
        return 0.0

    candidates = teacher_msgs if self.tutor_rm_mode == "all_teacher_turns_mean" else [teacher_msgs[0]]
    scores = []
    for teacher_response in candidates:
        messages = self._serialize_tutor_rm_messages(conversation, teacher_response)
        if messages is None:
            continue
        inputs = self.tutor_rm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.tutor_rm_max_length,
        ).to(self.tutor_rm_model.device)
        with torch.no_grad():
            logit = float(self.tutor_rm_model(inputs).logits.squeeze())
        scores.append(logit)

    return float(sum(scores) / len(scores)) if scores else 0.0


def get_tutor_rm_reward(self, conversations):
    if not self.use_tutor_rm:
        return [0.0 for _ in conversations]
    return [self._score_single_tutor_rm(c) for c in conversations]
```

Main path rules:
- use raw logits
- no clipping
- no `device_map="cuda:0"`
- no hardcoded TutorRM weight inside classroom

### 12.2 `PedagogicalRL/vllm_server.py`
Add endpoint:

```python
@app.post("/get_tutor_rm_reward")
```

It must mirror the existing reward endpoint pattern and return a **plain list**, not a JSON dict.

Extend the reward dataframe / reward collection to include `tutor_rm_reward`.
The stock `total_reward` in the server can stay as-is for logging; training itself must use the `reward_funcs` list order below.

### 12.3 `PedagogicalRL/src/vllm/client.py`
Add:

```python
def get_tutor_rm_reward(conversations, server_port: int = 8005):
    ...
```

It must call:

```text
http://localhost:{server_port}/get_tutor_rm_reward
```

and return the plain list from the response.

### 12.4 `PedagogicalRL/src/utils/utils.py`
Add:

```python
def construct_tutor_rm_reward_func(server_port: int = 8005):
    def tutor_rm_reward_func(completions, **kwargs):
        return get_tutor_rm_reward(completions, server_port=server_port)
    return tutor_rm_reward_func
```

### 12.5 `PedagogicalRL/train_rl.py`
Import the constructor and build reward funcs in this exact order:

```python
reward_funcs = [
    end_rm_reward,
    tutor_rm_reward,
    thinking_reward,
    end_of_conversation_reward,
    length_reward,
]
```

Pass `reward_weights=cfg.train.reward_weights` into `ClassroomGRPOConfig(...)`.

## 13) Add GDPO

### 13.1 `PedagogicalRL/src/grpo/config.py`
Add:

```python
reward_weights: Optional[list[float]] = field(
    default=None,
    metadata={"help": "Per-reward weights matching reward_funcs order."},
)

apply_gdpo: bool = field(
    default=False,
    metadata={"help": "Apply GDPO multi-reward normalization."},
)

gdpo_eps: float = field(
    default=1e-4,
    metadata={"help": "Stability epsilon used by GDPO normalization."},
)
```

### 13.2 `PedagogicalRL/src/grpo/trainer.py`
Right after `self.reward_funcs = reward_funcs`, add:

```python
if self.args.reward_weights is not None:
    if len(self.args.reward_weights) != len(self.reward_funcs):
        raise ValueError(
            f"reward_weights length {len(self.args.reward_weights)} != number of reward funcs {len(self.reward_funcs)}"
        )
    self.reward_weights = torch.tensor(self.args.reward_weights, dtype=torch.float32)
else:
    self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

self.apply_gdpo = bool(getattr(self.args, "apply_gdpo", False))
```

Replace the stock reward-to-advantage block with:

```python
rewards_per_func = gather(rewards_per_func)
rewards_per_func = rewards_per_func[(rewards_per_func != -199).any(dim=1)]
weights = self.reward_weights.to(device).unsqueeze(0)

if self.apply_gdpo and len(self.reward_weights) > 1:
    rewards_per_func_filter = torch.nan_to_num(rewards_per_func)

    all_reward_advantage = []
    for i in range(len(self.reward_weights)):
        reward_i = rewards_per_func_filter[:, i]

        each_reward_mean_grouped = reward_i.view(-1, self.num_generations).mean(dim=1)
        each_reward_std_grouped = reward_i.view(-1, self.num_generations).std(dim=1)

        each_reward_mean_grouped = each_reward_mean_grouped.repeat_interleave(
            self.num_generations, dim=0
        )
        each_reward_std_grouped = each_reward_std_grouped.repeat_interleave(
            self.num_generations, dim=0
        )

        each_reward_advantage = reward_i - each_reward_mean_grouped
        each_reward_advantage = each_reward_advantage / (
            each_reward_std_grouped + self.args.gdpo_eps
        )
        all_reward_advantage.append(each_reward_advantage)

    combined_reward_advantage = torch.stack(all_reward_advantage, dim=1)
    pre_bn_advantages = (combined_reward_advantage * weights).nansum(dim=1)
    bn_advantages_mean = pre_bn_advantages.mean()
    bn_advantages_std = pre_bn_advantages.std()
    advantages = (pre_bn_advantages - bn_advantages_mean) / (
        bn_advantages_std + self.args.gdpo_eps
    )

    rewards = (rewards_per_func * weights).nansum(dim=1)
else:
    rewards = (rewards_per_func * weights).nansum(dim=1)
    mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
        self.num_generations, dim=0
    )
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(
        self.num_generations, dim=0
    )
    advantages = rewards - mean_grouped_rewards
    if self.args.scale_rewards:
        advantages = advantages / (std_grouped_rewards + self.args.gdpo_eps)
```

### 13.3 `PedagogicalRL/train_rl.py`
When constructing `ClassroomGRPOConfig(...)`, pass:

```python
reward_weights=cfg.train.reward_weights,
apply_gdpo=cfg.train.use_gdpo,
gdpo_eps=cfg.train.gdpo_eps,
```

## 14) Generate reduced one-GPU configs

### 14.1 `config/deepspeed/zero3_1GPU.yaml`

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
python - <<'PY'
from pathlib import Path
import yaml
src = Path('config/deepspeed/zero3_4GPU.yaml')
dst = Path('config/deepspeed/zero3_1GPU.yaml')
cfg = yaml.safe_load(src.read_text())
cfg['num_processes'] = 1
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('wrote', dst)
PY
```

### 14.2 `config/train_rl/7b_tutorrm_grpo.yaml` and `7b_tutorrm_gdpo.yaml`

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
python - <<'PY'
from pathlib import Path
import copy
import yaml

base = yaml.safe_load(Path('config/train_rl/7b.yaml').read_text())

common = {
    'teacher_model': {
        'model_name_or_path': 'eth-nlped/TutorRL-7B',
        'vllm': {
            'number_of_gpus_per_instance': 1,
            'max_length': 4096,
            'max_num_seqs': 32,
            'gpu_memory_utilization': 0.22,
        },
    },
    'student_model': {
        'vllm': {
            'number_of_gpus_per_instance': 1,
            'max_length': 4096,
            'max_num_seqs': 32,
            'gpu_memory_utilization': 0.22,
        },
    },
    'judge_model': {
        'vllm': {
            'number_of_gpus_per_instance': 1,
            'max_length': 4096,
            'max_num_seqs': 32,
            'gpu_memory_utilization': 0.22,
        },
    },
    'train': {
        'number_of_problems_per_batch': 4,
        'num_samples_per_problem': 4,
        'per_device_train_batch_size': 1,
        'max_steps': 80,
        'gdpo_eps': 1e-4,
        'reward_weights': [1.0, 0.25, 1.0, 1.0, 1.0],
        'save_policy_to_disk_every_n': 5,
    },
    'generation': {
        'max_turns': 8,
        'max_tokens_in_conversation': 4096,
        'max_tokens_per_turn': 512,
        'max_tokens_per_student_attempt': 1024,
        'max_tokens_per_judge_attempt': 512,
        'number_student_attempts': 2,
        'number_judge_attempts': 1,
        'ignore_rejected_judge': True,
        'extra_penalty_for_rejected_judges': 0.75,
        'use_experimental_shared_memory': False,
        'use_tutor_rm': True,
        'tutor_rm_model_name_or_path': 'kpetyxova/towards-reward-modeling-tutors',
        'tutor_rm_max_length': 1024,
        'tutor_rm_mode': 'first_teacher_only',
    },
    'logging': {
        'wandb': False,
    },
}

def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v

for name, port, use_gdpo, save_dir in [
    ('7b_tutorrm_grpo.yaml', 8005, False, 'outputs/tutorrm_grpo'),
    ('7b_tutorrm_gdpo.yaml', 8006, True, 'outputs/tutorrm_gdpo'),
]:
    cfg = copy.deepcopy(base)
    deep_update(cfg, common)
    cfg['train']['use_gdpo'] = use_gdpo
    cfg['generation']['server_port'] = port
    cfg['logging']['save_dir'] = save_dir
    Path('config/train_rl', name).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('wrote', name)
PY
```

### 14.3 Validation

Run:

```bash
python - <<'PY'
from pathlib import Path
for p in [
    'config/deepspeed/zero3_1GPU.yaml',
    'config/train_rl/7b_tutorrm_grpo.yaml',
    'config/train_rl/7b_tutorrm_gdpo.yaml',
]:
    print('\n===', p, '===')
    print(Path(p).read_text()[:1200])
PY
```

Checklist:
- `zero3_1GPU.yaml` exists
- `7b_tutorrm_grpo.yaml` exists
- `7b_tutorrm_gdpo.yaml` exists
- GRPO config uses port `8005`
- GDPO config uses port `8006`
- both configs start from `eth-nlped/TutorRL-7B`
- both configs set `max_steps: 80`

## 15) Run A0 baselines

### 15.0 Mandatory benchmark server lifecycle rule

For every MathTutorBench run (A0a, A0b, A1 external, A2 external, A3 external), do not leave a vLLM benchmark server running across runs.
Always launch benchmark servers in the background with a PID file, wait on `/v1/models`, and kill that exact PID before starting the next benchmark server.

Create the PID directory once:

```bash
cd ~/tutor_gdpo_project/mathtutorbench
mkdir -p pids
```

### 15.1 TutorRL-7B baseline

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate
mkdir -p pids

vllm serve eth-nlped/TutorRL-7B \
  --served-model-name TutorRL-7B \
  --seed 42 \
  --tensor-parallel-size 1 > serve_TutorRL-7B.log 2>&1 &
echo $! > pids/TutorRL-7B.pid

until curl -fsS http://localhost:8000/v1/models >/dev/null; do
  sleep 5
done

/usr/bin/time -v ./run_mathtutorbench_suite.sh TutorRL-7B http://localhost:8000/v1 results 2>&1 | tee baseline_tutorrly7b_bench.log

kill "$(cat pids/TutorRL-7B.pid)" || true
sleep 5
kill -9 "$(cat pids/TutorRL-7B.pid)" 2>/dev/null || true
rm -f pids/TutorRL-7B.pid
```

### 15.2 TutorRL-7B-think baseline

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate
mkdir -p pids

vllm serve eth-nlped/TutorRL-7B-think \
  --served-model-name TutorRL-7B-think \
  --seed 42 \
  --tensor-parallel-size 1 > serve_TutorRL-7B-think.log 2>&1 &
echo $! > pids/TutorRL-7B-think.pid

until curl -fsS http://localhost:8000/v1/models >/dev/null; do
  sleep 5
done

/usr/bin/time -v ./run_mathtutorbench_suite.sh TutorRL-7B-think http://localhost:8000/v1 results 2>&1 | tee baseline_tutorrly7b_think_bench.log

kill "$(cat pids/TutorRL-7B-think.pid)" || true
sleep 5
kill -9 "$(cat pids/TutorRL-7B-think.pid)" 2>/dev/null || true
rm -f pids/TutorRL-7B-think.pid
```

### 15.3 Mandatory internal baseline evals

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate
mkdir -p logs
python eval.py \
  --config-name TutorRL.yaml \
  logging.save_dir=outputs/baseline_tutorrly7b_eval \
  logging.wandb=false \
  2>&1 | tee logs/internal_tutorrly7b.log
python eval.py \
  --config-name TutorRL-think.yaml \
  logging.save_dir=outputs/baseline_tutorrly7b_think_eval \
  logging.wandb=false \
  2>&1 | tee logs/internal_tutorrly7b_think.log
```

## 16) Run 5-step smoke tests

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
mkdir -p logs
source ~/tutor_gdpo_project/.venv/bin/activate
```

### 16.1 A1 smoke on GPU 0

```bash
export SERVER_PORT=8005
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_grpo.yaml \
  -- train.max_steps=5 \
  2>&1 | tee logs/smoke_grpo.log
```

### 16.2 A2 smoke on GPU 1

```bash
export SERVER_PORT=8006
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_gdpo.yaml \
  -- train.max_steps=5 \
  2>&1 | tee logs/smoke_gdpo.log
```

Success criteria:
- server starts
- training connects
- `/get_tutor_rm_reward` works
- checkpoint saves
- no port collision
- no shape error in GDPO

Checklist:
- [ ] A1 smoke passes
- [ ] A2 smoke passes
- [ ] no port collision occurs
- [ ] no global kill occurs
- [ ] TutorRM scoring path works
- [ ] GDPO branch runs without tensor-shape error

## 17) Run full A1 and A2

Start GPU logging first:

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 > logs/gpu0.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 > logs/gpu1.csv
```

### 17.1 A1 final

```bash
export SERVER_PORT=8005
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_grpo.yaml \
  2>&1 | tee logs/final_grpo.log
```

### 17.2 A2 final

```bash
export SERVER_PORT=8006
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_gdpo.yaml \
  2>&1 | tee logs/final_gdpo.log
```

## 18) Evaluate trained models internally

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

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

Collect these metrics for each trained model:
- delta solve rate
- leak solution
- Ped-RM micro
- Ped-RM macro

Checklist:
- [ ] A1 internal eval complete
- [ ] A2 internal eval complete
- [ ] metric summaries saved

## 19) Evaluate trained models externally on MathTutorBench

### 19.1 A1 external

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate
mkdir -p pids

vllm serve ~/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_grpo/model \
  --served-model-name tutorrm-grpo \
  --seed 42 \
  --tensor-parallel-size 1 > serve_tutorrm_grpo.log 2>&1 &
echo $! > pids/tutorrm-grpo.pid

until curl -fsS http://localhost:8000/v1/models >/dev/null; do
  sleep 5
done

/usr/bin/time -v ./run_mathtutorbench_suite.sh tutorrm-grpo http://localhost:8000/v1 results 2>&1 | tee tutorrm_grpo_bench.log

kill "$(cat pids/tutorrm-grpo.pid)" || true
sleep 5
kill -9 "$(cat pids/tutorrm-grpo.pid)" 2>/dev/null || true
rm -f pids/tutorrm-grpo.pid
```

### 19.2 A2 external

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate
mkdir -p pids

vllm serve ~/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_gdpo/model \
  --served-model-name tutorrm-gdpo \
  --seed 42 \
  --tensor-parallel-size 1 > serve_tutorrm_gdpo.log 2>&1 &
echo $! > pids/tutorrm-gdpo.pid

until curl -fsS http://localhost:8000/v1/models >/dev/null; do
  sleep 5
done

/usr/bin/time -v ./run_mathtutorbench_suite.sh tutorrm-gdpo http://localhost:8000/v1 results 2>&1 | tee tutorrm_gdpo_bench.log

kill "$(cat pids/tutorrm-gdpo.pid)" || true
sleep 5
kill -9 "$(cat pids/tutorrm-gdpo.pid)" 2>/dev/null || true
rm -f pids/tutorrm-gdpo.pid
```

Collect these metrics for each trained model:
- problem solving
- socratic questioning
- student solution correctness
- mistake location
- mistake correction
- scaffolding generation
- pedagogy following
- scaffolding generation hard
- pedagogy following hard

Checklist:
- [ ] A1 MathTutorBench eval complete
- [ ] A2 MathTutorBench eval complete
- [ ] scaffolding score computed for both
- [ ] result JSONs saved

## 20) Choose the winner and run LightEval once

Winner rule:
1. higher `mistake_correction`
2. if tied, higher mean over `mistake_correction`, `mistake_location`, `pedagogy_following`, `scaffolding_generation`
3. if still tied, lower train wall-clock time

cd ~/tutor_gdpo_project
source ~/tutor_gdpo_project/.venv/bin/activate

python aggregate_results.py --project-root ~/tutor_gdpo_project --output-dir ~/tutor_gdpo_project/summary_tables

eval "$(python - <<'PY'
import csv
from pathlib import Path

root = Path('~/tutor_gdpo_project').expanduser()

external = {}
with open(root / 'summary_tables' / 'results_external.csv', newline='') as f:
    for row in csv.DictReader(f):
        if row['model'] in {'TutorRM+GRPO', 'TutorRM+GDPO'}:
            external[row['model']] = row

eff = {}
with open(root / 'summary_tables' / 'results_efficiency.csv', newline='') as f:
    for row in csv.DictReader(f):
        eff[row['model']] = row

def tuple_score(row):
    primary = float(row['mistake_correction'])
    secondary = (
        float(row['mistake_correction']) +
        float(row['mistake_location']) +
        float(row['pedagogy_following']) +
        float(row['scaffolding_generation'])
    ) / 4.0
    return (primary, secondary)

candidates = ['TutorRM+GRPO', 'TutorRM+GDPO']
best = candidates[0]

for cand in candidates[1:]:
    if tuple_score(external[cand]) > tuple_score(external[best]):
        best = cand
    elif tuple_score(external[cand]) == tuple_score(external[best]):
        if float(eff[cand]['train_hours']) < float(eff[best]['train_hours']):
            best = cand

if best == 'TutorRM+GDPO':
    alias = 'tutorrm-gdpo'
    model = str(root / 'PedagogicalRL' / 'outputs' / 'tutorrm_gdpo' / 'model')
    config = '7b_tutorrm_gdpo.yaml'
else:
    alias = 'tutorrm-grpo'
    model = str(root / 'PedagogicalRL' / 'outputs' / 'tutorrm_grpo' / 'model')
    config = '7b_tutorrm_grpo.yaml'

print(f'export WINNER_LABEL=\"{best}\"')
print(f'export WINNER_ALIAS=\"{alias}\"')
print(f'export WINNER_MODEL=\"{model}\"')
print(f'export WINNER_CONFIG=\"{config}\"')
PY
)"

echo "Winner label: ${WINNER_LABEL}"
echo "Winner alias: ${WINNER_ALIAS}"
echo "Winner model: ${WINNER_MODEL}"
echo "Winner config: ${WINNER_CONFIG}"

Run LightEval only on the winner:

```bash
lighteval vllm \
  "model_name=${WINNER_MODEL},gpu_memory_utilization=0.85,max_model_length=4096,dtype=bfloat16,generation_parameters={max_new_tokens:2048,temperature:0.0}" \
  "lighteval|math_500|0|0,helm|mmlu|5|0,lighteval|gsm8k|4|0" \
  --use-chat-template
```

## 21) Aggregate final tables

```bash
cd ~/tutor_gdpo_project
source ~/tutor_gdpo_project/.venv/bin/activate
python aggregate_results.py --project-root ~/tutor_gdpo_project --output-dir ~/tutor_gdpo_project/summary_tables
```

Expected outputs:
- `summary_tables/results_internal.csv`
- `summary_tables/results_external.csv`
- `summary_tables/results_efficiency.csv`

### 21.1 Expected CSV schema

Copy from old `AGENTS.md`:

`results_internal.csv`

Columns:
- model
- delta_solve_rate
- leak_solution
- ped_rm_micro
- ped_rm_macro

Rows:
- TutorRL-7B
- TutorRL-7B-think
- TutorRM+GRPO
- TutorRM+GDPO

`results_external.csv`

Columns:
- model
- problem_solving
- socratic_questioning
- student_solution_correctness
- mistake_location
- mistake_correction
- scaffolding_generation
- pedagogy_following
- scaffolding_generation_hard
- pedagogy_following_hard

Rows:
- TutorRL-7B
- TutorRL-7B-think
- TutorRM+GRPO
- TutorRM+GDPO

`results_efficiency.csv`

Columns:
- model
- train_hours
- peak_gpu_mem_gb
- mathtutorbench_minutes
- avg_tutor_response_tokens

Rows:
- TutorRM+GRPO
- TutorRM+GDPO

### 21.2 Final write-up structure

Prepare a short report in this exact section order.

1. **Problem**
   - tutoring RL improves pedagogy but still has headroom on mistake remediation
   - adding a tutoring-specific reward creates a genuine multi-reward RL problem

2. **Base framework**
   - PedagogicalRL start checkpoint and multi-turn tutor–student–judge loop
   - MathTutorBench as external tutoring benchmark
   - released TutorRM reward model for mistake remediation

3. **Our changes**
   - add TutorRM reward to PedagogicalRL
   - compare stock summed-reward GRPO vs GDPO
   - first-teacher-only vs all-teacher-turns variant

4. **Implementation**
   - exact files edited
   - exact config values
   - parallel-safe launcher patch

5. **Results**
   - baseline rows
   - A1 vs A2 comparison
   - winner selection
   - efficiency table

6. **Sanity checks**
   - LightEval on the winner only

7. **Conclusion**
   - whether TutorRM helps
   - whether GDPO helps more than stock GRPO in this tutoring multi-reward setup

## 22) A3 only after A1 and A2 done, including full evaluation

Run A3 only after Sections 18, 19, 20, and 21 are complete and `WINNER_ALIAS`, `WINNER_MODEL`, and `WINNER_CONFIG` have already been set by Section 20.

### 22.1 Generate the A3 config from the winner config

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

: "${WINNER_ALIAS:?Run Section 20 first}"
: "${WINNER_CONFIG:?Run Section 20 first}"

python - <<'PY'
from pathlib import Path
import os
import yaml

root = Path('~/tutor_gdpo_project/PedagogicalRL').expanduser()
src = root / 'config' / 'train_rl' / os.environ['WINNER_CONFIG']
dst = root / 'config' / 'train_rl' / '7b_tutorrm_a3.yaml'

cfg = yaml.safe_load(src.read_text())
cfg['generation']['tutor_rm_mode'] = 'all_teacher_turns_mean'
cfg['generation']['server_port'] = 8007
cfg['train']['max_steps'] = 40
cfg['logging']['save_dir'] = f"outputs/{os.environ['WINNER_ALIAS'].replace('-', '_')}_a3"

dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {dst}")
PY
```

### 22.2 Run the A3 training job

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

export SERVER_PORT=8007
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_a3.yaml \
  2>&1 | tee logs/final_a3.log
```

### 22.3 Run the A3 external benchmark on the four target metrics only

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate
mkdir -p pids

export A3_ALIAS="${WINNER_ALIAS}-a3"
export A3_MODEL_PATH=~/tutor_gdpo_project/PedagogicalRL/outputs/${WINNER_ALIAS//-/_}_a3/model
export A3_BENCH_START=$(date +%s)

vllm serve "${A3_MODEL_PATH}" \
  --served-model-name "${A3_ALIAS}" \
  --seed 42 \
  --tensor-parallel-size 1 > "serve_${A3_ALIAS}.log" 2>&1 &
echo $! > "pids/${A3_ALIAS}.pid"

until curl -fsS http://localhost:8000/v1/models >/dev/null; do
  sleep 5
done

/usr/bin/time -v python main.py \
  --tasks mistake_location.yaml,mistake_correction.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=${A3_ALIAS},is_chat=True,temperature=0.0,max_tokens=1024 \
  2>&1 | tee "${A3_ALIAS}_nonped.log"

/usr/bin/time -v python main.py \
  --tasks scaffolding_generation.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=${A3_ALIAS},is_chat=True,temperature=0.0,max_tokens=1024 \
  2>&1 | tee "${A3_ALIAS}_scaffolding.log"

python reward_model/compute_scaffolding_score.py \
  --data_path "results/generations-${A3_ALIAS}-scaffolding_generation.json"

/usr/bin/time -v python main.py \
  --tasks pedagogy_following.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=${A3_ALIAS},is_chat=True,temperature=0.0,max_tokens=1024 \
  2>&1 | tee "${A3_ALIAS}_pedagogy.log"

python reward_model/compute_scaffolding_score.py \
  --data_path "results/generations-${A3_ALIAS}-pedagogy_following.json"

export A3_BENCH_END=$(date +%s)

python - <<'PY'
import json
import os
from pathlib import Path

alias = os.environ["A3_ALIAS"]
start = int(os.environ["A3_BENCH_START"])
end = int(os.environ["A3_BENCH_END"])

meta = {
    "model": alias,
    "tasks": [
        "mistake_location",
        "mistake_correction",
        "scaffolding_generation",
        "pedagogy_following",
    ],
    "elapsed_minutes": round((end - start) / 60.0, 4),
}

out = Path("results") / f"benchmark_meta-{alias}.json"
out.write_text(json.dumps(meta, indent=2))
print(f"wrote {out}")
PY

kill "$(cat pids/${A3_ALIAS}.pid)" || true
sleep 5
kill -9 "$(cat pids/${A3_ALIAS}.pid)" 2>/dev/null || true
rm -f "pids/${A3_ALIAS}.pid"
```

### 22.4 Expected A3 outputs

The A3 appendix run must leave behind at least:

```text
~/tutor_gdpo_project/PedagogicalRL/
  logs/
    final_a3.log
  outputs/
    tutorrm_grpo_a3/     or tutorrm_gdpo_a3/

~/tutor_gdpo_project/mathtutorbench/
  results/
    results-tutorrm-grpo-a3.yaml      or results-tutorrm-gdpo-a3.yaml
    benchmark_meta-tutorrm-grpo-a3.json or benchmark_meta-tutorrm-gdpo-a3.json
    generations-tutorrm-grpo-a3-mistake_location.json
    generations-tutorrm-grpo-a3-mistake_correction.json
    generations-tutorrm-grpo-a3-scaffolding_generation.json
    generations-tutorrm-grpo-a3-pedagogy_following.json
```

A3 is appendix-only by default. Do not let it overwrite the main A0/A1/A2 summary tables.

## 23) Required outputs the agent must leave behind

```text
~/tutor_gdpo_project/
  env_versions.json
  summary_tables/
    results_internal.csv
    results_external.csv
    results_efficiency.csv

~/tutor_gdpo_project/PedagogicalRL/
  logs/
    smoke_grpo.log
    smoke_gdpo.log
    final_grpo.log
    final_gdpo.log
    internal_tutorrly7b.log
    internal_tutorrly7b_think.log
    internal_tutorrm_grpo.log
    internal_tutorrm_gdpo.log
    final_a3.log
    gpu0.csv
    gpu1.csv
  outputs/
    tutorrm_grpo/
    tutorrm_gdpo/
    tutorrm_grpo_a3/     or tutorrm_gdpo_a3/
    tutorrm_grpo/eval_outputs/metrics.json
    tutorrm_grpo/eval_outputs/conversations.csv
    tutorrm_gdpo/eval_outputs/metrics.json
    tutorrm_gdpo/eval_outputs/conversations.csv

~/tutor_gdpo_project/mathtutorbench/
  results/
    results-TutorRL-7B.yaml
    results-TutorRL-7B-think.yaml
    results-tutorrm-grpo.yaml
    results-tutorrm-gdpo.yaml
    results-tutorrm-grpo-a3.yaml      or results-tutorrm-gdpo-a3.yaml
    benchmark_meta-TutorRL-7B.json
    benchmark_meta-TutorRL-7B-think.json
    benchmark_meta-tutorrm-grpo.json
    benchmark_meta-tutorrm-gdpo.json
    benchmark_meta-tutorrm-grpo-a3.json or benchmark_meta-tutorrm-gdpo-a3.json
    generations-TutorRL-7B-*.json
    generations-TutorRL-7B-think-*.json
    generations-tutorrm-grpo-*.json
    generations-tutorrm-gdpo-*.json
    generations-tutorrm-grpo-a3-mistake_location.json      or generations-tutorrm-gdpo-a3-mistake_location.json
    generations-tutorrm-grpo-a3-mistake_correction.json    or generations-tutorrm-gdpo-a3-mistake_correction.json
    generations-tutorrm-grpo-a3-scaffolding_generation.json or generations-tutorrm-gdpo-a3-scaffolding_generation.json
    generations-tutorrm-grpo-a3-pedagogy_following.json    or generations-tutorrm-gdpo-a3-pedagogy_following.json
```

## 24) One-screen execution checklist

- clone the three repos
- copy the six helper files into place
- create `.env`
- run `setup_env.sh`
- patch launcher scripts
- patch local logging fallback
- add TutorRM config + implementation + endpoint/client/wrapper + reward list
- add GDPO config + trainer branch + config wiring
- generate reduced configs
- run A0a and A0b
- run A1 and A2 smoke tests
- run A1 and A2 final training
- run internal evals
- run external evals
- run LightEval on winner only
- run `aggregate_results.py`
- leave behind summary tables and logs
