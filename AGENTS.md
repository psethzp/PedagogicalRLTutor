# AGENTS.md — Exact end-to-end implementation brief for TutorRM + GDPO on PedagogicalRL

## 0) Non-negotiable project choices

This project is fixed as follows.

- **Main training repo:** `eth-lre/PedagogicalRL`
- **Start checkpoint for training:** `eth-nlped/TutorRL-7B`
- **External benchmark repo:** `eth-lre/mathtutorbench`
- **Extra reward model repo:** `Kpetyxova/Towards_Reward_Modeling_for_Tutors`
- **Released tutor reward model:** `kpetyxova/towards-reward-modeling-tutors`
- **Main algorithmic comparison:**
  - **A1:** TutorRM + stock PedagogicalRL summed-reward GRPO
  - **A2:** TutorRM + GDPO advantage normalization
- **Baselines:**
  - **A0a:** `eth-nlped/TutorRL-7B` on MathTutorBench
  - **A0b:** `eth-nlped/TutorRL-7B-think` on MathTutorBench
- **Optional cheap add-on:**
  - **A3:** first-teacher-reply TutorRM vs all-teacher-turns-mean TutorRM, only after A1 and A2 are fully complete
- **Hardware assumption for the implementation:** two independent 1-GPU runs in parallel, one run per GPU
- **Training strategy:** short continuation training from the released TutorRL checkpoint, not full reproduction from raw Qwen

Do **not** redesign the project. Do **not** substitute another repo. Do **not** retrain the tutor reward model. Do **not** try to reproduce the full PedagogicalRL paper budget.

---

## 1) Final experiment set

Run exactly these experiments.

### A0 — Baselines
1. `eth-nlped/TutorRL-7B` on full MathTutorBench
2. `eth-nlped/TutorRL-7B-think` on full MathTutorBench

### A1 — TutorRM + stock GRPO
Continue `eth-nlped/TutorRL-7B` inside PedagogicalRL after adding the tutor-specific remediation reward. Keep the existing PedagogicalRL reward aggregation logic, which sums reward channels before grouped normalization.

### A2 — TutorRM + GDPO
Same as A1, but replace the stock summed-reward GRPO advantage computation with GDPO-style per-reward normalization before combination.

### A3 — Optional only if A1 and A2 finish early
Repeat only the better of A1/A2 with:
- `tutor_rm_mode = first_teacher_only`
- `tutor_rm_mode = all_teacher_turns_mean`

Use A3 only if A1 and A2, including evaluation, are already finished.

---

## 2) Top-level workflow summary

1. Clone the three repos.
2. Create one Python 3.11 virtual environment.
3. Install the three repos’ requirements into that single venv.
4. Patch PedagogicalRL launcher scripts so two ablations can run in parallel without killing each other.
5. Add the tutor-specific reward path to PedagogicalRL.
6. Add GDPO flags and trainer logic to PedagogicalRL.
7. Generate three new config files:
   - `config/deepspeed/zero3_1GPU.yaml`
   - `config/train_rl/7b_tutorrm_grpo.yaml`
   - `config/train_rl/7b_tutorrm_gdpo.yaml`
8. Run A0 baselines on MathTutorBench.
9. Run 5-step smoke tests for A1 and A2.
10. Run the full A1 and A2 continuation jobs in parallel.
11. Evaluate both trained models with:
    - PedagogicalRL internal eval
    - MathTutorBench external eval
    - LightEval sanity on the winner only
12. Export one final result table and one efficiency table.

---

## 3) Exact repo layout to create

```bash
mkdir -p ~/tutor_gdpo_project
cd ~/tutor_gdpo_project

git clone https://github.com/eth-lre/PedagogicalRL.git
git clone https://github.com/eth-lre/mathtutorbench.git
git clone https://github.com/Kpetyxova/Towards_Reward_Modeling_for_Tutors.git
```

Expected layout:

```text
~/tutor_gdpo_project/
  PedagogicalRL/
  mathtutorbench/
  Towards_Reward_Modeling_for_Tutors/
```

Checklist:
- [ ] All three repos clone successfully
- [ ] `PedagogicalRL/train_rl.py` exists
- [ ] `mathtutorbench/main.py` exists
- [ ] `Towards_Reward_Modeling_for_Tutors/inference.py` exists

---

## 4) Exact environment setup

Use a venv, not conda.

```bash
cd ~/tutor_gdpo_project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

pip install -r PedagogicalRL/requirements.txt
pip install -r mathtutorbench/requirements.txt
pip install -r Towards_Reward_Modeling_for_Tutors/requirements.txt
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
- [ ] Venv created
- [ ] All requirements installed without unresolved conflicts
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `import trl` works
- [ ] `import fastapi` works

---

## 5) Mandatory Stage 1 patch — make the launcher parallel-safe

This stage is mandatory.

### Why this stage exists
PedagogicalRL’s stock launcher hardcodes `http://localhost:8005/docs` when waiting for the server, and `stop_vllm_server.sh` uses global `pkill` on `uvicorn`, `multiprocess.spawn`, and `vllm_server.py`. That means two ablations running at once will collide and kill each other.

### Files to edit
- `PedagogicalRL/stop_vllm_server.sh`
- `PedagogicalRL/start_vllm_server.sh`
- `PedagogicalRL/start_rl_training.sh`

### 5.1) Replace `stop_vllm_server.sh`

Overwrite the file with this exact content:

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

### 5.2) Replace `start_vllm_server.sh`

Overwrite the file with this exact content:

```bash
#!/usr/bin/env bash
set -euo pipefail

./stop_vllm_server.sh
exec python vllm_server.py "$@"
```

Then:

```bash
chmod +x PedagogicalRL/start_vllm_server.sh
```

### 5.3) Patch `start_rl_training.sh`

Edit the script so it does **all** of the following:

1. Reads:

```bash
SERVER_PORT="${SERVER_PORT:-8005}"
```

2. After starting the background server, writes the PID file:

```bash
./start_vllm_server.sh "${SERVER_ARGS[@]}" &
SERVER_PID=$!
echo "${SERVER_PID}" > ".vllm_${SERVER_PORT}.pid"
```

3. Replaces the hardcoded wait loop URL:

```bash
until curl --output /dev/null --silent --head --fail "http://localhost:${SERVER_PORT}/docs"; do
  sleep 2
done
```

4. Leaves the rest of the accelerate launch logic intact.

Do **not** leave any hardcoded `8005` in that wait loop.

### 5.4) Validate the launcher patch

From `~/tutor_gdpo_project/PedagogicalRL`, run:

```bash
grep -n "localhost:" start_rl_training.sh
grep -n "pkill" stop_vllm_server.sh || true
```

Expected result:
- `start_rl_training.sh` should reference `localhost:${SERVER_PORT}`
- `stop_vllm_server.sh` should contain **no** `pkill`

Checklist:
- [ ] `stop_vllm_server.sh` is PID-file based only
- [ ] `start_vllm_server.sh` uses `exec python vllm_server.py "$@"`
- [ ] `start_rl_training.sh` waits on `localhost:${SERVER_PORT}`
- [ ] No hardcoded `8005` remains in the wait loop
- [ ] No global `pkill` remains

---

## 6) Mandatory Stage 2 patch — add TutorRM to PedagogicalRL

### Target behavior
Add one new reward channel named `tutor_rm_reward`.

It must:
- return `0.0` unless the conversation type is `ATTEMPTED`
- score the tutor’s first reply after the student’s initial attempt
- optionally support `all_teacher_turns_mean` later for A3
- use the released HF model `kpetyxova/towards-reward-modeling-tutors`
- use the reward model exactly as a sequence classifier, not as a generative model

### 6.1) Edit `config/train_rl_model.py`

Inside `GenerationConfig`, add these exact fields:

```python
use_tutor_rm: bool = True
tutor_rm_model_name_or_path: str = "kpetyxova/towards-reward-modeling-tutors"
tutor_rm_max_length: int = 1024
tutor_rm_mode: str = "first_teacher_only"
```

Inside `TrainConfig`, add these exact fields:

```python
use_gdpo: bool = False
gdpo_eps: float = 1e-4
reward_weights: list[float] = field(
    default_factory=lambda: [1.0, 0.5, 1.0, 1.0, 1.0]
)
```

If `field` is not imported, import it from `dataclasses`.

Interpretation of `reward_weights`:

```text
[end_rm_reward, tutor_rm_reward, thinking_reward, end_of_conversation_reward, length_reward]
```

### 6.2) Edit `src/classroom.py`

#### Imports
Add the missing import if needed:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
```

#### In `Classroom.__init__`
After the existing reward model setup, add TutorRM setup:

```python
self.use_tutor_rm = generation_cfg.use_tutor_rm
self.tutor_rm_mode = generation_cfg.tutor_rm_mode
self.tutor_rm_max_length = generation_cfg.tutor_rm_max_length

if self.use_tutor_rm:
    self.tutor_rm_tokenizer = AutoTokenizer.from_pretrained(
        generation_cfg.tutor_rm_model_name_or_path,
        use_fast=False,
    )
    self.tutor_rm_tokenizer.padding_side = "right"
    self.tutor_rm_tokenizer.truncation_side = "left"

    self.tutor_rm_model = AutoModelForSequenceClassification.from_pretrained(
        generation_cfg.tutor_rm_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    self.tutor_rm_model.eval()
```

#### Add helper methods
Add these two methods inside `Classroom`.

```python
def _score_single_tutor_rm(self, conversation) -> float:
    if conversation.type != ConversationType.ATTEMPTED:
        return 0.0

    hidden = conversation._get_hidden_conversation()
    student_msgs = [m["content"] for m in hidden if m["role"] == "student"]
    teacher_msgs = [m["content"] for m in hidden if m["role"] == "teacher"]

    if len(student_msgs) == 0 or len(teacher_msgs) == 0:
        return 0.0

    student_attempt = student_msgs[0]
    if self.tutor_rm_mode == "all_teacher_turns_mean":
        candidate_teacher_responses = teacher_msgs
    else:
        candidate_teacher_responses = [teacher_msgs[0]]

    scores = []
    for teacher_response in candidate_teacher_responses:
        user_content = (
            f"Problem: {conversation.problem}\n\n"
            f"Student attempt: {student_attempt}\n\n"
            f"Gold Solution: {conversation.answer}"
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": teacher_response},
        ]
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

        # chosen implementation detail: clip into [-1, 1] so the new reward
        # stays on a controlled scale before trainer weighting
        scores.append(max(-1.0, min(1.0, logit)))

    return float(sum(scores) / len(scores))


def get_tutor_rm_reward(self, conversations):
    if not self.use_tutor_rm:
        return [0.0 for _ in conversations]
    return [self._score_single_tutor_rm(c) for c in conversations]
```

### 6.3) Edit `vllm_server.py`

Mirror the existing reward endpoint pattern.

Add a new endpoint named:

```python
@app.post("/get_tutor_rm_reward")
```

It must:
- parse the incoming conversation batch exactly like the existing reward endpoints
- call `classroom.get_tutor_rm_reward(conversations)`
- return a JSON dict with the list of rewards

Then extend the server-side reward dataframe so it now includes:
- `end_rm_reward`
- `tutor_rm_reward`
- `thinking_reward`
- `end_of_conversation_reward`
- `length_reward`

And define:

```python
total_reward = (
    end_rm_reward
    + tutor_rm_reward
    + thinking_reward
    + end_of_conversation_reward
    + length_reward
)
```

### 6.4) Edit `src/vllm/client.py`

Add a client wrapper that mirrors the style of the existing reward RPC functions:

```python
def get_tutor_rm_reward(conversations: List[str], server_port: int = 8005):
    ...
```

It must hit:

```text
http://localhost:{server_port}/get_tutor_rm_reward
```

### 6.5) Edit `src/utils/utils.py`

Add a wrapper constructor in the same style as the existing reward constructors:

```python
def construct_tutor_rm_reward_func(server_port: int = 8005):
    def tutor_rm_reward_func(completions, **kwargs):
        return get_tutor_rm_reward(completions, server_port=server_port)
    return tutor_rm_reward_func
```

### 6.6) Edit `train_rl.py`

1. Import the new reward constructor.
2. Instantiate it with `cfg.generation.server_port`.
3. Insert it into the reward list in this exact order:

```python
reward_funcs = [
    end_rm_reward,
    tutor_rm_reward,
    thinking_reward,
    end_of_conversation_reward,
    length_reward,
]
```

Checklist:
- [ ] `GenerationConfig` has TutorRM fields
- [ ] `TrainConfig` has `use_gdpo`, `gdpo_eps`, `reward_weights`
- [ ] `Classroom` loads the TutorRM classifier
- [ ] `Classroom.get_tutor_rm_reward(...)` exists
- [ ] `vllm_server.py` exposes `/get_tutor_rm_reward`
- [ ] `src/vllm/client.py` has a TutorRM client wrapper
- [ ] `src/utils/utils.py` has `construct_tutor_rm_reward_func`
- [ ] `train_rl.py` includes TutorRM in `reward_funcs`

---

## 7) Mandatory Stage 3 patch — add GDPO to the trainer

### 7.1) Edit `src/grpo/config.py`

Add these config fields if they are not already present:

```python
reward_weights: Optional[list[float]] = field(
    default=None,
    metadata={"help": "Per-reward weights matching reward_funcs order."},
)

apply_gdpo: bool = field(
    default=False,
    metadata={"help": "Apply GDPO multi-reward normalization."},
)
```

If the file already contains a similar field name, reuse the same name consistently. The implementation below assumes `apply_gdpo` and `reward_weights` are the active names.

### 7.2) Edit `src/grpo/trainer.py`

#### In `__init__`
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

#### In the reward-to-advantage block
Replace the stock block that currently:
- gathers `rewards_per_func`
- sums across reward functions
- computes grouped mean and std on the summed scalar rewards
- forms `advantages`

with this exact two-branch logic.

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

    pre_bn_advantages = (
        combined_reward_advantage * weights
    ).nansum(dim=1)

    bn_advantages_mean = pre_bn_advantages.mean()
    bn_advantages_std = pre_bn_advantages.std()

    advantages = (
        pre_bn_advantages - bn_advantages_mean
    ) / (bn_advantages_std + self.args.gdpo_eps)

    # keep raw weighted total reward for logging
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

#### In trainer config wiring
Make sure `ClassroomGRPOConfig(...)` receives:
- `reward_weights=cfg.train.reward_weights`
- `apply_gdpo=cfg.train.use_gdpo`
- `gdpo_eps=cfg.train.gdpo_eps`

If `ClassroomGRPOConfig` is constructed inside `train_rl.py`, pass those fields there.

Checklist:
- [ ] `src/grpo/config.py` accepts `reward_weights`
- [ ] `src/grpo/config.py` accepts `apply_gdpo`
- [ ] `src/grpo/trainer.py` stores `self.reward_weights`
- [ ] `src/grpo/trainer.py` stores `self.apply_gdpo`
- [ ] Trainer has a stock branch and a GDPO branch
- [ ] `train_rl.py` passes `reward_weights`, `apply_gdpo`, `gdpo_eps`

---

## 8) Create the reduced one-GPU configs

### 8.1) Create `config/deepspeed/zero3_1GPU.yaml`

Generate it from the stock 4-GPU config.

Run this exact script:

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
python - <<'PY'
from pathlib import Path
import yaml

src = Path('config/deepspeed/zero3_4GPU.yaml')
dst = Path('config/deepspeed/zero3_1GPU.yaml')
text = src.read_text()
# source file is one-line YAML; safe_load handles it
cfg = yaml.safe_load(text)
cfg['num_processes'] = 1
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('wrote', dst)
PY
```

### 8.2) Create `config/train_rl/7b_tutorrm_grpo.yaml` and `7b_tutorrm_gdpo.yaml`

Run this exact script:

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
        'reward_weights': [1.0, 0.5, 1.0, 1.0, 1.0],
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

### 8.3) Validation

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
- [ ] `zero3_1GPU.yaml` exists
- [ ] `7b_tutorrm_grpo.yaml` exists
- [ ] `7b_tutorrm_gdpo.yaml` exists
- [ ] GRPO config uses port `8005`
- [ ] GDPO config uses port `8006`
- [ ] both configs start from `eth-nlped/TutorRL-7B`
- [ ] both configs set `max_steps: 80`

---

## 9) Run A0 baselines before any training

### 9.1) Baseline A0a — TutorRL-7B

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate

vllm serve eth-nlped/TutorRL-7B \
  --seed 42 \
  --tensor-parallel-size 1
```

In a second shell:

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate

python main.py \
  --tasks problem_solving.yaml,socratic_questioning.yaml,student_solution_correctness.yaml,mistake_location.yaml,mistake_correction.yaml,scaffolding_generation.yaml,pedagogy_following.yaml,scaffolding_generation_hard.yaml,pedagogy_following_hard.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=eth-nlped/TutorRL-7B,is_chat=True,temperature=0.0,max_tokens=1024

python reward_model/compute_scaffolding_score.py \
  --data_path results/generations-eth-nlped-TutorRL-7B.json
```

Stop the server.

### 9.2) Baseline A0b — TutorRL-7B-think

```bash
vllm serve eth-nlped/TutorRL-7B-think \
  --seed 42 \
  --tensor-parallel-size 1
```

In a second shell:

```bash
python main.py \
  --tasks problem_solving.yaml,socratic_questioning.yaml,student_solution_correctness.yaml,mistake_location.yaml,mistake_correction.yaml,scaffolding_generation.yaml,pedagogy_following.yaml,scaffolding_generation_hard.yaml,pedagogy_following_hard.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=eth-nlped/TutorRL-7B-think,is_chat=True,temperature=0.0,max_tokens=1024

python reward_model/compute_scaffolding_score.py \
  --data_path results/generations-eth-nlped-TutorRL-7B-think.json
```

Optional but recommended internal baseline eval:

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

python eval.py --config-name TutorRL.yaml logging.wandb=false
python eval.py --config-name TutorRL-think.yaml logging.wandb=false
```

Checklist:
- [ ] A0a MathTutorBench results exist
- [ ] A0b MathTutorBench results exist
- [ ] scaffolding score computed for both baselines
- [ ] optional internal baseline evals completed

---

## 10) Run 5-step smoke tests for A1 and A2

Before smoke tests, create logs dir.

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
mkdir -p logs
```

### 10.1) Smoke A1 — TutorRM + stock GRPO on GPU 0

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

export SERVER_PORT=8005
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_grpo.yaml \
  -- train.max_steps=5 \
  2>&1 | tee logs/smoke_grpo.log
```

### 10.2) Smoke A2 — TutorRM + GDPO on GPU 1

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

export SERVER_PORT=8006
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_gdpo.yaml \
  -- train.max_steps=5 \
  2>&1 | tee logs/smoke_gdpo.log
```

### 10.3) Smoke-test success criteria

Both smoke logs must show all of the following:
- server starts successfully
- trainer connects successfully
- reward endpoint `/get_tutor_rm_reward` is called without error
- at least one checkpoint save occurs
- process exits cleanly

Checklist:
- [ ] A1 smoke passes
- [ ] A2 smoke passes
- [ ] no port collision occurs
- [ ] no global kill occurs
- [ ] TutorRM scoring path works
- [ ] GDPO branch runs without tensor-shape error

---

## 11) Run full A1 and A2 in parallel

### 11.1) Optional GPU usage logging

Start these in separate shells before training:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 > logs/gpu0.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 > logs/gpu1.csv
```

### 11.2) Final A1 — TutorRM + stock GRPO

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

export SERVER_PORT=8005
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_grpo.yaml \
  2>&1 | tee logs/final_grpo.log
```

### 11.3) Final A2 — TutorRM + GDPO

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

export SERVER_PORT=8006
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v ./start_rl_training.sh \
  --config_file config/deepspeed/zero3_1GPU.yaml \
  --num_processes 1 \
  --config-name 7b_tutorrm_gdpo.yaml \
  2>&1 | tee logs/final_gdpo.log
```

Checklist:
- [ ] A1 full run completes
- [ ] A2 full run completes
- [ ] checkpoints written to `outputs/tutorrm_grpo` and `outputs/tutorrm_gdpo`
- [ ] logs captured
- [ ] GPU usage logs captured

---

## 12) Evaluate both trained models internally

### 12.1) A1 internal eval

```bash
cd ~/tutor_gdpo_project/PedagogicalRL
source ~/tutor_gdpo_project/.venv/bin/activate

python eval.py \
  --config-name TutorRL.yaml \
  teacher_model.model_name_or_path=outputs/tutorrm_grpo/model \
  logging.wandb=false
```

### 12.2) A2 internal eval

```bash
python eval.py \
  --config-name TutorRL.yaml \
  teacher_model.model_name_or_path=outputs/tutorrm_gdpo/model \
  logging.wandb=false
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

---

## 13) Evaluate both trained models externally on MathTutorBench

### 13.1) A1 external eval

```bash
cd ~/tutor_gdpo_project/mathtutorbench
source ~/tutor_gdpo_project/.venv/bin/activate

vllm serve ~/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_grpo/model \
  --seed 42 \
  --tensor-parallel-size 1
```

In a second shell:

```bash
python main.py \
  --tasks problem_solving.yaml,socratic_questioning.yaml,student_solution_correctness.yaml,mistake_location.yaml,mistake_correction.yaml,scaffolding_generation.yaml,pedagogy_following.yaml,scaffolding_generation_hard.yaml,pedagogy_following_hard.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=tutorrm-grpo,is_chat=True,temperature=0.0,max_tokens=1024

python reward_model/compute_scaffolding_score.py \
  --data_path results/generations-tutorrm-grpo.json
```

### 13.2) A2 external eval

```bash
vllm serve ~/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_gdpo/model \
  --seed 42 \
  --tensor-parallel-size 1
```

In a second shell:

```bash
python main.py \
  --tasks problem_solving.yaml,socratic_questioning.yaml,student_solution_correctness.yaml,mistake_location.yaml,mistake_correction.yaml,scaffolding_generation.yaml,pedagogy_following.yaml,scaffolding_generation_hard.yaml,pedagogy_following_hard.yaml \
  --provider completion_api \
  --model_args base_url=http://localhost:8000/v1,model=tutorrm-gdpo,is_chat=True,temperature=0.0,max_tokens=1024

python reward_model/compute_scaffolding_score.py \
  --data_path results/generations-tutorrm-gdpo.json
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

---

## 14) Choose the winner and run LightEval sanity only once

Pick the winner by this exact ranking rule:

1. higher `mistake_correction`
2. if tied, higher mean over:
   - mistake_correction
   - mistake_location
   - pedagogy_following
   - scaffolding_generation
3. if still tied, lower train wall-clock time

Suppose the winner path is stored in:

```bash
export WINNER_MODEL=~/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_gdpo/model
```

Run LightEval only on the winner:

```bash
lighteval vllm \
  "model_name=${WINNER_MODEL},gpu_memory_utilization=0.85,max_model_length=4096,dtype=bfloat16,generation_parameters={max_new_tokens:2048,temperature:0.0}" \
  "lighteval|math_500|0|0,helm|mmlu|5|0,lighteval|gsm8k|4|0" \
  --use-chat-template
```

Checklist:
- [ ] winner selected using the exact rule above
- [ ] winner LightEval complete
- [ ] sanity metrics saved

---

## 15) A3 only if there is time

Run A3 only after A1 and A2 are fully complete and evaluated.

### A3 change
Create one extra config from the winner config with:

```yaml
generation:
  tutor_rm_mode: all_teacher_turns_mean
```

Everything else stays identical.

Run exactly one additional 40-step continuation from the same base checkpoint `eth-nlped/TutorRL-7B` and evaluate only on:
- mistake_correction
- mistake_location
- pedagogy_following
- scaffolding_generation

Do **not** run full A3 if A1/A2 already consumed the available time budget.

---

## 16) Required output artifacts

The agent must leave behind these files/directories:

```text
~/tutor_gdpo_project/PedagogicalRL/
  logs/
    smoke_grpo.log
    smoke_gdpo.log
    final_grpo.log
    final_gdpo.log
    gpu0.csv
    gpu1.csv
  outputs/
    tutorrm_grpo/
    tutorrm_gdpo/

~/tutor_gdpo_project/mathtutorbench/
  results/
    generations-eth-nlped-TutorRL-7B.json
    generations-eth-nlped-TutorRL-7B-think.json
    generations-tutorrm-grpo.json
    generations-tutorrm-gdpo.json
```

And the agent must create these summary tables as CSV:

### `results_internal.csv`
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

### `results_external.csv`
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

### `results_efficiency.csv`
Columns:
- model
- train_hours
- peak_gpu_mem_gb
- mathtutorbench_minutes
- avg_tutor_response_tokens

Rows:
- TutorRM+GRPO
- TutorRM+GDPO

---

## 17) Final write-up structure the agent must prepare

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
   - optional first-teacher-only vs all-teacher-turns variant

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

---

## 18) One-screen execution checklist

### Setup
- [ ] clone the three repos
- [ ] create Python 3.11 venv
- [ ] install all requirements

### Patch launcher
- [ ] replace `stop_vllm_server.sh`
- [ ] replace `start_vllm_server.sh`
- [ ] patch `start_rl_training.sh`

### Patch training repo
- [ ] add TutorRM config fields
- [ ] add TutorRM model loading in `Classroom`
- [ ] add TutorRM reward endpoint/client/wrapper
- [ ] add TutorRM to `reward_funcs`
- [ ] add GDPO config fields
- [ ] add GDPO logic in trainer

### Configs
- [ ] generate `zero3_1GPU.yaml`
- [ ] generate `7b_tutorrm_grpo.yaml`
- [ ] generate `7b_tutorrm_gdpo.yaml`

### Runs
- [ ] A0a TutorRL-7B benchmark
- [ ] A0b TutorRL-7B-think benchmark
- [ ] A1 smoke
- [ ] A2 smoke
- [ ] A1 final
- [ ] A2 final
- [ ] internal evals
- [ ] external evals
- [ ] LightEval on winner

### Outputs
- [ ] logs saved
- [ ] checkpoints saved
- [ ] CSV result tables created
- [ ] final report outline created

