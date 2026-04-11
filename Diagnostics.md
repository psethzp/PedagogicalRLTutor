Yes. For a **workflow smoke**, stop trying to make `vllm==0.8.3` work on that pre-Ampere GPU right now. Your present failure is in the inference backend path, but there is a second blocker waiting behind it anyway: upstream `train_rl.py` still hardcodes `torch.bfloat16`, `attn_implementation="flash_attention_2"`, and `bf16=True`, while `start_rl_training.sh` waits on `/docs`, and `vllm_server.py` only reaches `uvicorn.run(...)` **after** `Classroom(...)` has already been created. So even a perfect vLLM bypass on the server side would not, by itself, make A1/A2 train on CPU or an old GPU. ([GitHub][1])

The repo is still very mockable at the right seams. `MathTutorBench` accepts **any OpenAI-type API** through `completion_api`; PedagogicalRL’s own client is just plain HTTP over `/sample_conversations`, reward endpoints, and `/wait_batch`; and TRL’s GRPO trainer supports **custom reward functions** and even a custom `rollout_func`. That means you can validate almost all orchestration and most of the GDPO/GRPO wiring without real vLLM and without real 7B weights. ([GitHub][2])

The right split is:

1. **Pure plumbing smoke** for A0/A3 and benchmark orchestration: no real LLMs at all.
2. **Algorithm smoke** for A1/A2: tiny CPU causal LM for the optimizer path, but mocked rollout/inference and mocked TutorRM endpoint.
3. **Skip or stub Stage 6** in the no-model smoke, because your own runbook already treats LightEval as a separate, winner-only sanity stage in a separate env after winner selection.  

## What to mock, exactly

### A0a / A0b / A3 external benchmark smoke

Do **not** use `vllm serve`. Run a tiny **mock OpenAI-compatible server** instead.

`MathTutorBench`’s README examples show `--provider completion_api --model_args base_url=http://localhost:8000/v1,model=...`, and it explicitly says `completion_api` supports any OpenAI-type API. So your mock server only needs these endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions    # include both, safest
```

and response bodies in standard OpenAI-style JSON. ([GitHub][2])

Use **deterministic pseudo-random text**, keyed by prompt hash, not true randomness. That gives reproducibility and still exercises downstream scoring, file writing, and aggregation. For example:

* `mistake_location`: “The first error is in line 2: sign mistake.”
* `mistake_correction`: “Try distributing the minus sign before combining terms.”
* `scaffolding_generation` / `pedagogy_following`: short tutoring-style responses with one question plus one hint.

This validates:

* benchmark provider wiring,
* `/v1/models` polling,
* task loop execution,
* `compute_scaffolding_score.py`,
* result JSON/YAML writing,
* `aggregate_results.py`,
* winner-selection plumbing.  

It does **not** validate vLLM, CUDA, xformers, or model loading.

### A1 / A2 / internal eval smoke

Do **not** create a separate fake server for PedagogicalRL first. The cleaner seam is lower down: mock `ParallelvLLMInference`, because both `vllm_server.py` **and** `eval.py` instantiate `Classroom`, and `Classroom` imports `ParallelvLLMInference` from `src.vllm.data_parallel_vllm`. That class exposes a very small interface: constructor, `run_batch`, `sleep`, and `cleanup`. ([GitHub][3])

So add a file such as:

```text
PedagogicalRL/src/vllm/mock_data_parallel_vllm.py
```

with the **same class name and method signatures** as the real one:

```python
class InferenceTask:
    GENERATE = "generate"
    REWARD = "reward"
    EMBEDDING = "embedding"
    CLASSIFY = "classify"

class ParallelvLLMInference:
    def __init__(self, *args, inference_task=InferenceTask.GENERATE, **kwargs): ...
    def run_batch(self, messages, sampling_params, meta=None): ...
    def sleep(self): pass
    def cleanup(self): pass
```

Then patch the import in `src/classroom.py` to switch on an env var:

```python
if os.getenv("MOCK_BACKEND", "0") == "1":
    from src.vllm.mock_data_parallel_vllm import ParallelvLLMInference, InferenceTask
else:
    from src.vllm.data_parallel_vllm import ParallelvLLMInference, InferenceTask
```

That one patch makes **both** the training-side FastAPI server and `eval.py` use the mock inference backend. ([GitHub][4])

## What `run_batch` should return

Your mock `run_batch` must imitate only the parts of the real outputs that the repo actually reads.

For teacher/student/judge generation, `Classroom` only uses `response.outputs[0].text` or iterates over `response.outputs`. For judge parsing it slices out the substring between `{` and `}` and runs `json.loads(...)` into `JudgeResponse`. So the mock must return:

* teacher outputs: normal tutor text,
* student outputs: normal student text or boxed answers,
* judge outputs: valid JSON snippets such as:

```json
{"reasoning":"safe tutoring response","decision":"OK"}
```

because that is the exact parser path in `Classroom`. ([GitHub][4])

For the end reward, you do **not** need a separate reward model. `Classroom._compute_rewards_from_prompts(...)` already has a built-in bypass: if `reward_model.model_name_or_path == "Answer"`, it extracts `\boxed{...}` and compares against the gold answer; if it is `"None"`, it returns zeros. So for the smoke set:

```yaml
reward_model:
  model_name_or_path: "Answer"
```

and make the mock student final solution sometimes output correct `\boxed{...}` and sometimes wrong `\boxed{...}` so rewards vary. ([GitHub][4])

## TutorRM in smoke mode

For the **TutorRM integration smoke**, do **not** load the real `kpetyxova/towards-reward-modeling-tutors` model. In your own handoff, the fixed reward order is:

```text
[end_rm_reward, tutor_rm_reward, thinking_reward, end_of_conversation_reward, length_reward]
```

and the A1/A2 smoke success criteria explicitly include that `/get_tutor_rm_reward` works and GDPO runs without shape errors. So in smoke mode, implement `/get_tutor_rm_reward` as a deterministic function of conversation text and keep the endpoint contract the same: plain list of floats. That validates the integration and the reward ordering without pulling the real classifier.  

A good deterministic smoke formula is:

```python
h = int(md5(text.encode()).hexdigest(), 16)
tutor_rm = ((h % 200) - 100) / 100.0
```

Then make the other rewards non-collinear too:

```python
end_rm = 1.0 if boxed_answer_correct else 0.0
thinking = 0.5 if "<think>" in text else 0.0
end_of_conversation = 1.0 if turns_finished else -0.5
length = -min(num_tokens, 256) / 256.0
```

The key point is: **do not return constants**. GDPO normalizes per-reward and across rewards, so you want each reward channel to vary across samples. Your own handoff’s GDPO branch depends on non-degenerate `rewards_per_func`. 

## Why mocking only the FastAPI server is not enough

Because `eval.py` does **not** talk to the FastAPI server. It directly instantiates `Classroom(...)` and runs `classroom.sample_conversations(...)`. So if you only fake `vllm_server.py`, A1/A2 training may get past HTTP, but internal eval will still try to use the real vLLM-backed `ParallelvLLMInference`. That is why the correct smoke patch point is the lower-level inference class, not only the server. ([GitHub][5])

## What still needs a tiny real model

For **A1/A2 training**, you cannot fully mock everything if you want to test the actual GRPO/GDPO optimizer path. `train_rl.py` still constructs a real `ClassroomGRPOTrainer` with a causal LM model and then calls `trainer.train(...)`. TRL’s GRPO trainer supports custom reward functions and even a custom `rollout_func`, but the trainer still needs a real causal LM to optimize. ([Hugging Face][6])

So the minimal algorithm smoke is:

* keep **mock rollout/inference** via `MOCK_BACKEND=1`,
* keep **mock TutorRM endpoint**,
* use a **tiny CPU trainable model** for the tutor policy.

Use:

```text
trl-internal-testing/tiny-LlamaForCausalLM-3.2
```

It is a **2.05M-parameter** minimal test model and the model card exposes a **chat template**, which is useful because the repo’s client path applies chat templates to message lists. ([Hugging Face][7])

## The one training patch you still need

In smoke mode, patch `train_rl.py` so it does **not** insist on BF16 + FlashAttention-2.

Use an env gate:

```python
SMOKE_CPU = os.getenv("SMOKE_CPU", "0") == "1"

if SMOKE_CPU:
    torch_dtype = torch.float32
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        use_cache=True,
    )
    use_bf16 = False
else:
    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if train_config.gradient_checkpointing else True,
    )
    use_bf16 = True
```

and then replace the hardcoded:

```python
bf16=True
```

with:

```python
bf16=use_bf16
```

Otherwise the no-GPU smoke still dies in the trainer path even after all inference is mocked. That is directly implied by the upstream code. ([GitHub][1])

## Launcher config for no-GPU smoke

Do **not** reuse `zero3_1GPU.yaml` for a CPU smoke. Your upstream launcher insists on `--config_file`, but Accelerate supports CPU training through config and also supports launch without a prior `accelerate config` as long as the right flags or config are passed. For the smoke, create a tiny Accelerate config with:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: no
num_processes: 1
use_cpu: true
machine_rank: 0
num_machines: 1
rdzv_backend: static
```

That is consistent with Accelerate’s documented config fields and CPU support. ([Hugging Face][8])

## Exact smoke matrix

### A0a / A0b

Use `mock_openai_server.py` and keep the normal benchmark commands. Change only the base URL and alias.

This proves benchmark orchestration and output writing. It does **not** prove TutorRL-7B serving. ([GitHub][2])

### A1 GRPO

Use all three knobs:

```bash
export MOCK_BACKEND=1
export SMOKE_CPU=1
export SERVER_PORT=8005
```

and run `start_rl_training.sh` with:

* CPU accelerate config,
* tiny Llama tutor model,
* `train.max_steps=2`,
* `dataset.max_train_examples=8`,
* `train.number_of_problems_per_batch=1`,
* `train.num_samples_per_problem=2`,
* `generation.max_turns=2`,
* `generation.max_tokens_per_turn=64`,
* `generation.max_tokens_per_student_attempt=64`,
* `generation.max_tokens_per_judge_attempt=64`,
* `reward_model.model_name_or_path=Answer`,
* `logging.wandb=false`.

This proves launcher, HTTP server startup, mock rollout, reward wiring, checkpoint save, and the stock GRPO branch. Your own smoke stage expects exactly those kinds of signals: server starts, training connects, TutorRM path works, checkpoint saves. 

### A2 GDPO

Same as A1, but:

* `SERVER_PORT=8006`,
* `train.use_gdpo=true`.

This proves the GDPO branch and its tensor shapes, provided your mock rewards vary across samples.  

### Internal eval

Use:

```bash
export MOCK_BACKEND=1
```

and run `eval.py` normally. Since `eval.py` constructs `Classroom(...)` directly, the lower-level inference mock is what makes this stage work. This validates metrics computation and output writing, not real model quality. ([GitHub][5])

### External A1 / A2 benchmark after training

Use `mock_openai_server.py` again and point the benchmark to the trained alias or checkpoint path only for naming/plumbing purposes. That validates the post-train benchmark workflow and aggregation, not the checkpoint’s real serving behavior. 

### Stage 6 LightEval

For the **no-model** smoke, skip it or stub it. In your own runbook it is a separate, winner-only sanity stage in its own env after winner selection, not part of the core A0/A1/A2/A3 path. So omitting it from the no-model smoke does not weaken the core workflow test.  

## What this smoke will and will not prove

It **will** prove:

* stage orchestration,
* launcher behavior,
* HTTP contracts,
* benchmark provider wiring,
* reward order and weighting,
* TutorRM endpoint integration,
* GRPO vs GDPO branch execution,
* internal eval metrics/output generation,
* aggregation and winner-selection plumbing.  

It will **not** prove:

* real vLLM startup,
* xformers / FlashAttention compatibility,
* real CUDA behavior,
* real 7B checkpoint loading,
* real TutorRM classifier semantics,
* real benchmark scores.

So yes, there is a clean bypass. The right target is a **three-knob smoke mode**:

```text
MOCK_BACKEND=1   # replace ParallelvLLMInference everywhere in Classroom paths
SMOKE_CPU=1      # tiny CPU causal LM + eager/fp32 trainer path
MOCK_OPENAI=1    # fake OpenAI-compatible server for MathTutorBench
```

That gets you a genuine **end-to-end code-flow test** for A0/A1/A2/A3 without solving the old-GPU vLLM problem first.

[1]: https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/train_rl.py "https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/train_rl.py"
[2]: https://github.com/eth-lre/mathtutorbench "GitHub - eth-lre/mathtutorbench: Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors, EMNLP 2025 Oral · GitHub"
[3]: https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/src/vllm/data_parallel_vllm.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/src/classroom.py "https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/src/classroom.py"
[5]: https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/eval.py "raw.githubusercontent.com"
[6]: https://huggingface.co/docs/trl/grpo_trainer "GRPO Trainer · Hugging Face"
[7]: https://huggingface.co/trl-internal-testing/tiny-LlamaForCausalLM-3.2 "https://huggingface.co/trl-internal-testing/tiny-LlamaForCausalLM-3.2"
[8]: https://huggingface.co/docs/accelerate/usage_guides/intel_cpu "https://huggingface.co/docs/accelerate/usage_guides/intel_cpu"
