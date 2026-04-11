Use this exact tiny smoke.

| Stage                                         | Model                                                                                                                                       | Backend                                                                                    | Data / fraction                                           | Exact settings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A0a baseline smoke**                        | `HuggingFaceTB/SmolLM2-135M-Instruct`                                                                                                       | **vLLM**                                                                                   | **1 example per MathTutorBench task**                     | Serve the tiny model but keep the alias as **`TutorRL-7B`** so the filenames stay identical to your real run. Use `--dtype half --tensor-parallel-size 1 --max-model-len 1024 --gpu-memory-utilization 0.35`. Do **not** use HF-direct here; MathTutorBench’s normal local path is vLLM + `main.py`. ([Hugging Face][1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **A0b baseline smoke**                        | same underlying model: `HuggingFaceTB/SmolLM2-135M-Instruct`                                                                                | **vLLM**                                                                                   | **1 example per MathTutorBench task**                     | Serve the same tiny model again, but keep the alias as **`TutorRL-7B-think`** so the pipeline still emits the expected second baseline row. Same serve flags as A0a.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **A1 GRPO smoke**                             | **teacher = student = judge = `HuggingFaceTB/SmolLM2-135M-Instruct`**; keep the **real TutorRM** `kpetyxova/towards-reward-modeling-tutors` | **PedagogicalRL normal flow = vLLM for teacher/student/judge + HF classifier for TutorRM** | **2 sampled prompt groups total**                         | Do **not** use TutorRL weights at all. Override the three model paths to the 135M instruct model. Set: `train.number_of_problems_per_batch=1`, `train.num_samples_per_problem=2`, `train.per_device_train_batch_size=1`, `train.max_steps=2`, `train.save_policy_to_disk_every_n=1`, `teacher/student/judge.vllm.number_of_gpus_per_instance=1`, `teacher/student/judge.vllm.max_length=1024`, `teacher/student/judge.vllm.max_num_seqs=4`, `teacher/student/judge.vllm.gpu_memory_utilization=0.15`, `generation.max_turns=2`, `generation.max_tokens_in_conversation=512`, `generation.max_tokens_per_turn=96`, `generation.max_tokens_per_student_attempt=96`, `generation.max_tokens_per_judge_attempt=48`, `generation.number_student_attempts=1`, `generation.number_judge_attempts=1`, `generation.use_tutor_rm=true`, `generation.tutor_rm_mode=first_teacher_only`, `train.use_gdpo=false`, `logging.save_dir=outputs/tutorrm_grpo`. The TutorRM paper’s released reward model uses a **0.5B backbone**, so this is the heaviest non-tiny component you are keeping on purpose to validate the real reward path. ([arXiv][2]) |
| **A2 GDPO smoke**                             | same as A1                                                                                                                                  | **same as A1**                                                                             | **same as A1**                                            | Same overrides as A1, except `train.use_gdpo=true`, `generation.server_port=8006`, `logging.save_dir=outputs/tutorrm_gdpo`. On a 2070-class card, run **A1 then A2 sequentially**, not in parallel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Internal eval smoke after A1/A2**           | the two trained smoke outputs                                                                                                               | **PedagogicalRL eval path**                                                                | **full default internal eval**                            | Leave this one **full**. Your current runbook does not expose a clean row-limit knob for internal eval, so for smoke just run it unchanged on the tiny trained models. Keep `logging.save_dir=outputs/tutorrm_grpo` and `logging.save_dir=outputs/tutorrm_gdpo` so `eval_outputs/metrics.json` and `eval_outputs/conversations.csv` land where your final run expects them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **External MathTutorBench smoke after A1/A2** | `outputs/tutorrm_grpo/model` and `outputs/tutorrm_gdpo/model`                                                                               | **vLLM**                                                                                   | **1 example per task**                                    | Serve the trained smoke models with aliases **`tutorrm-grpo`** and **`tutorrm-gdpo`**. Run the smoke benchmark on sliced configs so you still get `results-*.yaml`, `generations-*.json`, and the pedagogical reward-model scoring files in the same schema as the real run. MathTutorBench’s `main.py` just loads the task config and passes the split string through to `datasets.load_dataset`, and Hugging Face datasets supports split slicing like `train[:1]` / `test[:1]`. ([GitHub][3])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Winner + LightEval smoke**                  | whichever of `tutorrm-grpo` / `tutorrm-gdpo` wins the smoke aggregator                                                                      | **LightEval env**                                                                          | keep the current 3-task sanity set                        | Use the winner only. For the smoke run, shrink the backend load: `gpu_memory_utilization=0.35`, `max_model_length=1024`, `generation_parameters={max_new_tokens:128,temperature:0.0}`. Keep the task list the same.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **A3 smoke**                                  | start from the smoke winner config/output branch                                                                                            | **same as A1/A2 + vLLM for the 4-task appendix eval**                                      | **1 training step + 1 example on each of the 4 A3 tasks** | Generate `7b_tutorrm_a3.yaml` from the winner smoke config. Change only: `generation.tutor_rm_mode=all_teacher_turns_mean`, `generation.server_port=8007`, `train.max_steps=1`, `logging.save_dir=outputs/<winner>_a3`. Then run the 4-task A3 benchmark only on **1 example each** for `mistake_location`, `mistake_correction`, `scaffolding_generation`, and `pedagogy_following`, and still write the A3 `benchmark_meta-*.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

### Use this exact tiny model pack

* **teacher**: `HuggingFaceTB/SmolLM2-135M-Instruct`
* **student**: `HuggingFaceTB/SmolLM2-135M-Instruct`
* **judge**: `HuggingFaceTB/SmolLM2-135M-Instruct`
* **reward model**: `kpetyxova/towards-reward-modeling-tutors`
  SmolLM2 is available in **135M / 360M / 1.7B** sizes and is explicitly positioned as lightweight enough for on-device use, so the **135M instruct** variant is the safest smoke choice. The 0.5B TutorRM stays real so the reward path is actually validated. ([Hugging Face][1])

### Do this for the MathTutorBench smoke subset

Create **9 copied smoke configs** from the upstream 9 configs and only change the split strings:

* `train` → `train[:1]`
* `test` → `test[:1]`

Do **not** change prompts, task names, or metrics. That gives you the **same task code path**, just one row per task. `main.py` loads the task config and the dataloader passes the split string directly to `datasets.load_dataset`, so slice syntax is valid here. ([GitHub][3])

### vLLM vs HF-direct

* **Use vLLM** for **A0, A1, A2, A3, and MathTutorBench**.
* **Do not use HF-direct** for those stages.
* HF-direct is fine only for a one-line **TutorRM import sanity check**, but it does **not** validate the actual workflow you care about. MathTutorBench’s documented local-model path is vLLM serving followed by `main.py`. ([GitHub][4])

### RTX 2070 / pre-Ampere specifics

Use `--dtype half` on every vLLM serve path. Pre-Ampere GPUs do not support BF16 in the way newer cards do, and vLLM explicitly notes to use float16/`half` on lower compute-capability GPUs. ([GitHub][5])

### Only if it still OOMs

Do **not** start with quantization. Start with the 135M recipe above.
Only if it still OOMs, add to the vLLM serve commands:

```bash id="2vf1er"
--load-format bitsandbytes --quantization bitsandbytes
```

and install:

```bash id="nu644l"
pip install "bitsandbytes>=0.45.0"
```

vLLM documents BitsAndBytes support and its CLI `serve` path exposes both `--load-format bitsandbytes` and `--quantization bitsandbytes`. ([vLLM][6])

### What you will get from this smoke

You will get the **same filenames and table schemas** as the real run, because:

* A0a still writes as `TutorRL-7B`
* A0b still writes as `TutorRL-7B-think`
* A1 still writes as `tutorrm-grpo`
* A2 still writes as `tutorrm-gdpo`
* A3 still writes as `<winner>-a3`

So `aggregate_results.py` can still emit:

* `results_internal.csv`
* `results_external.csv`
* `results_efficiency.csv`

These numbers are **smoke-only**, but they will already be in the final file format you need.

So the exact tiny smoke is:

* **one tiny instruct model everywhere**
* **real TutorRM kept**
* **vLLM everywhere that matters**
* **1 example per benchmark task**
* **2 training steps for A1/A2**
* **1 training step for A3**
* **sequential on one small GPU**
* **same output names as the real pipeline**

[1]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct?utm_source=chatgpt.com "HuggingFaceTB/SmolLM2-135M-Instruct"
[2]: https://arxiv.org/abs/2603.24375?utm_source=chatgpt.com "Towards Reward Modeling for AI Tutors in Math Mistake Remediation"
[3]: https://raw.githubusercontent.com/eth-lre/mathtutorbench/main/main.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/eth-lre/mathtutorbench/main/README.md "raw.githubusercontent.com"
[5]: https://github.com/vllm-project/vllm/issues/12216?utm_source=chatgpt.com "[Usage]: BNB quantization not supported for Paligemma2 ..."
[6]: https://docs.vllm.ai/en/v0.8.1/features/quantization/bnb.html?utm_source=chatgpt.com "BitsAndBytes — vLLM"
