Yes. Use **2 environments only**:

* **core env** for **everything in your actual paper pipeline**
* **lighteval env** for the **winner-only sanity benchmark after training**

Make sure to update the setup.sh and constratints accordingly so that can directly run that to setup these two separate envs as needed for different stages.


## The stage map

The clean stage sequence is:

| Stage   | What happens                                                                        | Which env         |
| ------- | ----------------------------------------------------------------------------------- | ----------------- |
| Stage 0 | setup, patching, config generation, validation                                      | **core env**      |
| Stage 1 | A0a = TutorRL-7B baseline on MathTutorBench                                         | **core env**      |
| Stage 2 | A0b = TutorRL-7B-think baseline on MathTutorBench                                   | **core env**      |
| Stage 3 | A1/A2 smoke tests                                                                   | **core env**      |
| Stage 4 | A1/A2 full training                                                                 | **core env**      |
| Stage 5 | internal eval + MathTutorBench external eval + aggregate results + winner selection | **core env**      |
| Stage 6 | winner-only LightEval sanity run                                                    | **lighteval env** |
| Stage 7 | A3 appendix run on the winner branch                                                | **core env**      |

This split will **not** cause issues because the two envs are not trying to share live Python state. They only share **files on disk**:

* the core env writes checkpoints into `~/tutor_gdpo_project/PedagogicalRL/outputs/...`
* the core env writes MathTutorBench results into `~/tutor_gdpo_project/mathtutorbench/results/...`
* the lighteval env later reads the **winner model path** from disk and runs one evaluation pass on that path

MathTutorBench’s intended usage is exactly “serve a model locally with vLLM, run `main.py`, then run `compute_scaffolding_score.py`,” and Lighteval is explicitly a separate evaluation toolkit with its own install path and optional extras. ([GitHub][1])

The only rule you must follow is: **do not run a LightEval vLLM job while a core-env MathTutorBench vLLM server is still running**. Your current runbook already puts LightEval **after** winner selection, which is the right place for it.

---

## Environment 1: core env

This is the one that runs the actual paper.

Use it for:

* Stage 0 setup/patching/config generation
* A0a baseline
* A0b baseline
* A1 smoke
* A2 smoke
* A1 final
* A2 final
* internal PedagogicalRL eval
* external MathTutorBench eval
* aggregate results / winner selection
* A3 config generation
* A3 training
* A3 4-task appendix benchmark

### Exact versions to use in the core env

Use the exact versions from your current runbook, plus keep your already-working compression alignment unchanged:

```text
python==3.11
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
vllm==0.8.3
flash-attn==2.7.4.post1
transformers==4.50.3
trl==0.18.0
accelerate==1.6.0
datasets==3.1.0
hydra-core==1.3.2
omegaconf==2.3.0
antlr4-python3-runtime==4.9.3
llmcompressor==0.4.1
compressed-tensors==0.9.2
```

Why this env should stay like this:

* your runbook explicitly fixes the core stack to Python 3.11, Torch 2.6.0, vLLM 0.8.3, Transformers 4.50.3, TRL 0.18.0, Accelerate 1.6.0, Datasets 3.1.0, Hydra 1.3.2, and OmegaConf 2.3.0
* the live PedagogicalRL requirements still pin `antlr4-python3-runtime==4.9.3`
* PedagogicalRL also still pins `vllm==0.8.3`, `hydra-core==1.3.2`, `omegaconf==2.3.0`, and `flash-attn==2.7.4.post1` in its own requirements file ([GitHub][2])

### How to verify the core env

Run this before Stage 1:

```bash
source ~/tutor_gdpo_project/.venv/bin/activate

python - <<'PY'
import importlib.metadata as md
pkgs = [
    "torch","torchvision","torchaudio","vllm","flash-attn",
    "transformers","trl","accelerate","datasets",
    "hydra-core","omegaconf","antlr4-python3-runtime",
    "llmcompressor","compressed-tensors","fastapi","uvicorn"
]
for p in pkgs:
    try:
        print(f"{p}=={md.version(p)}")
    except Exception as e:
        print(f"{p}: MISSING ({e})")
PY

python - <<'PY'
import torch, transformers, trl, vllm, hydra, omegaconf, fastapi, uvicorn
print("cuda_available =", torch.cuda.is_available())
PY

python -m pip check
pip freeze > ~/tutor_gdpo_project/env_core.freeze
```

What counts as success:

* all imports succeed
* `torch.cuda.is_available()` is `True` on the actual GPU machine
* `pip check` does not show a conflict you are ignoring
* `env_core.freeze` exists

If CUDA is still `False`, that is not an env-design issue. It just means you are still not on the actual GPU runtime.

---

## Environment 2: lighteval env

This env exists only for the **winner-only sanity benchmark** after A1/A2 are complete.

Use it for:

* Stage 6 only

Do **not** use it for A0/A1/A2/A3.

### Why separate LightEval at all?

Because current Lighteval has its own dependency line:

* Python **>= 3.10**
* optional extras via `pip install lighteval[<group>]`
* the `vllm` extra exists
* current metadata shows `lighteval==0.13.0` is the latest release on PyPI
* current project metadata also shows `transformers>=4.54.0`, `torch>=2.0,<3.0`, `latex2sympy2_extended==1.0.6`, and `vllm>=0.11.0` under the `vllm` extra

That is **not** the same dependency profile as your PedagogicalRL core env, which is exactly why separating it is safer. ([Hugging Face][3])

### Exact versions to use in the lighteval env

Use:

```text
python==3.11
lighteval==0.13.0
vllm==0.11.0
```

Do **not** try to force the lighteval env to match the core env’s Transformers/vLLM stack. Isolation is the whole point. Current Lighteval exposes a `vllm` extra and its current metadata requires `vllm>=0.11.0`, not your core env’s `vllm==0.8.3`. ([PyPI][4])

### How to create and verify the lighteval env

```bash
python3.11 -m venv ~/tutor_gdpo_project/.venv_lighteval
source ~/tutor_gdpo_project/.venv_lighteval/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Install without the vllm extra to avoid the current resolver conflict.
pip install "lighteval==0.13.0" "vllm==0.11.0"

python - <<'PY'
import importlib.metadata as md
for p in ["lighteval","vllm","transformers","torch","latex2sympy2_extended"]:
    try:
        print(f"{p}=={md.version(p)}")
    except Exception as e:
        print(f"{p}: MISSING ({e})")
PY

lighteval --help
python -m pip check
pip freeze > ~/tutor_gdpo_project/env_lighteval.freeze
```

What counts as success:

* `lighteval --help` works
* package versions print
* `pip check` is clean
* `env_lighteval.freeze` exists

---

## Exact experiment order with env switching

### Stage 0 — setup and patching

Activate **core env**.

Run:

* clone repos
* copy helper files
* create `.env`
* run `setup_env.sh`
* patch launcher
* patch logging fallback
* patch TutorRM
* patch GDPO
* generate `zero3_1GPU.yaml`
* generate `7b_tutorrm_grpo.yaml`
* generate `7b_tutorrm_gdpo.yaml`
* validate configs

### Stage 1 — A0a baseline

Still in **core env**.

Run:

* serve `eth-nlped/TutorRL-7B`
* run `run_mathtutorbench_suite.sh`
* kill exact PID
* internal baseline eval for TutorRL-7B

### Stage 2 — A0b baseline

Still in **core env**.

Run:

* serve `eth-nlped/TutorRL-7B-think`
* run `run_mathtutorbench_suite.sh`
* kill exact PID
* internal baseline eval for TutorRL-7B-think

### Stage 3 — A1/A2 smoke tests

Still in **core env**.

Run:

* A1 smoke on GPU 0
* A2 smoke on GPU 1

### Stage 4 — A1/A2 full training

Still in **core env**.

Run:

* A1 final on GPU 0
* A2 final on GPU 1

### Stage 5 — A1/A2 evaluation and winner selection

Still in **core env**.

Run:

* internal eval on `outputs/tutorrm_grpo/model`
* internal eval on `outputs/tutorrm_gdpo/model`
* external MathTutorBench on `tutorrm-grpo`
* external MathTutorBench on `tutorrm-gdpo`
* `aggregate_results.py`
* winner selection script from your runbook

At the end of this stage you should have:

* `WINNER_LABEL`
* `WINNER_ALIAS`
* `WINNER_MODEL`
* `WINNER_CONFIG`

### Stage 6 — winner-only LightEval

Deactivate core env and activate **lighteval env**.

Run:

* the exact `lighteval vllm ...` command from your runbook using `WINNER_MODEL`

Then you are done with the lighteval env.

### Stage 7 — A3 appendix run

Deactivate lighteval env and go back to **core env**.

Run:

* generate `7b_tutorrm_a3.yaml` from `WINNER_CONFIG`
* run A3 training
* run the 4-task A3 benchmark
* write `benchmark_meta-${A3_ALIAS}.json`

That is it.

---

## Why this will not cause issues

It is safe because:

* **core env** owns all training, serving, benchmarking, aggregation, and appendix work
* **lighteval env** only runs after A1/A2 are over and only reads the winner model path from disk
* no stage depends on importing a Python object from another env
* the only shared artifacts are files: checkpoints, CSVs, JSONs, logs
* MathTutorBench explicitly expects local model serving with vLLM and benchmark execution as separate steps, and Lighteval explicitly supports separate installation with optional extras and vLLM backends ([GitHub][1])

So the rule is:

* **A0/A1/A2/A3 = core env**
* **winner-only LightEval = lighteval env**

Nothing else needs a separate env.

## Recommended final setup

Use exactly **2 envs**:

* `~/tutor_gdpo_project/.venv` or `.venv_core` = core env
* `~/tutor_gdpo_project/.venv_lighteval` = lighteval env

Skip the `math-verify` env entirely unless you later decide you want extra standalone symbolic math-equivalence checks that are outside your main paper pipeline.

* Stage 0 to Stage 7 commands only, no explanation.

[1]: https://raw.githubusercontent.com/eth-lre/mathtutorbench/main/README.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/eth-lre/PedagogicalRL/main/requirements.txt "raw.githubusercontent.com"
[3]: https://huggingface.co/docs/lighteval/installation?utm_source=chatgpt.com "Installation"
[4]: https://pypi.org/project/lighteval/?utm_source=chatgpt.com "lighteval"
