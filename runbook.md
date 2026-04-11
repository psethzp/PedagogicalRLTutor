# Runbook

1. Initialized runbook in `/cluster/scratch/ppurkayastha/Acads/Nachi/PedagogicalRLTutor` and set this as the working project root.
2. Created project workspace directory: `tutor_gdpo_project`.
3. Cloned required repos into `tutor_gdpo_project`: `PedagogicalRL`, `mathtutorbench`, and `Towards_Reward_Modeling_for_Tutors`.
4. Copied helper files into `tutor_gdpo_project` and installed `run_mathtutorbench_suite.sh` into `tutor_gdpo_project/mathtutorbench/` with executable permissions.
5. Created `.env` from `.env.example` in `tutor_gdpo_project`.
6. Attempted `setup_env.sh` but failed because `python3.11` was not found in PATH.
7. Located available Python 3.11 modules via `module spider python`.
8. Loaded `stack/2025-06` + `gcc/8.5.0` + `python/3.11.9` modules to provide `python3.11`.
9. Re-ran `setup_env.sh` with the Python 3.11 module, but the command timed out before completion; env setup needs a retry to finish and write `env_versions.json`.
10. Encountered dependency conflict installing `vllm==0.8.3` (requires `transformers>=4.51.0`), so updated `tutor_gdpo_project/setup_env.sh` to install vLLM with `--no-deps` while keeping the locked `transformers==4.50.3` constraint.
11. Re-ran `setup_env.sh` and hit a failure building `flash-attn==2.7.4.post1` because `CUDA_HOME`/`nvcc` were missing.
12. Identified `cuda/12.6.2` as compatible with `stack/2025-06` + `gcc/8.5.0`; will use it to provide `nvcc`/`CUDA_HOME` for flash-attn.
13. Flash-attn build failed due to a cross-device hardlink error; updated `tutor_gdpo_project/setup_env.sh` to install `flash-attn` with `PIP_NO_CACHE_DIR=1`.
14. Updated `tutor_gdpo_project/setup_env.sh` to filter `vllm` out of `PedagogicalRL/requirements.txt` during the dependency install phase to avoid the `transformers>=4.51.0` resolver conflict.
15. Split `setup_env.sh` to install `lighteval` in a separate pip step to reduce resolver backtracking time.
16. Added constraint pins for `starlette==0.46.2`, `anyio==4.13.0`, and `latex2sympy2_extended==1.0.6` in `tutor_gdpo_project/constraints_tutor_gdpo.txt` to reduce pip backtracking.
17. Added constraint pins for `fsspec==2024.9.0`, `dill==0.3.8`, and `multiprocess==0.70.16` to reduce further resolver backtracking.
18. Re-checked `AGENTS.md` section headings to confirm required stages and tasks before continuing.
19. Installed dependency batch: cachetools, colorama, timeout_decorator, evaluate, polars, scikit-learn (noted vLLM missing-deps warnings).
20. Installed dependency batch: wandb, peft, bitsandbytes (noted vLLM missing-deps warnings).
21. Installed dependency: deepspeed (and its deps); noted vLLM missing-deps warnings.
22. Installed dependency: liger-kernel.
23. Installed llmcompressor (pulled compressed-tensors 0.10.2; noted vLLM version conflict).
24. Installed math-verify with `--no-deps` to avoid antlr4 runtime version conflict; will validate during sanity checks.
25. Installed dependency batch: google-generativeai, langchain, langchain-core, langchain-anthropic, ipywidgets and their transitive deps.
26. Verified all packages from project requirements are installed (missing list empty).
27. Installed vLLM dependency batch (blake3, cloudpickle, depyf, gguf, importlib_metadata, lark, llguidance, lm-format-enforcer, outlines, partial-json-parser, prometheus-client, prometheus-fastapi-instrumentator, pyzmq, msgspec, sentencepiece, tiktoken, watchfiles, python-json-logger, openai) and downgraded `compressed-tensors` to 0.9.2 (conflicts with llmcompressor).
28. Installed opencv-python-headless, mistral_common[opencv], and xgrammar for vLLM.
29. Verified vLLM imports and versions (transformers 4.50.3 / trl 0.18.0 / vllm 0.8.3).
30. Ran sanity checks; torch reports `cuda_available=False` in this environment.
31. Generated `env_versions.json` in `tutor_gdpo_project`.
32. Checked `nvidia-smi`; NVIDIA driver not available in this environment (CUDA unavailable).
33. Patched `PedagogicalRL/stop_vllm_server.sh`, `start_vllm_server.sh`, and `start_rl_training.sh` for port-safe server lifecycle with PID files.
34. Validated launcher patch: start script uses `localhost:${SERVER_PORT}` and no `pkill` remains in `stop_vllm_server.sh`.
35. Applied local-only logging fallback patch in `PedagogicalRL/train_rl.py` (WANDB guard, resolved config save, train summary).
36. Applied local-only logging fallback patch in `PedagogicalRL/vllm_server.py` (local batch CSVs, WANDB guard, save-dir creation).
37. Applied local-only logging fallback patch in `PedagogicalRL/eval.py` (local metrics/conversations outputs, WANDB guard).
38. Added TutorRM config fields in `config/train_rl_model.py` (GenerationConfig + TrainConfig).
39. Implemented TutorRM scoring in `src/classroom.py` (model load, serialization, reward methods).
40. Added TutorRM reward endpoint and column in `vllm_server.py`.
41. Added TutorRM client call and reward wrapper in `src/vllm/client.py` and `src/utils/utils.py`.
42. Updated `train_rl.py` reward ordering to include TutorRM and passed reward weights.
43. Added GDPO wiring in `PedagogicalRL/src/grpo/config.py`, `PedagogicalRL/src/grpo/trainer.py`, and `PedagogicalRL/train_rl.py` (reward_weights validation, apply_gdpo flag, gdpo_eps usage, GDPO advantage block).
44. Attempted to generate reduced configs with `python` but `python` was not in PATH; re-ran with the venv Python.
45. Generated `PedagogicalRL/config/deepspeed/zero3_1GPU.yaml` from `zero3_4GPU.yaml`.
46. Generated reduced training configs `PedagogicalRL/config/train_rl/7b_tutorrm_grpo.yaml` and `PedagogicalRL/config/train_rl/7b_tutorrm_gdpo.yaml`.
47. Validated reduced configs by printing the first 1200 chars and confirmed `server_port` values (8005/8006) via `rg`.
48. Noted that stages 15+ (baselines, smokes, training, external evals) are intentionally skipped for now to avoid weight downloads/experiments per instruction.
49. Attempted a full import check for the “missing packages” list; the long import timed out but surfaced `llmcompressor` import errors (compressed-tensors mismatch) and `math_verify` errors (missing latex2sympy2_extended).
50. Installed `latex2sympy2_extended==1.0.6` which upgraded `antlr4-python3-runtime` to 4.13.2; reverted back to 4.9.3 because Hydra/OmegaConf failed with 4.13.2.
51. Tested Hydra/OmegaConf with `antlr4-python3-runtime==4.9.3` to confirm config parsing works.
52. Temporarily upgraded `compressed-tensors` to 0.10.2, then reverted to 0.9.2 to align with the vLLM pin; confirmed `vllm` still imports.
53. Downgraded `llmcompressor` to 0.4.1 (compatible with compressed-tensors 0.9.2) and updated both `constraints_tutor_gdpo.txt` files; `llmcompressor` now imports.
54. Confirmed `latex2sympy2_extended`/`math_verify` still fail under antlr4 4.9.3 due to generated parser/runtime mismatch; left unresolved pending decision because antlr4 4.13.2 breaks Hydra/OmegaConf.
55. Updated both `constraints_tutor_gdpo.txt` files to pin `llmcompressor==0.4.1` (compatible with `compressed-tensors==0.9.2`).
56. Updated `setup_env.sh` to exclude `math-verify` from the main requirements install and install it separately with `--no-deps`; synced the root `setup_env.sh` from the project copy.
57. Re-verified `llmcompressor` imports cleanly under `compressed-tensors==0.9.2`; `math_verify`/`latex2sympy2_extended` remain broken under `antlr4-python3-runtime==4.9.3`.
58. Read `ENV.md` to confirm the two-env split (core env + lighteval env) and stage mapping.
59. Removed `lighteval` from the core `setup_env.sh` installs (both root and `tutor_gdpo_project` copies).
60. Added `constraints_lighteval.txt` (root and `tutor_gdpo_project`) to pin `lighteval==0.13.0` and `vllm==0.11.0`.
61. Added `setup_lighteval_env.sh` (root and `tutor_gdpo_project`) to build the separate `.venv_lighteval` with LightEval-specific dependencies.
62. Tried running `setup_lighteval_env.sh` and discovered `python3.11` is not on PATH in this session.
63. Updated `setup_lighteval_env.sh` to accept `PYTHON_BIN` override and added a guard if the executable is missing; synced the project copy.
64. Running `setup_lighteval_env.sh` failed because `lighteval[vllm]==0.13.0` requires `vllm<0.10.2`, conflicting with the `vllm==0.11.0` pin from `ENV.md`.
65. Updated `setup_lighteval_env.sh` to install `lighteval==0.13.0` without the `vllm` extra and then install `vllm==0.11.0` separately; synced the project copy.
66. Re-ran the LightEval env install with an extended timeout; `lighteval==0.13.0` and `vllm==0.11.0` installed successfully into `tutor_gdpo_project/.venv_lighteval`.
67. Verified LightEval env versions (lighteval 0.13.0, vllm 0.11.0, transformers 5.5.3, torch 2.8.0, latex2sympy2_extended 1.0.6), ran `lighteval --help`, and confirmed `pip check` is clean.
68. Wrote `tutor_gdpo_project/env_lighteval.freeze` from the LightEval env.
69. Updated `ENV.md` to install `lighteval==0.13.0` without the `vllm` extra (avoids the resolver conflict while still pinning `vllm==0.11.0`).
70. Added `PYTHON_BIN` override + guard to both `setup_env.sh` copies so the core env setup can run even if `python3.11` is not on PATH.
71. Created new independent `smokes/` folder with subdirs for datasets, configs, models, and scripts.
72. Added `smokes/scripts/create_smoke_configs.py` and generated 9 MathTutorBench smoke configs with split slicing in `smokes/mathtutorbench_configs/`.
73. Added `smokes/scripts/build_smoke_datasets.py` and generated one-example JSONL slices for each task in `smokes/datasets/` (handles both Hub datasets and local MathDial JSON files).
74. Added `smokes/scripts/download_smoke_models.py` and downloaded smoke model weights into `smokes/models/` for SmolLM2-135M-Instruct and the TutorRM model.
75. Created `smokes/slurm/logs` and `smokes/slurm/pids` directories for SLURM outputs and vLLM PID files.
76. Added SLURM sbatch for DEMO Stage A0a baseline smoke: `smokes/slurm/stage_a0a_baseline_smoke.sbatch`.
77. Added SLURM sbatch for DEMO Stage A0b baseline smoke: `smokes/slurm/stage_a0b_baseline_smoke.sbatch`.
78. Added SLURM sbatch for DEMO Stage A1 GRPO smoke training: `smokes/slurm/stage_a1_grpo_smoke.sbatch`.
79. Added SLURM sbatch for DEMO Stage A2 GDPO smoke training: `smokes/slurm/stage_a2_gdpo_smoke.sbatch`.
80. Added SLURM sbatch for internal eval smoke (GRPO): `smokes/slurm/stage_internal_eval_smoke_grpo.sbatch`.
81. Added SLURM sbatch for internal eval smoke (GDPO): `smokes/slurm/stage_internal_eval_smoke_gdpo.sbatch`.
82. Added SLURM sbatch for external MathTutorBench smoke (GRPO): `smokes/slurm/stage_external_mtbench_smoke_grpo.sbatch`.
83. Added SLURM sbatch for external MathTutorBench smoke (GDPO): `smokes/slurm/stage_external_mtbench_smoke_gdpo.sbatch`.
84. Added SLURM sbatch for winner-only LightEval smoke: `smokes/slurm/stage_winner_lighteval_smoke.sbatch`.
85. Added SLURM sbatch for A3 smoke (training + 4-task benchmark): `smokes/slurm/stage_a3_smoke.sbatch`.
86. Updated smoke SLURM sbatch files to make vLLM ports configurable (VLLM_PORT/BASE_URL) and avoid collisions across stages.
87. Added configurable SERVER_PORT for A1/A2 smoke training and TRAIN_SERVER_PORT/VLLM_PORT for A3 smoke (training vs benchmark serve ports).
88. Canceled the initial smoke SLURM jobs (A0a/A0b/A1/A2) after repeated VLLM connection refusals.
89. Found root cause for A0a/A0b: `vllm serve` failed due to missing `uvloop` (ModuleNotFoundError) in the core env.
90. Added `uvloop==0.19.0` to both constraints files and core `setup_env.sh` copies, then installed `uvloop` into the core env.
91. Added server startup timeout/health checks to `PedagogicalRL/start_rl_training.sh` to avoid infinite waits if the server exits or never becomes ready.
92. Added similar VLLM startup timeout/health checks to smoke sbatch files that run `vllm serve` (A0a/A0b/external GRPO/external GDPO/A3 benchmark).
93. Verified `uvloop` import works in the core env.
94. Identified missing `python-multipart` as the cause of vLLM server startup failures.
95. Added `python-multipart==0.0.9` to both constraints files and both `setup_env.sh` copies.
96. Installed `python-multipart==0.0.9` into the core env.
97. Resubmitted smoke jobs: A0a=63045948, A0b=63045950, A1=63045952, A2=63045954.
