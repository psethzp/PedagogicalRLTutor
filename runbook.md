# Runbook

## Scope
- Track every setup step for the TutorRM + GDPO implementation.
- Stage 1 only for now: workspace bootstrap and repo cloning.

## Stage 1 Checklist
- [ ] Create project workspace
- [ ] Clone PedagogicalRL
- [ ] Clone mathtutorbench
- [ ] Clone Towards_Reward_Modeling_for_Tutors
- [ ] Verify required entry files exist

## Step Log
- Created workspace root at `~/tutor_gdpo_project`.
- Cloned `eth-lre/PedagogicalRL` into `~/tutor_gdpo_project/PedagogicalRL`.
- Cloned `eth-lre/mathtutorbench` into `~/tutor_gdpo_project/mathtutorbench`.
- Cloned `Kpetyxova/Towards_Reward_Modeling_for_Tutors` into `~/tutor_gdpo_project/Towards_Reward_Modeling_for_Tutors`.
- Verified required entry files:
  - `PedagogicalRL/train_rl.py`
  - `mathtutorbench/main.py`
  - `Towards_Reward_Modeling_for_Tutors/inference.py`
- Stage 1 complete.
- Added `~/tutor_gdpo_project/setup_env.sh` for reproducible venv bootstrap.
- Made `setup_env.sh` executable.
- Created `~/tutor_gdpo_project/.venv`.
- Installed the Stage 2 base Python dependencies into the venv.
- Ran import sanity checks successfully for:
  - `torch`
  - `transformers`
  - `trl`
  - `fastapi`
  - `uvicorn`
  - `datasets`
- `torch.cuda.is_available()` returned `False` on this machine, so GPU execution remains for the later CUDA box.
- Stage 2 complete.
- Patched `PedagogicalRL/stop_vllm_server.sh` to use PID-file shutdown only.
- Patched `PedagogicalRL/start_vllm_server.sh` to `exec python vllm_server.py "$@"`.
- Patched `PedagogicalRL/start_rl_training.sh` to use `SERVER_PORT`, write `.vllm_${SERVER_PORT}.pid`, and wait on `localhost:${SERVER_PORT}`.
- Verified `start_rl_training.sh` references `localhost:${SERVER_PORT}`.
- Verified `stop_vllm_server.sh` contains no `pkill`.
- Verified shell syntax for all three launcher scripts with `bash -n`.
- Stage 3 complete.
- Added TutorRM and GDPO config fields to `config/train_rl_model.py`.
- Added TutorRM classifier loading and scoring to `src/classroom.py`.
- Added `/get_tutor_rm_reward` to `vllm_server.py`.
- Added `get_tutor_rm_reward` RPC helper in `src/vllm/client.py`.
- Added `construct_tutor_rm_reward_func` in `src/utils/utils.py`.
- Added TutorRM reward to `train_rl.py` in the required order.
- Verified `python -m py_compile` on the patched training and server modules.
- Verified `RLModelTrainingConfig()` exposes the new TutorRM and GDPO defaults.
- Stage 4 complete.
- Added `reward_weights`, `apply_gdpo`, and `gdpo_eps` to `src/grpo/config.py`.
- Added weighted reward handling and GDPO advantage normalization to `src/grpo/trainer.py`.
- Wired `reward_weights`, `apply_gdpo`, and `gdpo_eps` through `train_rl.py` into `ClassroomGRPOConfig`.
- Verified `python -m py_compile` on `src/grpo/config.py`, `src/grpo/trainer.py`, and `train_rl.py`.
- Verified `ClassroomGRPOConfig(...)` accepts custom `reward_weights`, `apply_gdpo`, and `gdpo_eps`.
- Stage 5 complete.
- Created `config/deepspeed/zero3_1GPU.yaml`.
- Created `config/train_rl/7b_tutorrm_grpo.yaml`.
- Created `config/train_rl/7b_tutorrm_gdpo.yaml`.
- Verified all three YAML files parse successfully.
- Verified the TutorRM configs point to `eth-nlped/TutorRL-7B`, use ports `8005` and `8006`, and set `max_steps: 80`.
- Stage 6 complete.
