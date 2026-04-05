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
