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
