=== Batch 1

cd /cluster/scratch/ppurkayastha/Acads/Nachi/PedagogicalRLTutor

# Preflight: vLLM ping on a custom port (uses TORCH_SDPA + VLLM_USE_V1=0 by default)
sbatch smokes/slurm/stage_vllm_ping_smoke.sbatch

# A0a + A0b + A1 + A2 all parallel (ports: 8100/8101/8005/8006)
sbatch smokes/slurm/stage_a0a_baseline_smoke.sbatch
sbatch smokes/slurm/stage_a0b_baseline_smoke.sbatch
sbatch smokes/slurm/stage_a1_grpo_smoke.sbatch
sbatch smokes/slurm/stage_a2_gdpo_smoke.sbatch

Submitted batch job 63044368
Submitted batch job 63044370
Submitted batch job 63044372
Submitted batch job 63044374

====

Batch 2 (depends on A1/A2 outputs)

cd /cluster/scratch/ppurkayastha/Acads/Nachi/PedagogicalRLTutor

# internal evals + external benches in parallel (ports: 8102/8103)
sbatch smokes/slurm/stage_internal_eval_smoke_grpo.sbatch
sbatch smokes/slurm/stage_internal_eval_smoke_gdpo.sbatch
sbatch smokes/slurm/stage_external_mtbench_smoke_grpo.sbatch
sbatch smokes/slurm/stage_external_mtbench_smoke_gdpo.sbatch


Batch 3 (depends on GDPO external + internal done; assumes GDPO winner)

cd /cluster/scratch/ppurkayastha/Acads/Nachi/PedagogicalRLTutor

# LightEval + A3 can run in parallel (ports: 8104 for A3 benchmark)
sbatch --export=WINNER_MODEL=/cluster/scratch/ppurkayastha/Acads/Nachi/PedagogicalRLTutor/tutor_gdpo_project/PedagogicalRL/outputs/tutorrm_gdpo/model \
  smokes/slurm/stage_winner_lighteval_smoke.sbatch

sbatch --export=WINNER_LABEL=tutorrm_gdpo,WINNER_CONFIG=7b_tutorrm_gdpo.yaml \
  smokes/slurm/stage_a3_smoke.sbatch
