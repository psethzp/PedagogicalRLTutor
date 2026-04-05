#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/tutor_gdpo_project}"
CONSTRAINTS_FILE="${2:-$PROJECT_ROOT/constraints_tutor_gdpo.txt}"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.env"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

python3.11 -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
pip install ninja packaging

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

pip install -c "$CONSTRAINTS_FILE" \
  transformers==4.50.3 \
  trl==0.18.0 \
  accelerate==1.6.0 \
  datasets==3.1.0 \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  huggingface_hub==0.30.2 \
  fastapi==0.115.12 \
  uvicorn==0.34.0 \
  python-dotenv==1.0.1 \
  tenacity==9.0.0 \
  pynvml==12.0.0

pip install -c "$CONSTRAINTS_FILE" vllm==0.8.3
pip install -c "$CONSTRAINTS_FILE" flash-attn==2.7.4.post1 --no-build-isolation

pip install -c "$CONSTRAINTS_FILE" \
  -r "$PROJECT_ROOT/PedagogicalRL/requirements.txt" \
  -r "$PROJECT_ROOT/Towards_Reward_Modeling_for_Tutors/requirements.txt" \
  lighteval

pip install -c "$CONSTRAINTS_FILE" --no-deps \
  -r "$PROJECT_ROOT/mathtutorbench/requirements.txt"

python - <<'PY'
import json, os, sys
import torch, transformers, trl, fastapi, uvicorn, datasets, importlib.metadata as im
out = {
    'python': sys.version,
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'transformers': transformers.__version__,
    'trl': trl.__version__,
    'datasets': datasets.__version__,
    'fastapi': fastapi.__version__,
    'uvicorn': uvicorn.__version__,
    'vllm': im.version('vllm'),
}
with open(os.path.join(os.getcwd(), 'env_versions.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY
