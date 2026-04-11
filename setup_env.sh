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

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN '$PYTHON_BIN' not found; set PYTHON_BIN to a Python 3.11 executable." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
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
  sacrebleu==2.5.1 \
  huggingface_hub==0.30.2 \
  fastapi==0.115.12 \
  uvicorn==0.34.0 \
  uvloop==0.19.0 \
  python-multipart==0.0.9 \
  python-dotenv==1.0.1 \
  tenacity==9.0.0 \
  pynvml==12.0.0

pip install -c "$CONSTRAINTS_FILE" --no-deps vllm==0.8.3
PIP_NO_CACHE_DIR=1 pip install -c "$CONSTRAINTS_FILE" flash-attn==2.7.4.post1 --no-build-isolation

REQ_TMP="$(mktemp)"
grep -v -E '^(vllm|math-verify)([=<>]|$)' "$PROJECT_ROOT/PedagogicalRL/requirements.txt" > "$REQ_TMP"
pip install -c "$CONSTRAINTS_FILE" \
  -r "$REQ_TMP" \
  -r "$PROJECT_ROOT/Towards_Reward_Modeling_for_Tutors/requirements.txt"
rm -f "$REQ_TMP"

pip install -c "$CONSTRAINTS_FILE" --no-deps math-verify

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
