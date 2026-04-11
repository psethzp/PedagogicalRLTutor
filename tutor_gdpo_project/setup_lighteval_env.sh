#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/tutor_gdpo_project}"
CONSTRAINTS_FILE="${2:-$PROJECT_ROOT/constraints_lighteval.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN '$PYTHON_BIN' not found; set PYTHON_BIN to a Python 3.11 executable." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv_lighteval"
# shellcheck disable=SC1090
source "$PROJECT_ROOT/.venv_lighteval/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install without the vllm extra to avoid resolver conflicts with vllm==0.11.0.
pip install -c "$CONSTRAINTS_FILE" "lighteval==0.13.0" "vllm==0.11.0"

python - <<'PY'
import importlib.metadata as md
for pkg in ["lighteval", "vllm", "transformers", "torch", "latex2sympy2_extended"]:
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}: MISSING ({exc})")
PY

python -m pip check
pip freeze > "$PROJECT_ROOT/env_lighteval.freeze"
