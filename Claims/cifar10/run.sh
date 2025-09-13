#!/usr/bin/env bash
cd .. 
cd ..
cd artifact
set -euo pipefail

# ===== Config =====
ENV_NAME="${ENV_NAME:-deepprov-env}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
PY_BIN="${PY_BIN:-$CONDA_HOME/envs/$ENV_NAME/bin/python}"
# ==================

# Sanitize this process so user fgsme-packages can't interfere
export PYTHONNOUSERFGSME=1
unset PYTHONPATH || true

# Check the env's Python exists
if [[ ! -x "$PY_BIN" ]]; then
  echo "[ERROR] Can't find env Python at: $PY_BIN"
  echo "        Make sure the conda env '${ENV_NAME}' exists under ${CONDA_HOME},"
  echo "        or set PY_BIN to the correct interpreter path."
  exit 1
fi

echo "[INFO] Using interpreter: $PY_BIN"
"$PY_BIN" - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0], "| Torch:", getattr(torch, "__version__", "not installed"))
PY

# --- Helper to run repo scripts inside the env ---
run() { "$PY_BIN" "$@"; }
run main_cifar10.py 

echo "[OK] Full pipeline completed Check expected folders per claim." 

