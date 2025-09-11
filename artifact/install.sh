#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
ENV_NAME="deepprov-env"
PYTHON_VERSION="3.10"
REPO_DIR="${REPO_DIR:-$PWD}"
REQ_FILE="${REPO_DIR}/artifact/requirements.txt"

# CUDA tag requested (PyTorch wheel index)
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_CHANNEL="https://download.pytorch.org/whl/${CUDA_TAG}"

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_BIN="${CONDA_BIN:-$CONDA_HOME/bin/conda}"
# ------------------------

# Helper: run conda in a clean process (no user site, no plugins, no PYTHONPATH)
_conda_env() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 CONDA_NO_PLUGINS=1 "$CONDA_BIN" "$@"
}

# Helper: run *inside* the env without activating
_crun() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 CONDA_NO_PLUGINS=1 "$CONDA_BIN" run -n "$ENV_NAME" "$@"
}

echo "==> Repo dir: $REPO_DIR"

echo "==> System checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] NVIDIA driver:"
  nvidia-smi || true
else
  echo "[WARN] No NVIDIA GPU detected. Installing CUDA ${CUDA_TAG} wheels anyway (driver must support them at runtime)."
fi

# Conda setup (user-local if missing); no shell hook; no PATH edits
if [[ ! -x "$CONDA_BIN" ]]; then
  if [[ ! -d "$CONDA_HOME" ]]; then
    echo "[INFO] Installing Miniconda (user-local) into $CONDA_HOME"
    TMPD=$(mktemp -d); pushd "$TMPD" >/dev/null
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o m.sh
    bash m.sh -b -p "$CONDA_HOME"
    popd >/dev/null
  else
    echo "[INFO] Found existing Miniconda at $CONDA_HOME"
  fi
else
  echo "[INFO] Using existing conda at $CONDA_BIN"
fi

# Create env (no activation / no plugins / user site ignored)
# Force solver=classic AND pin channels to defaults for this operation only.
if ! _conda_env env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "==> Creating env ${ENV_NAME} (python=${PYTHON_VERSION}) [classic solver, defaults channel]"
  _conda_env create -y --solver=classic --override-channels -c defaults \
    -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "==> Env ${ENV_NAME} already exists"
fi

# Verify Python inside env (no activation)
_crun python - <<'PY'
import sys
print("Python in env:", sys.version)
PY

echo "==> Upgrading pip/wheel/setuptools"
_crun python -m pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch/torchvision/torchaudio from ${TORCH_CHANNEL}"
# Let pip pick latest compatible cu129 builds
_crun python -m pip install --index-url "${TORCH_CHANNEL}" torch torchvision torchaudio

# Capture torch version (e.g., 2.x.y) without build suffix
TORCH_VER=$(_crun python - <<'PY'
import torch, re
v = torch.__version__
print(re.split(r'\+|;', v)[0])
PY
)
echo "==> Detected torch version: ${TORCH_VER}"

# --- PyG install (robust: detect Torch version, try minor.0 page, soft-optional deps) ---

# 1) Detect Torch version reliably
TORCH_VER=$(_crun python - <<'PY'
import re
try:
    import torch
    v = torch.__version__
    print(re.split(r'[+;]', v)[0].strip())   # e.g., "2.6.1"
except Exception:
    print("")                                 # let shell handle fallback
PY
)

# Fallback via pip metadata if python import didn't work
if [[ -z "${TORCH_VER}" ]]; then
  TORCH_VER="$(_crun python -m pip show torch 2>/dev/null | awk '/^Version:/ {print $2}')"
fi

if [[ -z "${TORCH_VER}" ]]; then
  echo "[ERROR] Could not determine PyTorch version in env; ensure torch is installed."
  exit 1
fi

# 2) Build both exact and minor.0 wheel pages
TORCH_MM="$(printf "%s" "${TORCH_VER}" | awk -F. '{print $1"."$2}')"
PYG_URL_EXACT="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
PYG_URL_MINOR0="https://data.pyg.org/whl/torch-${TORCH_MM}.0+${CUDA_TAG}.html"

echo "==> Installing PyG from:"
echo "    - ${PYG_URL_EXACT}"
echo "    - ${PYG_URL_MINOR0} (fallback)"

# 3) Install core package first (underscore names per PyG quick start)
#    torch_geometric will work even if optional extensions are missing.
_crun python -m pip install \
  -f "${PYG_URL_EXACT}" -f "${PYG_URL_MINOR0}" \
  torch_geometric

# 4) Try optional binary extensions individually; don't fail the whole setup if one is missing
for PKG in torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib; do
  if ! _crun python -m pip install -f "${PYG_URL_EXACT}" -f "${PYG_URL_MINOR0}" "$PKG"; then
    echo "[WARN] Optional PyG dependency '$PKG' not available for torch=${TORCH_VER} ${CUDA_TAG}; continuing."
  fi
done
# -------------------------------------------------------------------------------

# Project requirements (EXCEPT Torch/PyG we just installed)
if [[ -f "${REQ_FILE}" ]]; then
  echo "==> Installing project requirements (excluding Torch/PyG already installed)"
  grep -viE '^(torch($|==)|torchvision($|==)|torchaudio($|==)|torch[-_]?geometric($|==)|torch[-_]?scatter($|==)|torch[-_]?sparse($|==)|torch[-_]?cluster($|==)|torch[-_]?spline[-_]?conv($|==))' \
    "${REQ_FILE}" > /tmp/req-non-torch.txt || true
  if [[ -s /tmp/req-non-torch.txt ]]; then
    _crun python -m pip install -r /tmp/req-non-torch.txt
  else
    echo "[INFO] No additional non-Torch requirements to install."
  fi
else
  echo "[INFO] No requirements.txt found at ${REQ_FILE}"
fi

# Create standard folders if missing
mkdir -p data models runs artifacts data/attributions_data

# Machine summary (run inside env)
_crun python - <<'PY'
import json, platform, os, subprocess, sys
def run(x):
    try:
        return subprocess.check_output(x, shell=True, text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        return str(e)
info = {
  "python": sys.version,
  "platform": platform.platform(),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",""),
  "nvidia_smi": run("nvidia-smi"),
  "torch_version": None,
  "torch_cuda_is_available": None,
  "torch_cuda_version": None,
  "pip_freeze": run("pip freeze"),
}
try:
    import torch
    info["torch_version"] = torch.__version__
    info["torch_cuda_is_available"] = torch.cuda.is_available()
    info["torch_cuda_version"] = torch.version.cuda
except Exception as e:
    info["torch_import_error"] = str(e)
with open("machine_summary.json","w") as f:
    json.dump(info, f, indent=2)
print("Wrote machine_summary.json")
PY
echo "peparing ember dataset"
curl -fL https://ember.elastic.co/ember_dataset_2018_2.tar.bz2 | tar -xj -C ./artifcat/data

_crun python  - <<'PY'
import inspect, importlib, pathlib, sys

# locate the installed file
features = importlib.import_module("ember.features")
path = pathlib.Path(inspect.getfile(features))
txt = path.read_text()

# the buggy call wraps a single string like: transform([raw_obj['entry']])
fixed = txt.replace(
    'FeatureHasher(50, input_type="string").transform([raw_obj[\'entry\']])',
    'FeatureHasher(50, input_type="string").transform([[raw_obj[\'entry\']]])'
)

if txt == fixed:
    print("Nothing replaced (file may already be patched):", path)
else:
    path.write_text(fixed)
    print("Patched:", path)
PY

_crun python - <<'PY'
import ember
ember.create_vectorized_features("./artifact/data/ember2018/")
PY
echo "==> Installation complete."
echo
echo "Use the env WITHOUT activation (safest, base Python untouched):"
echo "  ${CONDA_BIN} run -n ${ENV_NAME} python -c 'import torch; print(torch.__version__)'"
echo
echo "If you prefer to activate later, do it in a sanitized subshell to avoid PYTHONPATH issues:"
echo "  env -u PYTHONPATH PYTHONNOUSERSITE=1 CONDA_NO_PLUGINS=1 ${CONDA_BIN} shell.bash hook >/tmp/conda_hook.sh"
echo "  source /tmp/conda_hook.sh && conda activate ${ENV_NAME}"
echo
echo "Lite run:        ${CONDA_BIN} run -n ${ENV_NAME} bash run_lite.sh"
echo "Full run (GPU):  ${CONDA_BIN} run -n ${ENV_NAME} bash run_full.sh"
