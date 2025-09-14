#!/usr/bin/env bash
set -euo pipefail

# --- Move to repo artifact root (adjust if needed) ---
cd ..; cd ..; cd artifact

# Use non-interactive matplotlib backend
MPLBACKEND=Agg
# ===== Config =====
ENV_NAME="${ENV_NAME:-deepprov-env}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
PY_BIN="${PY_BIN:-$CONDA_HOME/envs/$ENV_NAME/bin/python}"
# ==================

# Sanitize so user site-packages can't interfere
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --------- USER CONFIG (edit as needed) ---------
DATASET="${DATASET:-mnist}"
MODEL_NAME="${MODEL_NAME:-mnist_1}"
FOLDER="${FOLDER:-Ground_Truth_pth}"
MODEL_TYPE="${MODEL_TYPE:-pytorch}"
EPOCHS="${EPOCHS:-25}"
EXPLA_MODE="${EXPLA_MODE:-Saliency}"
# ------------------------------------------------

# ---------- Args ----------
ATTACK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--attack)
      ATTACK="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage:
  $(basename "$0")                  # benign-only (no attack)
  $(basename "$0") --attack FGSM    # run once for ATTACK=FGSM

Optional env vars:
  ENV_NAME, CONDA_HOME, PY_BIN, DATASET, MODEL_NAME, FOLDER, MODEL_TYPE, EPOCHS, EXPLA_MODE
EOF
      exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 2 ;;
  esac
done
# --------------------------

# Check interpreter
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

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# 0) Always build benign graphs once (no attack flag)
log "Step 0: Building benign graphs (no attack)"
run activations_extractor.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -model_type "$MODEL_TYPE" \
  -task graph



# ---- Single-attack pipeline (no loop) ----
log "=================================================="
log "Processing ATTACK = $ATTACK"
log "=================================================="

# 1) Generate adversarial graphs for this attack
log "Step 1: Generating adversarial graphs for $ATTACK"
run activations_extractor.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -model_type "$MODEL_TYPE" \
  -task graph \
  -attack "$ATTACK"

# 2) Train GNN on generated graphs (per attack)
log "Step 2: Training GNN for $ATTACK (epochs=$EPOCHS)"
run train_on_graph.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -model_type "$MODEL_TYPE" \
  -task graph \
  -attack "$ATTACK" \
  -epochs "$EPOCHS" \
  -save True

# Derived model path matches your convention:
# models/GNN_<model_name>_<attack>_<model_type>
MODEL_PATH="models/GNN_${MODEL_NAME}_${ATTACK}_${MODEL_TYPE}"

# Make a per-attack attribution output folder to avoid overwrites
ATTR_DIR="data/attributions_data/${ATTACK}"

# 3a) Explain GNN (benign mode)
log "Step 3a: Explaining GNN (benign) for $ATTACK"
run train_on_graph.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -model_type "$MODEL_TYPE" \
  -task GNN_explainer \
  -expla_mode "$EXPLA_MODE" \
  -model_path "$MODEL_PATH" \
  -attr_folder "$ATTR_DIR"

# 3b) Explain GNN with the attack context
log "Step 3b: Explaining GNN (attack=$ATTACK) for $ATTACK"
run train_on_graph.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -model_type "$MODEL_TYPE" \
  -task GNN_explainer \
  -attack "$ATTACK" \
  -expla_mode "$EXPLA_MODE" \
  -model_path "$MODEL_PATH" \
  -attr_folder "$ATTR_DIR"

# 4) Repairing the model using the dumped attributions
log "Step 4: Repair model for $ATTACK"
run main.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -attack "$ATTACK" \
  -expla_mode "$EXPLA_MODE" \
  -ben_thresh 97 \
  -attr_folder "$ATTR_DIR"

log "DONE: $ATTACK"
echo "[OK] Full pipeline completed. Check expected folders per claim."
