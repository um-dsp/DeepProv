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

# Sanitize this process so user site-packages can't interfere
export PYTHONNOUSERSITE=1
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

#!/usr/bin/env bash
set -euo pipefail

# --------- USER CONFIG (edit as needed) ---------
DATASET="mnist"
MODEL_NAME="mnist_1"
FOLDER="Ground_Truth_pth"
MODEL_TYPE="pytorch"
EPOCHS=25
EXPLA_MODE="Saliency"

# List your six attacks here (use your exact names)
ATTACKS=(FGSM PGD APGD-DLR Square SPSA SIT)
# ------------------------------------------------

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# 0) Prepare benign graphs ONCE (no attack flag)
log "Step 0: Building benign graphs (no attack)"
run activations_extractor.py \
  -dataset "$DATASET" \
  -model_name "$MODEL_NAME" \
  -folder "$FOLDER" \
  -model_type "$MODEL_TYPE" \
  -task graph

# Loop over each attack
for ATTACK in "${ATTACKS[@]}"; do
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

  # Derived model path matches your example:
  # models/GNN_<dataset>_<model_name>_<attack>_<model_type>
  MODEL_PATH="models/GNN_${MODEL_NAME}_${ATTACK}_${MODEL_TYPE}"

  # Make a per-attack attribution output folder to avoid overwrites
  ATTR_DIR="data/attributions_data/${ATTACK}"

  # 3) Explain GNN and dump attributions (benign mode)
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

  # 3b) Explain GNN with the attack context (if you want both)
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
done

log "All attacks completed successfully."


echo "[OK] Full pipeline completed Check expected folders per claim." 

