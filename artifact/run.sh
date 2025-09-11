#!/usr/bin/env bash
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

# 1) Generate ground-truth graphs (can be large)
# run activations_extractor.py \
#   -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph

# # 2) Generate adversarial graphs (SIT shown)
# run activations_extractor.py \
#   -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph -attack SIT 

# # 3) Train GNN on generated graphs (increase epochs as needed)
# run train_on_graph.py \
#   -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph -attack SIT -epochs 5 -save True

# # 4) (Optional) Explain GNN and dump attributions
# run train_on_graph.py \
#   -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task GNN_explainer \
#   -expla_mode Saliency -model_path models/GNN_mnist_1_SIT_pytorch \
#   -attr_folder data/attributions_data/

# run train_on_graph.py \
#   -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task GNN_explainer -attack SIT \
#   -expla_mode Saliency -model_path models/GNN_mnist_1_SIT_pytorch \
#   -attr_folder data/attributions_data/

# 5) (Optional) EMBER repair example from README
# run main.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth \
#   -attack SIT -expla_mode Saliency -ben_thresh 90 \
#   -attr_folder data/attributions_data/

# echo "[OK] Full pipeline completed (example MNIST + SIT). See ./runs and ./models for outputs."


#!/usr/bin/env bash
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

# 1) Generate ground-truth graphs (can be large)
# run activations_extractor.py \
#   -dataset ember -model_name ember_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph

# # 2) Generate adversarial graphs (SIT shown)
# run activations_extractor.py \
#   -dataset ember -model_name ember_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph -attack EMBER 

# # 3) Train GNN on generated graphs (increase epochs as needed)
# run train_on_graph.py \
#   -dataset ember -model_name ember_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph -attack EMBER -epochs 5 -save True

# # 4) (Optional) Explain GNN and dump attributions
# run train_on_graph.py \
#   -dataset ember -model_name ember_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task GNN_explainer \
#   -expla_mode Saliency -model_path models/GNN_ember_1_EMBER_pytorch \
#   -attr_folder data/attributions_data/

# run train_on_graph.py \
#   -dataset ember -model_name ember_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task GNN_explainer -attack EMBER \
#   -expla_mode Saliency -model_path models/GNN_ember_1_EMBER_pytorch \
#   -attr_folder data/attributions_data/

# 5) (Optional) EMBER repair example from README
# run main.py -dataset ember -model_name ember_1 -folder Ground_Truth_pth \
#   -attack EMBER -expla_mode Saliency -ben_thresh 90 \
#   -attr_folder data/attributions_data/

# echo "[OK] Full pipeline completed (example MNIST + SIT). See ./runs and ./models for outputs."

# 1) Generate ground-truth graphs (can be large)
# run activations_extractor.py \
#   -dataset cuckoo -model_name cuckoo_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph

# # # 2) Generate adversarial graphs (SIT shown)
# run activations_extractor.py \
#   -dataset cuckoo -model_name cuckoo_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph -attack CKO 

# # # 3) Train GNN on generated graphs (increase epochs as needed)
# run train_on_graph.py \
#   -dataset cuckoo -model_name cuckoo_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task graph -attack CKO -epochs 5 -save True

# # # 4) (Optional) Explain GNN and dump attributions
# run train_on_graph.py \
#   -dataset cuckoo -model_name cuckoo_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task GNN_explainer \
#   -expla_mode Saliency -model_path models/GNN_cuckoo_1_CKO_pytorch \
#   -attr_folder data/attributions_data/

# run train_on_graph.py \
#   -dataset cuckoo -model_name cuckoo_1 -folder Ground_Truth_pth \
#   -model_type pytorch -task GNN_explainer -attack CKO \
#   -expla_mode Saliency -model_path models/GNN_cuckoo_1_CKO_pytorch \
#   -attr_folder data/attributions_data/

# 5) (Optional) EMBER repair example from README
run main.py -dataset cuckoo -model_name cuckoo_1 -folder Ground_Truth_pth \
  -attack CKO -expla_mode Saliency -ben_thresh 92 \
  -attr_folder data/attributions_data/
