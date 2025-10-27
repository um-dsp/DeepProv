# DeepProv: Behavioral Characterization and Repair of Neural Networks via Inference Provenance Graph Analysis

  
This repository contains the  reprodicable implementation of our ACSAC 2025 paper **[DeepProv: Behavioral Characterization and Repair of Neural Networks via
Inference Provenance Graph Analysis](https://arxiv.org/pdf/2509.26562)**. DeepProv is a novel system that captures a DNN's inference on an input as an inference provenance graph (IPGs). Using the IPG representation, it captures the computational information flow/causality of inference. It then uses IPGs of different settings (e.g., that of benign and adversarial) to empirically and structurally characterize IPGs. Finally, using the IPG characterizations as inference provenance insights, it systematically repairs the DNN given a repair goal (e.g., adversarial robustness). With adversarial robustness as the repair goal, we demonstrate its effectiveness on image classification (MNIST and CIFAR-10) and malware detection (EMBER and CuckooTraces).<br />

### Datasets, Pre-trained Models, and Attacks

- **Datasets**:
  - MNIST is automatically loaded via Keras.
  - CIFAR-10 is automatically loaded via Keras.
  - EMBER is automatically set up via `install.sh`.
  - The Cuckoo CSV file is present in `artifact/data/`.

- **Pre-trained Models**: We provide pre-trained models (`mnist`, `cuckoo_1`, `ember_1`, `cifar10_2`). These models are available in `artifact/models/`.

- **Supported Attacks**: As of now, *DeepProv* supports the following attacks:<br />
  - **MNIST**: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Auto-PGD with DLR loss (APGD-DLR), Square, and SIT.<br />
  - **CIFAR10**: Fast Gradient Sign Method (FGSM). We skip the rest of the attacks since they are demonstrated in MNIST and they are heavy, they will take more than 1400 minutes as stated in the paper appendix about the computation overhead.
Due to RAM size restrictions, we pre-computed the graphs for cifar10 and did the emperical and structural analysis as well as we generated the benign distribution to repair the model and saved these proxies to showcase the repair part only for cifar10 on FGSM,<br />
  - **CuckooTraces**: *Bit-Flip* — incrementally flips `0` bits to `1`, starting from the first one in a feature vector until model evasion is observed.<br />
  - **EMBER**: *Emb-Att* — incrementally perturbs features within valid value ranges/options until model evasion is observed.<br />

More details about the supported attacks are provided in the paper.

### Installation
## Clone the repo (with Git LFS)

This repository uses **Git LFS** for large artifacts (models, pickles, precomputed graphs). To get the full repository including all large files, follow these steps.

### Clone over HTTPS (requires your own GitHub token)
```bash
# 1) Install Git LFS (one-time setup on your machine)
git lfs install

# 2) Clone the repository
git clone https://github.com/um-dsp/DeepProv.git
cd DeepProv

# 3) Pull large files tracked by LFS (you may be prompted to authenticate)
git lfs pull
```
When prompted for a password, **use your own GitHub Personal Access Token (PAT)** (not your password).
- For public repositories, a token with **Read access to contents** is sufficient.
- You can create a fine-grained token at [https://github.com/settings/tokens](https://github.com/settings/tokens).

### Verify LFS
To list the files tracked by LFS:
```bash
git lfs ls-files
```

To install the requirements and set up the environment to run the experiments:
```bash
chmod 0755 install.sh
./install.sh
```
To reproduce the paper’s results, visit the Claims/ folder and run the corresponding run.sh files.

For the public infrastructure, we support Google Colab. To reproduce the experiments use the link to the  notebook on google colab in the file infrastructure/info.txt
