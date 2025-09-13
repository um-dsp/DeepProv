# DeepProv: Behavioral Characterization and Repair of Neural Networks via Inference Provenance Graph Analysis

  

In this repository, we include code and notebooks related to *DeepProv*—inference provenance capture and characterization for deep neural networks. In particular, we instantiate *DeepProv* to capture provenance under both benign and adversarial settings, supporting robustness analysis and model repair. In line with the paper, we consider multiple DNN models for image classification (e.g., MNIST) and malware detection models (e.g., CuckooTraces, EMBER).<br />

### Datasets, Pre-trained Models, and Attacks

- **Datasets**:
  - MNIST is automatically loaded via Keras.
  - CIFAR-10 is automatically loaded via Keras.
  - EMBER is automatically set up via `install.sh`.
  - The Cuckoo CSV file is present in `artifact/data/`.

- **Pre-trained Models**: We provide pre-trained models (`mnist`, `cuckoo_1`, `ember_1`, `cifar10_2`). These models are available in `artifact/models/`.

- **Supported Attacks**: As of now, *DeepProv* supports the following attacks:<br />
  - **MNIST**: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Auto-PGD with DLR loss (APGD-DLR), Square, and SIT.<br />
  - **CIFAR10**: Fast Gradient Sign Method (FGSM) We skip the rest of the attacks since they are demonstrated in MNIST and they are heavy, they will take more than 1400 minutes as stated in the paper appendix about the computation overhead.
Due to RAM size restrictions, we pre-computed the graphs for cifar10 and did the emperical and structural analysis as well as we generated the benign distribution to repair the model and saved these proxies to showcase the repair part only for cifar10 on FGSM,<br />
  - **CuckooTraces**: *Bit-Flip* — incrementally flips `0` bits to `1`, starting from the first one in a feature vector until model evasion is observed.<br />
  - **EMBER**: *Emb-Att* — incrementally perturbs features within valid value ranges/options until model evasion is observed.<br />

More details about the supported attacks are provided in the paper.

### Installation
To install the requirements and set up the environment to run the experiments:
```bash
chmod 0755 install.sh
./install.sh
```
To reproduce the paper’s results, visit the Claims/ folder and run the corresponding run.sh files.

