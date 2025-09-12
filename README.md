# DeepProv: Behavioral Characterization and Repair of Neural Networks via Inference Provenance Graph Analysis

  

In this repository we include code and notebooks related to *DeepProv* inference provenance capture and characterization for deep neural networks. Particularly, we instantiated *DeepProv* capture to be performed on benign and adversarial settings, serving the repair goal of robustness analysis. Inline with the paper, we consider multiple DNN models for image classifications (i.e., MNIST) and malware detection models (i.e., CuckooTraces, EMBER) <br  />

  

### Datasets, Pre-trained Models, and Attacks:

  

  

-  **Datasets**:

- MNIST is automatically loaded via Keras.

- CIFAR-10 automatically loaded via Keras.

- Ember automatically loaded via install.sh

- Cuckoo csv file present in artifcat/data.

  **Pre-trained Models**: We offer pre-trained models: mnist, cuckoo_1, and ember_1, cifar10_2. These models are available in the artifact/models.
  

-  **Supported Attacks**: As of now, DeepProv supports the following attacks : <br  />

- MNIST: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD) Auto PGD with DLR loss function (APGD-DLR) and Square , SPSA and SIT <br  />

- CuckooTraces: *Bit-Flip* --It incrementally flips `0` bits to `1`, starting by the first one in a feature vector until model evasion is observed <br  />

- EMBER: *Emb-Att* --it incrementally perturbs features within valid value ranges/options until model evasion is observed <br  />

More details about supported attacks are provided in the paper

### Insatallation:
To install the requirements and the convenient environment to run the experiments: 
```$ chmod 0755 install.sh ``` \
```$ ./install.sh ```

To reproduce the paper results visit claims folder and run all corresponding run.sh files


