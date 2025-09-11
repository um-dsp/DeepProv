# DeepProv: Behavioral Characterization and Repair of Neural Networks via Inference Provenance Graphs

  

In this repository we include code and notebooks related to *DeepProv* inference provenance capture and characterization for deep neural networks. Particularly, we instantiated *DeepProv* capture to be performed on benign and adversarial settings, serving the repair goal of robustness analysis. Inline with the paper, we consider multiple DNN models for image classifications (i.e., MNIST) and malware detection models (i.e., CuckooTraces, EMBER) <br  />

  

### Datasets, Pre-trained Models, and Attacks:

  

  

-  **Datasets**:

- MNIST is automatically loaded via Keras.

- CIFAR-10 needs a manual python library update which is pytroch_gemoetric to get attributions for heterongenous graphs. The issue is fixed in the github repo of pytorch_geometric but still

under testing and conflicts issues. But this is does not affect the correctness and reliability of CIFAR-10 results since we fixed the issue locally to establish the experiemnts.

- To test *DeepProv* on malware data, you need to download the CuckooTraces ([link here](https://drive.google.com/file/d/1RAMqh5VVlmSR-Z6mVbiHCti5viyntqwS/view?usp=sharing)) and EMBER ([link here](https://ember.elastic.co/ember_dataset_2018_2.tar.bz2)) datasets and add them in the folder `./data/`. By default DeepProv will look for them in that path<br  />

  -  **Complementary Data**: Needed to reproduce experiments. Please download the files and place them in the data folder created before of the project ([link here](https://drive.google.com/drive/folders/1trvTg5dUrSlfewM-gbG7UtHqcWUNcfCC?usp=sharing))

  

-  **Pre-trained Models**: We offer pre-trained models: mnist_1 , mnist_2, mnist_3, cuckoo_1, and ember_1. These models are available to download [here](https://drive.google.com/drive/folders/1ps2N57PSGscZkq1k4VERovd0a1B3lZDC?usp=sharing). Once downloaded to `DeepProv/models/` directory, the `model.txt` file has the model architecture details of each model.

  

-  **Supported Attacks**: As of now, DeepProv supports the following attacks : <br  />

- MNIST: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD) Auto PGD with DLR loss function (APGD-DLR) and Square , SPSA is not included in the public shared code because it is computationally expensive and requires significant human intervention to use efficiently, such as manually saving and loading the generated attack samples throughout the experiments <br  />

- CuckooTraces: *Bit-Flip* --It incrementally flips `0` bits to `1`, starting by the first one in a feature vector until model evasion is observed <br  />

- EMBER: *Emb-Att* --it incrementally perturbs features within valid value ranges/options until model evasion is observed <br  />

More details about supported attacks are provided in the paper

  

  

***

  

  

### Insatallation:

  

Clone the project,

  

```$ cd DeepProv ```

  

  

```$ pip install -r requirements.txt ```

  

***

  

  

### Inference Provenance Graph Extraction:

  

  

-  ` activations_extractor.py`: The first step for our characterization approach is to extract activations of the target model `model`. It takes the following parameters in the given order:

  

  

-  **Dataset Name**: mnist | cuckoo |ember <br  />

  

-  **Pre-Trained Model Name**: cuckoo_1 | ember_1 | mnist_1 <br  />

  

-  **Folder**: Ground_Truth | Benign | Adversarial <br>

  

It is required to use these exact folder names. This parameter sets what folder will the generated activations be saved, the default file path is

  

`folder/dataset_name/model_name/<attack>/`.

  

(**Note**: Make Sure to create the folder with the above path before running the activation generation)

  

-  **Attack Name**: FGSM | APGD-DLR | PGD |square|CKO |EMBER | None <br  />

  

(**Note**: This parameter is optional. if specified, DeepProv will apply the attack on the dataset. if attack is None and the folder input is set to adversarial it will throw an Error. -stop parameter is to precise the number of batchs of graph to generate (1000 graphs per batch) <br  />

  

-  **tasks**: default is to get emperical characterization and graph for the extrating graphs from the model inputs.<br  />

  

  

(**Note**: if attack is None and the folder input is set to adversarial it will throw an Error/ use -stop parameter to precise the number of batchs of graphs when task==graph to generate (1000 graphs per batch) <br  />

  

**Sample Commands :**  <br  />

  

>  `python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task default`  <br  />

  

>  ` python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Benign_pth -model_type pytorch -task default`  <br  />

  

>  ` python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Benign_pth -model_type pytorch -task default -attack FGSM`  <br  />

  

>  `python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task graph `  <br  />

  

>  `python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task graph -attack FGSM`  <br  />

  

  

***

  

The model activations of ground truth, test benign and adversarial data are stored in each respective folder in text files named as [true label]\_[predicted label]\_[index] (e.g., 0_0_150.txt). Each file contains the values of every node in every layer of the model for that specific sample.

  

  

### Empirical Characterization:

  

Use [Empirical_Characterization_Ember_Cuckoo.ipynb](/Empirical_Characterization_Ember_Cuckoo.ipynb) and [Empirical_Characterization_MNIST.ipynb](/Empirical_Characterization_MNIST.ipynb) to compute the proposed graph-related metrics for empirical characterization.

  

  

---

  

  

#### Graph Neural Network Model: training a model `graph_model` on the extracted graph data

  

  

` train_on_graph.py`: We train another model called `graph_model` that learns the GNN of the target model `model`. This model should be also stored and should be initiated and stored in a .pth file <br  />

  

  

To Train and test a predefined model on the activations set use the CLI command with the following parameters: Dataset Name, Model Name and Attack

  

(The Above arguments will just be used to locate the needed activations in the project folder).

  

The command also expects the following arguments: <br  />

  

- Model Path : represents the path to save the feature extraction model that is trained to seperate activations of benign and adversarial samples <br/>

  

(**Note**: We have pre-trained models on Ground_Truth benign and FGSM data, and tested on Benign and FGSM test data).

  

  

**Sample Commands:**  <br  />

  

To train GNN and save it using the generated graphs use the following command : <br  />  <br  />

  

>  `python train_on_graph.py -dataset mnist -model_name mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -attack FGSM -epochs 5 -save True`  <br  />

  

  

To explain the GNN and visualize the structred attributions of the graphs use the following commands : <br  />  <br  />

  

>  `python train_on_graph.py -dataset mnist -model_name mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_2_FGSM_pytorch -attack FGSM -expla_mode Saliency -attr_folder data/attributions_data/`  <br  />

  

>  `python train_on_graph.py -dataset mnist -model_name mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_2_FGSM_pytorch -expla_mode Saliency -attr_folder data/attributions_data/ `  <br  />

  

  

#### Attribution: Perform ML explanation on the `graph_model` to identify relevant nodes.

  

  

-  ` gen_attributions.py`: this file explains how to transform generated activations to dataset and train an torch adversarial detection model. ` attributionUtils.py` holds different predefined architecture that cover all the dataset we research and produce satisfactory performance.

  

` Attributions :` in the same file we showcase the steps to generate the attributions of the models on a batch of input,

  

---

  

-  ` Attribution.py`: The thrid step: is to perform **Attributions** on the trained `graph_model`. This file should include a fucntion that takes as input a `graph_model` and `data_name`, the model should be imported automaticcaly from `Models/GNN_[model_name]_[attack]_[model_type]`. Example: `Models/GNN_mnist_1_FGSM_pytorch.pth`

  

***

  

### Structural Characterization

  

Use [Structural_Characterization.ipynb](/Structural_Characterization.ipynb) to compute the proposed graph-related metrics for structural characterization.

  

  

***

  

### Robustness enhancement model repair

  

After generating the empirical and structural characterizations, you can run the model repair algorithm for the goal of robustness enhancement using an example of the following commands by specifying the attack, the dataset and the benign threshold for the model at hand.<br  />  <br  />

  

>  `python3 main.py -dataset ember -model_name ember_1 -folder Ground_Truth_pth -attack EMBER -expla_mode Saliency -ben_thresh 90 -attr_folder data/attributions_data/`  <br  />

  

### Paper plots:

  

After running the whole framework on a specific dataset and attack, use [PaperPlots.ipynb](/PaperPlots.ipynb) to reproduce the paper figures.
