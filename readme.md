# Torch Inference Graph Extractor

This project allows you to generate inference provenance graphs from PyTorch models, either from:

* Pretrained torchvision models (e.g., ResNet, VGG, DenseNet, etc.), or
* Custom `.pth` models along with user-defined model architectures and input tensors.

The output is saved as a pickle file containing the PyTorch Geometric heterogeneous graph.

---

## Requirements

* Python 3.10+

### Install dependencies

```bash
pip install torch torchvision torch-geometric
```

---

## Usage

### 1. Pretrained Mode

Use a pretrained model from torchvision on IMAGENET, generate CIFAR-10-like input, and extract the graph.

Supported model names:
`resnet18`, `alexnet`, `vgg11`, `squeezenet`, `densenet121`, `mobilenet_v2`

#### Example

```bash
python main.py --mode pretrained --model resnet18 --out_dir ./graphs
```

#### Output

A file like `./graphs/resnet18_graph.pkl` will be created.

---

### 2. Custom Mode

Use your own `.pth` model with its architecture and a pickled input tensor.

#### Example

```bash
python generate_graph.py \
  --mode custom \
  --model_path ./checkpoints/mymodel.pth \
  --model_def ./models/mymodel.py \
  --model_class MyModel \
  --input_pickle ./inputs/input_tensor.pkl \
  --out_pickle ./graphs/mymodel_graph.pkl
```

#### Notes

* `model_def` should be a `.py` file containing the model architecture.
* `model_class` is the class name (case-sensitive) inside that `.py` file.
* `input_pickle` must contain a pre-saved PyTorch tensor (e.g., `torch.randn(1, 3, 224, 224)`).

---

## Loading a Saved Graph

Use the script `load_graph.py`:
Example:
python load_graph.py "graphs/resnet18_graph.pkl"
