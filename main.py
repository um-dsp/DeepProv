import os
import sys
import torch
import pickle
import argparse
import importlib.util
from torchvision.models import (
    resnet18, alexnet, vgg11, squeezenet1_0, densenet121, mobilenet_v2
)

from Graph_Extraction import GenericInferenceGraph  # <-- Replace with actual module

model_dict = {
    "resnet18": resnet18(weights='IMAGENET1K_V1'),
    "alexnet": alexnet(weights='IMAGENET1K_V1'),
    "vgg11": vgg11(weights='IMAGENET1K_V1'),
    "squeezenet": squeezenet1_0(weights='IMAGENET1K_V1'),
    "densenet121": densenet121(weights='IMAGENET1K_V1'),
    "mobilenet_v2": mobilenet_v2(weights='IMAGENET1K_V1'),
}

parser = argparse.ArgumentParser(description="Generate inference graph")
parser.add_argument('--mode', type=str, choices=['pretrained', 'custom'], required=True, help='Mode to run the script')
parser.add_argument('--model', type=str, help='Pretrained model name (used only if mode=pretrained)')
parser.add_argument('--out_dir', type=str, default='./graphs', help='Directory to save output graph (for pretrained)')
parser.add_argument('--model_path', type=str, help='.pth file path (used only if mode=custom)')
parser.add_argument('--model_def', type=str, help='Path to model definition .py file (used only if mode=custom)')
parser.add_argument('--model_class', type=str, help='Model class name in the model definition (used only if mode=custom)')
parser.add_argument('--input_pickle', type=str, help='Pickled input tensor path (used only if mode=custom)')
parser.add_argument('--out_pickle', type=str, help='Output pickle file path (for custom model)')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

if args.mode == 'pretrained':
    if args.model not in model_dict:
        raise ValueError(f"Invalid model name. Choose from: {list(model_dict.keys())}")
    
    model = model_dict[args.model].eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    graph_builder = GenericInferenceGraph(model, device='cpu')
    graph_builder.extract_activations(dummy_input)
    hetero_graph = graph_builder.build_hetero_graph()

    out_path = os.path.join(args.out_dir, f"{args.model}_graph.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(hetero_graph, f)
    print(f"[+] Pretrained model graph saved to: {out_path}")

elif args.mode == 'custom':
    required_fields = [args.model_path, args.model_def, args.model_class, args.input_pickle, args.out_pickle]
    if not all(required_fields):
        raise ValueError("Missing required arguments for custom mode.")

    spec = importlib.util.spec_from_file_location("custom_model", args.model_def)
    custom_model_module = importlib.util.module_from_spec(spec)
    sys.modules["custom_model"] = custom_model_module
    spec.loader.exec_module(custom_model_module)

    ModelClass = getattr(custom_model_module, args.model_class)
    model = ModelClass()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    with open(args.input_pickle, 'rb') as f:
        input_tensor = pickle.load(f)

    graph_builder = GenericInferenceGraph(model, device='cpu')
    graph_builder.extract_activations(input_tensor)
    hetero_graph = graph_builder.build_hetero_graph()

    with open(args.out_pickle, 'wb') as f:
        pickle.dump(hetero_graph, f)
    print(f"[+] Custom model graph saved to: {args.out_pickle}")
