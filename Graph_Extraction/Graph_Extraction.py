import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import ast

class ForwardCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "self":
                self.calls.append(node.func.attr)
        self.generic_visit(node)

def get_all_weighted_modules(model):
    weighted_layers = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'weight') and isinstance(mod.weight, torch.Tensor):
            weighted_layers.append((name, mod))
    return weighted_layers

def extract_interblock_weights(model):
    weighted_layers = get_all_weighted_modules(model)
    weights_between = {}

    for i in range(len(weighted_layers) - 1):
        src_name, src_mod = weighted_layers[i]
        dst_name, dst_mod = weighted_layers[i + 1]
        key = f"{src_name}->{dst_name}"
        weights_between[key] = (src_mod.weight.detach().cpu(), dst_mod.weight.detach().cpu())

    weights_between[f"{weighted_layers[-1][0]}->output"] = None
    return weights_between

def get_activation_thresholds():
    return {
        nn.ReLU: 1e-5,
        nn.Sigmoid: 0.5,
        nn.Tanh: 0.1,
        nn.LeakyReLU: 1e-4
    }

def is_node_active(tensor, threshold):
    return tensor.abs().max() > threshold

class GenericInferenceGraph:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.activations = []
        self.layer_names = []
        self.modules = []
        self.module_to_index = {}

    def _hook_fn(self, module, input, output):
        self.activations.append(output.detach().cpu())
        self.layer_names.append(str(module))
        self.modules.append(module)

    def _register_hooks(self):
        index = 0
        self.hooks = []
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
                self.module_to_index[module] = index
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                index += 1

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def extract_activations(self, x):
        self.activations = []
        self.layer_names = []
        self.modules = []
        self._register_hooks()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x.to(self.device))
        self._remove_hooks()

    def build_hetero_graph(self):
        data = HeteroData()
        weights_map = extract_interblock_weights(self.model)
        thresholds = get_activation_thresholds()
        active_mask = []

        for i, (act, mod) in enumerate(zip(self.activations, self.modules)):
            flat_act = act.squeeze(0)
            act_type = type(mod)
            threshold = thresholds.get(act_type, 1e-5)
            active = torch.tensor([is_node_active(node, threshold) for node in flat_act])
            active_mask.append(active)
            data[f"Layer_{i}_{tuple(act.shape[1:])}"].x = flat_act
            data[f"Layer_{i}_{tuple(act.shape[1:])}"].active_mask = active

        for i in range(len(self.activations) - 1):
            src = self.activations[i].squeeze(0)
            dst = self.activations[i + 1].squeeze(0)
            src_type = f"Layer_{i}_{tuple(self.activations[i].shape[1:])}"
            dst_type = f"Layer_{i+1}_{tuple(self.activations[i+1].shape[1:])}"

            src_mask = active_mask[i]
            dst_mask = active_mask[i + 1]
            active_src_idx = torch.where(src_mask)[0]
            active_dst_idx = torch.where(dst_mask)[0]

            edge_index = torch.cartesian_prod(active_src_idx, active_dst_idx).T

            weight_key = list(weights_map.keys())[i] if i < len(weights_map) else None
            weight_pair = weights_map.get(weight_key)
            if weight_pair is not None:
                _, dst_w = weight_pair
                edge_attr = dst_w.reshape(-1, *dst_w.shape[2:]) if dst_w.ndim == 4 else dst_w.reshape(-1, 1)
            else:
                edge_attr = torch.ones(edge_index.size(1), 1)

            data[(src_type, "connects", dst_type)].edge_index = edge_index
            data[(src_type, "connects", dst_type)].edge_attr = edge_attr

        return data

