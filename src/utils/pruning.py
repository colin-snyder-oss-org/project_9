# src/utils/pruning.py
import torch.nn.utils.prune as prune
import torch

def apply_structured_pruning(model, amount=0.5):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
