# src/utils/export_onnx.py
import torch
import torch.onnx
from models.sparse_cnn import SparseCNN

def export_model_to_onnx(model, input_size, onnx_file_path):
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    print(f"Model exported to ONNX format at '{onnx_file_path}'")
