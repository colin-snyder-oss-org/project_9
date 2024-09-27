# src/utils/quantization.py
import torch
import torch.quantization as quantization

def prepare_model_for_quantization(model):
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    quantization.prepare_qat(model, inplace=True)
    return model

def convert_model_to_quantized(model):
    quantized_model = quantization.convert(model.eval(), inplace=False)
    return quantized_model
