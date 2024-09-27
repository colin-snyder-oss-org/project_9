# src/deployment/export_tvm.py
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor

def export_model_with_tvm(onnx_model_path, target, target_host, input_shape, output_path):
    onnx_model = onnx.load(onnx_model_path)
    shape_dict = {'input': input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    lib.export_library(output_path)
    print(f"Model compiled with TVM and saved to '{output_path}'")
