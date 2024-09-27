# src/deployment/edge_inference.py
import tvm
from tvm.contrib import graph_executor
import numpy as np
from PIL import Image
from torchvision import transforms

def run_edge_inference(config):
    # Load the compiled module
    lib = tvm.runtime.load_module(config['tvm_compiled_model'])
    device = tvm.device(config['device'], 0)
    module = graph_executor.GraphModule(lib['default'](device))

    # Preprocess the input image
    image = Image.open(config['input_image']).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.CenterCrop(config['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])
    input_tensor = preprocess(image).unsqueeze(0).numpy()

    # Set input and run inference
    module.set_input('input', tvm.nd.array(input_tensor.astype('float32')))
    module.run()
    output = module.get_output(0).asnumpy()
    predicted_class = np.argmax(output)

    print(f"Predicted class: {predicted_class}")
