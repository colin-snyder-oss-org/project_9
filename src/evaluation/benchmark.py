# src/evaluation/benchmark.py
import time
import torch
from src.data.data_loader import get_data_loader
from models.sparse_cnn import SparseCNN

def benchmark_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseCNN(num_classes=config['num_classes']).to(device)
    model.eval()

    data_loader = get_data_loader(config, mode='val')
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += end_time - start_time
            total_images += inputs.size(0)

    avg_time_per_image = total_time / total_images
    print(f"Average inference time per image: {avg_time_per_image*1000:.2f} ms")
