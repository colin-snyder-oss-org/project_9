# src/training/validate.py
import torch
import torch.nn as nn
from src.data.data_loader import get_data_loader
from models.sparse_cnn import SparseCNN
from src.utils.model_utils import load_model
from src.utils.quantization import convert_model_to_quantized

def validate_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseCNN(num_classes=config['num_classes']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    model, optimizer, _ = load_model(model, optimizer, config['model_checkpoint'], device)

    # Convert model to quantized version
    quantized_model = convert_model_to_quantized(model)
    quantized_model.to(device)
    quantized_model.eval()

    criterion = nn.CrossEntropyLoss()
    val_loader = get_data_loader(config, mode='val')

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = quantized_model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Validation Loss: {total_loss/len(val_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")
