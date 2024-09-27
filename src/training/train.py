# src/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data.data_loader import get_data_loader
from models.sparse_cnn import SparseCNN
from src.utils.model_utils import save_model
from src.utils.dynamic_sparsity import DynamicSparsity
from src.utils.pruning import apply_structured_pruning
from src.utils.quantization import prepare_model_for_quantization

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseCNN(num_classes=config['num_classes'], sparsity_config=config['sparsity']).to(device)

    # Prepare model for quantization-aware training
    model = prepare_model_for_quantization(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    train_loader = get_data_loader(config, mode='train')
    val_loader = get_data_loader(config, mode='val')

    dynamic_sparsity = DynamicSparsity(config['sparsity'])

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply dynamic sparsity
            inputs = dynamic_sparsity.apply_sparsity(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        # Apply structured pruning at specified epochs
        if epoch in config['prune_epochs']:
            apply_structured_pruning(model, amount=config['prune_amount'])

        scheduler.step()

        # Save model checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            save_model(model, optimizer, epoch, f"checkpoints/model_epoch_{epoch+1}.pth")
