# src/utils/model_utils.py
import torch
import os

def save_model(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)

def load_model(model, optimizer, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{path}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{path}'")
        start_epoch = 0
    return model, optimizer, start_epoch
