# src/data/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class CustomDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.annotations = self._load_annotations(annotations_file)
        self.transform = transform

    def _load_annotations(self, annotations_file):
        # Load annotations
        # Placeholder for actual implementation
        return []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.annotations[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loader(config, mode='train'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std']),
        ]),
        'val': transforms.Compose([
            transforms.Resize(config['input_size']),
            transforms.CenterCrop(config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std']),
        ]),
    }

    dataset = CustomDataset(
        data_dir=config['data'][mode]['data_dir'],
        annotations_file=config['data'][mode]['annotations_file'],
        transform=data_transforms[mode]
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True if mode == 'train' else False,
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return data_loader
