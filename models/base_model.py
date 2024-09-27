# models/base_model.py
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def initialize_weights(self):
        pass
