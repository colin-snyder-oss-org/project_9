# tests/test_models.py
import unittest
import torch
from models.sparse_cnn import SparseCNN

class TestModels(unittest.TestCase):
    def test_sparse_cnn_forward(self):
        model = SparseCNN(num_classes=10)
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()
