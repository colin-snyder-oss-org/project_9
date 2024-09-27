# models/custom_layers/sparse_conv.py
import torch
import torch.nn as nn
from torch.autograd import Function
import sparse_conv_cuda

class SparseConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, mask):
        ctx.save_for_backward(input, weight, bias, mask)
        ctx.stride = stride
        ctx.padding = padding
        output = sparse_conv_cuda.forward(input, weight, bias, stride, padding, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input, grad_weight, grad_bias = sparse_conv_cuda.backward(
            grad_output, input, weight, bias, stride, padding, mask)
        return grad_input, grad_weight, grad_bias, None, None, None

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SparseConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return SparseConvFunction.apply(input, self.weight * self.mask, self.bias, self.stride, self.padding, self.mask)
