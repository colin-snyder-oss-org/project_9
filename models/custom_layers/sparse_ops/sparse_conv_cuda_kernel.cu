// models/custom_layers/sparse_ops/sparse_conv_cuda.cpp
#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
std::vector<torch::Tensor> sparse_conv_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    torch::Tensor mask);

std::vector<torch::Tensor> sparse_conv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    torch::Tensor mask);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

std::vector<torch::Tensor> sparse_conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    torch::Tensor mask) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(mask);
    return sparse_conv_cuda_forward(input, weight, bias, stride, padding, mask);
}

std::vector<torch::Tensor> sparse_conv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    torch::Tensor mask) {
    CHECK_CUDA(grad_output);
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(mask);
    return sparse_conv_cuda_backward(grad_output, input, weight, bias, stride, padding, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_conv_forward, "Sparse Convolution forward (CUDA)");
    m.def("backward", &sparse_conv_backward, "Sparse Convolution backward (CUDA)");
}
