from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_cnn_edge',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='sparse_conv_cuda',
            sources=[
                'models/custom_layers/sparse_ops/sparse_conv_cuda.cpp',
                'models/custom_layers/sparse_ops/sparse_conv_cuda_kernel.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'tqdm',
        'onnx',
        'onnxruntime',
        'pillow',
        'matplotlib',
        'scikit-learn',
        'pytest',
        'pycuda',
        'pyopencl',
        'Cython',
    ],
    author='Your Name',
    author_email='youremail@example.com',
    description='Sparse CNNs for real-time object detection on edge devices',
    url='https://github.com/yourusername/sparse-cnn-edge-detection',
)
