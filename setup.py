from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = 'relu_cuda',
    ext_modules = [CUDAExtension(
        'relu_cuda', ['relu_cuda.cpp', 'relu_cuda_kernel.cu']
        )],
    cmdclass = {
        'build_ext': BuildExtension
    })
