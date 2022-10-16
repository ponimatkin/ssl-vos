#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14', '-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_61,code=compute_61',
    '-gencode', 'arch=compute_70,code=compute_70',
    '-ccbin', '/usr/bin/gcc'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args, 'cuda-path': ['/gpfslocalsys/cuda/11.2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })