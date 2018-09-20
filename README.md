# CUDA + ATen for pytorch
- relu_cuda_kernel.cu -- just naive cuda file, but contains tensor 
- relu_cuda.cpp -- pybind11 
- jit.py -- you can just import module from this file 
or python3 setup.py install 
- setup.py -- build or install the lib
- relu.py -- layer module example, within gradient check
- test_relu.py -- a simple regression example contained the custom relu op.

## Usage example
'''
import torch
from jit import relu_cuda

...
x = torch.randn(5, device=torch.device('cuda'), dtype = torch.float)
y = relu_cuda.forward(x)
...
'''
another example:
'''
import torch
from relu import MyReLU

...
relu = MyReLU() #just a relu layer


'''
