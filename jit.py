from torch.utils.cpp_extension import load
relu_cuda = load(
        'relu_cuda', ['relu_cuda.cpp','relu_cuda_kernel.cu'], verbose = True)

#help(relu_cuda)
