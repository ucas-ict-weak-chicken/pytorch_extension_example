#include <torch/torch.h>

#include "relu_cuda.hpp"

using namespace light;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



at::Tensor relu_forward(at::Tensor & input) {
    CHECK_INPUT(input);
    return relu_cuda_forward(input);
}

at::Tensor relu_backward(at::Tensor& input, at::Tensor &in_diff){
    CHECK_INPUT(input);
    CHECK_INPUT(in_diff);
    return relu_cuda_backward(input, in_diff);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu_forward, "ReLU forward(CUDA)");
    m.def("backward", &relu_backward, "ReLU backward(CUDA)");
}
