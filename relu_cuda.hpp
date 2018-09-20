#ifndef __RELU_CUDA_HPP
#define __RELU_CUDA_HPP


#include <ATen/ATen.h>

namespace light {
at::Tensor relu_cuda_forward(at::Tensor & input);
at::Tensor relu_cuda_backward(at::Tensor& input, at::Tensor &in_diff);
}

#endif
