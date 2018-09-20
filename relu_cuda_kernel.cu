#include "relu_cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>


namespace light {
template <typename DType>
__device__ __forceinline__ DType relu(DType x) {
    return fmax(0.0, x);
}

template <typename DType>
__device__ __forceinline__ DType d_relu(DType x) {
    return x < 0.0 ? 0.0: 1.0;
}

template <typename DType>
__global__ void relu_cuda_forward_kernel(const DType * __restrict__ input,
        DType * __restrict__ output, const int size){
    const uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) {
        output[id] = relu(input[id]);
    }
}

template <typename DType>
__global__ void relu_cuda_backward_kernel(const DType * __restrict__ input,
        const DType * __restrict__ in_diff, DType * __restrict__ out_diff, const int size) {
    const uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size) {
        out_diff[id] = in_diff[id] * d_relu(input[id]);
    }
}

//
at::Tensor relu_cuda_forward(at::Tensor& input) {
    const auto size = input.numel();

    const int threads = 1024;
    const int blocks = (size + threads-1)/threads;

    auto output = input.clone();
    AT_DISPATCH_FLOATING_TYPES(at::CUDA(at::kFloat), "relu_forward_cuda", ([&] {
    relu_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(),
            output.data<scalar_t>(), size);
    }));

    return output;
}

at::Tensor relu_cuda_backward(at::Tensor& input, at::Tensor &in_diff){
    auto out_diff = in_diff.clone();
    const auto size = input.numel();
    
    const int threads = 1024;
    const int blocks = (size + threads-1)/threads;
    AT_DISPATCH_FLOATING_TYPES(at::CUDA(at::kFloat), "relu_backward_cuda", ([&] {
    relu_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(),
            in_diff.data<scalar_t>(), out_diff.data<scalar_t>(), size);
    }));

    return out_diff;
}


} //! namespace light
