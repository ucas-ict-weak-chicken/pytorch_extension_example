
from torch import nn
from torch.autograd import Function
import torch

from jit import relu_cuda

torch.manual_seed(42)

class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return relu_cuda.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        grad_input = relu_cuda.backward(input, grad_output)
        return grad_input


class MyReLU(nn.Module):
    def __init__(self):
        super(MyReLU, self).__init__()

    def forward(self, input):
        return ReLUFunction.apply(input)


if __name__ == '__main__':
    #gradient check

    from torch.autograd import gradcheck
    w = torch.randn(10, device = torch.device('cuda'), requires_grad = True, dtype=torch.float)
    X = (w,)
    test = gradcheck(ReLUFunction.apply, X, eps=1e-4)
    print('test gradient:', test)
