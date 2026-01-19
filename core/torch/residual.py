import torch

class _ResidualFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_residual, x_prev):
        return x_residual + x_prev

    @staticmethod
    def backward(ctx, grad_out):
        """
        For backward we just return the gradient to each element.

        Forward: y = a + b
        Backward:
            dL/dx = dL/dy * dy/da + dL/dy * dy/db
            dy/da = dy/db = 1
        """
        return grad_out, grad_out

class Residual(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x_residual, x_prev):
        return _ResidualFn.apply(x_residual, x_prev)