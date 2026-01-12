import torch

class _NormRmsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, eps=1e-6):
        """
        Root Mean Square normalization.
        Layer norm of transformers is done on head dimension.

        Formula:
            y = x * gamma / r
            r = sqrt( eps + sum(x^2) / N), where N is the dim of x
        
        :param x: Input [B, S, N, H]
        :param gain: scaling factor
        :param eps: optional denominator value for numerical stability
        """
        rms = torch.sqrt(eps + torch.sum(x**2, dim=-1, keepdim=True) / x.shape[-1])

        ctx.save_for_backward(x, gamma, rms)

        return x * gamma / rms

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for RMS Prop.

        Formula:
            y = n(x)  / r(x), n(x) =x*gamma, r(x) = rms(x)
            where i is index of contracting dimension            

            1. derivative for gamma dL/dgamma
            gamma is only shape [H], so we need to sum all other dimensions

            dL/dgamma   = dL/dy * dy/dgamma
                        = grad_out *  d(x * gamma / r)/d_x
                                    = grad_out *  x/r
            --> dL/dgamma = sum_{B,S,N}(grad_out *  x/r [B, S, N, H]])
            
            2. derivative along input dL/dx
            2.1 dL/dx {r} = sum(dL/dy * dy/dr) * dr/dx
            2.2 dL/dx {n} = dL/dy * dy/dn * dn/dx
            dL/dy = grad_out
            
            --> 2.1
            dy/dr = -x * g / r^2

            r = sqrt( eps + sum(x^2) / N) = sqrt(u)
            dr/dx = d_sqrt(u) /dx
                    = 1/2 * 1/sqrt(u) * du/dx
                    = 1/(2*r) * 2x/N
                    = x/(N* r)

            dL/dx   = sum(grad_out * -x * g / r^2) * x/(N* r)
            Note the sum is required because r [B, S, N, 1] is broadcased to fit x [B, S, N, H], we must collapse x last dim H into 1.

            --> 2.2
            dn/dx = g
            dy/dn = 1/r
            dL/dx = grad_out * 1/r * g
            Note here that there is no sum because all elements have the same shape [B, S, N, H]

            --> total
            dL/dx = grad_out * 1/r * g + sum(grad_out * -x * g / r^2) * x/(N* r)
        
        :param x: grad_out is dL/dy [B, S, N, H]
        :return: d_gamma is dL/dgamma [H]
        :return: d_x is dL/dx [B, S, N, H]
        """
        x, gamma, rms = ctx.saved_tensors

        # 1. backprop for gamma weights
        # sum along all dimensions except the last one [H]
        # dL/dgamma = sum_{B,S,N}(grad_out *  x/r [B, S, N, H]])
        d_gamma = (grad_out * x / rms).sum(dim=tuple(range(x.ndim - 1)))

        # 2. backprop for input
        # dL/dx = grad_out * 1/r * g + sum(grad_out * -x * g / r^2) * x/(N * r)
        d_x = grad_out * gamma/rms - (grad_out * x * gamma / rms**2).sum(dim=-1, keepdims=True) * x / (x.shape[-1] * rms)

        return d_x, d_gamma, None
    
class NormRms(torch.nn.Module):

    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones((head_dim,)))
    
    def forward(self, x):
        return _NormRmsFn.apply(x, self.weights, self.eps)


if __name__ == "__main__":

    # Create random input [Batch, Seq, Num, Head]
    B, S, N, H = 2, 5, 4, 8
    x = torch.randn(B, S, N, H, requires_grad=True)

    # 1. Apply your custom function
    model = NormRms(H)
    output = model.forward(x)

    # 2. Create a dummy loss to trigger backprop
    loss = output.sum()

    # 3. Compute gradients
    loss.backward()

    print("Custom Gradient shape:", x.grad.shape)

    # --- Verification ---
    # Let's compare against PyTorch's built-in softmax to ensure correctness
    x_ref = x.clone().detach().requires_grad_(True)
    out_ref = torch.nn.functional.rms_norm(x_ref, normalized_shape=(x.shape[-1],), weight=model.weights, eps=model.eps)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad, x_ref.grad)

    diff = torch.max(torch.abs(x.grad - x_ref.grad))
    print(f"Both implementations are nearly identical: {diff.item()}")