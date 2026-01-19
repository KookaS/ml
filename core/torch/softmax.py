import torch

class _SoftmaxFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, dim=-1):
        """
        Softmax forward pass in a numerically stable way, done on the last dimension.

        Formula:
            s = e^x / sum(e^x)
        
        :param x: Input [B, S, N, H]
        :return: Output [B, S, N, H]
        """
        emax = torch.max(x, dim=dim, keepdim=True).values # prevent overflow
        ex = torch.exp(x - emax)
        s = ex / torch.sum(ex, dim=dim, keepdim=True)

        ctx.save_for_backward(s)
        ctx.s_dim = dim

        return s
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Softmax backward pass.
        We do the partial derivative of the logarithm and rearragne.

        Jacobian Formula:
            s = softmax(x)
            d log(s_i) /dx_j = 1/s * ds_i / dx_j -- simplification --> dlog(s) = 1/s * ds
            ds = s * dlog(s)

            log(s) = log(exp^x / sum(exp^x)) = log(exp^x) - log(sum(exp^x)) = x - log(sum(exp^x))
            dlog(s) = dx - dlog(sum(exp^x))

            dx = dx_i / dx_j = 1{i==j} --> dx = 1 if i==j else 0
            --> dlog(s) = 1{i==j} - dlog(sum(exp^x))

            dlog(sum(exp^x)) = 1/sum(exp^x) * dsum(exp^x)
            dsum(exp^x) = dsum(exp^x_i) / dx_j = exp^x{i==j}, because dexp^x = exp^x
            --> dlog(sum(exp^x)) = exp_{x_j}/sum(exp^x) = s_j
            --> dlog(s) = 1{i==j} - s_j

            --> ds = s * (1{i==j} - s_j)
            ds_i / dx_j = s_i * (1{i==j} - s_j)
            OR
            ds_i/dx_j = s_i * (dx_i/dx_j - s_j)

        Jacobian:
            J_{ij} = {
                s_i * (1 - s_i), if i==j
                -s_i * s_j, if i!=j
            }
            J_{ij} = s_i * (gamma_{ij} - s_j), where gamma is either 1 (i=j) or 0 (i!=j)

        Deep Learning:
            we don't compute the Jacobian, we compute the gradient of the loss with respect to the input.
            we use Vector-Jacobian Product, O(N).

            grad_input = s * (grad_output - sum(grad_output * s))
            The VJP formula can be derived from the Jacobian.
            Without VJP optimization, we would multiply the gradient to the jacobian, O(N^2).

        VJP Formula:
            h_j = grad_input --> what we aim to find
            g_j = dL/ds_i --> grad_output
            h_j = sum_i(dL/ds_i * ds_i/dx_j) = sum_i(g_i * J_{ij})
            h_j = sum_i(g_i * s_i * (gamma_{ij} - s_j))
                = sum_i(g_i * s_i * gamma_{ij})  - sum_i(g_i * s_i * s_j)
                = g_j * s_j - s_j * sum_i(g_i * s_i)
                = s_j * (g_j - sum_i(g_i * s_i))

        Final Formula:
            grad_input = s * (grad_output - sum(grad_output * s))

        
        :param x: grad_output is dL/ds [B, S, N, H]
        :return: grad_input is dL/dx [B, S, N, H]
        """
        s, = ctx.saved_tensors # unpack tuple
        s_dim = ctx.s_dim

        term = (grad_out * s).sum(dim=s_dim, keepdim=True)
        grad_input = s * (grad_out - term)
        
        # Return gradients for both inputs (x, dim)
        return grad_input, None



def softmax(x, dim=-1):
    """
    Call softmax as a function.
    
    :param x: Input
    :param dim: Dimension on which to operate, default -1.
    """
    return _SoftmaxFn.apply(x, dim)

def d_softmax(grad_out, s, dim=-1):
    """
    Call softmax as a function.
    
    :param x: Input
    :param dim: Dimension on which to operate, default -1.
    """
    class MockContext:
        def __init__(self, saved_tensors, s_dim):
            self.saved_tensors = saved_tensors
            self.s_dim = s_dim
    ctx = MockContext(saved_tensors=(s,), s_dim=dim)
    dx, _ = _SoftmaxFn.backward(ctx, grad_out)
    return dx

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim=-1):
        """
        Call softmax as a torch.nn module.
        
        :param x: Input
        :param dim: Dimension on which to operate, default -1.
        """
        return _SoftmaxFn.apply(x, dim)

if __name__ == "__main__":

    # Create random input [Batch, Seq, Num, Head]
    B, S, N, H = 2, 5, 4, 8
    x = torch.randn(B, S, N, H, requires_grad=True)

    # 1. Apply your custom function
    output = softmax(x, dim=-1)

    # 2. Create a dummy loss to trigger backprop
    loss = output.sum()

    # 3. Compute gradients
    loss.backward()

    print("Custom Gradient shape:", x.grad.shape)

    # --- Verification ---
    # Let's compare against PyTorch's built-in softmax to ensure correctness
    x_ref = x.clone().detach().requires_grad_(True)
    out_ref = torch.nn.functional.softmax(x_ref, dim=-1)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad, x_ref.grad)

    diff = torch.max(torch.abs(x.grad - x_ref.grad))
    print(f"Both implementations are nearly identical: {diff.item()}")