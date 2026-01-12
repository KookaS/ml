import torch
from torch import einsum

class _MlpFn(torch.autograd.Function):
    """
    Mlp with two linear layers without biases, with ReLU activation in between.
    """

    @staticmethod   
    def forward(ctx, x, w_in, w_out):
        """
        Z[B, F] = X[B, D] @ Win[D, F] + Bin[D, F]
        A[B, F] = Activation (Z)
        Out [B, D] = A[B, F] @ Wout[F, D] + Bout[F, D]
        """
        z = einsum('bd,df->bf', x, w_in)
        a = torch.nn.functional.relu(z)
        out = einsum('bf,fd->bd', a, w_out)
        ctx.save_for_backward(x, z, w_in, w_out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        d is partial derivative, go layer by layer
        dL/dOut = grad_Out

        1. Backprop through Wout
        dL/dWout = dL/dOut * dOut/dWout
        grad_Wout[F, D] = A.T[F, B] @ grad_Out[B, D]
        
        2. Backprop through Win
        dL/dWin = dL/dOut * dOut/dA * dA/dZ * dZ/dWin
        dOut/dA = Wout
        dA/dZ = Act'(Z) = d (z * (z>0)) /dz = (z > 0)
        dZ/dWin = X
        grad_Z[B, F] = (grad_Out[B, D] @ Wout.T[D, F]) * (z > 0)
        grad_Win[D, F] = X.T[D, B] @ grad_Z[B, F]
        Note: with einsum the order does not matter

        3. Backprop through X
        To compute the input gradient for deeper networks (not required for the first layer)
        dL/dx = dL/dOut * dOut/dA * dA/dZ * dZ/dX (we reuse previous results)
        grad_X[B, D] = grad_Z[B, F] @ Win.T[F, D]
        """
        x, z, w_in, w_out = ctx.saved_tensors
        a = torch.nn.functional.relu(z)
        w_grad_out = einsum('bf,bd->fd', a, grad_out)

        a_grad = einsum('bd,fd->bf', grad_out, w_out)
        z_grad = a_grad * (z > 0)
        w_grad_in = einsum('bd,bf->df', x, z_grad)

        grad_x = einsum('bf,df->bd', z_grad, w_in)

        return grad_x, w_grad_in, w_grad_out

class Mlp(torch.nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        # init the weights for the optimizer
        self.w_in = torch.nn.Parameter(torch.empty((d_model, d_ff)))
        self.w_out = torch.nn.Parameter(torch.empty((d_ff, d_model)))

        torch.nn.init.xavier_normal_(self.w_in)
        torch.nn.init.xavier_normal_(self.w_out)

    def load_checkpoint(self, params):
        # refill the empty tensors
        with torch.no_grad():
            self.w_in.copy_(params['w_in'])
            self.w_out.copy_(params['w_out'])

    def forward(self, x):
        return _MlpFn.apply(x, self.w_in, self.w_out)

if __name__ == "__main__":
    B, D, F = 8, 64, 256
    torch.manual_seed(42)

    # --- Setup Data ---
    # Input X
    x = torch.randn(B, D, requires_grad=True)
    
    # Weights (Shared between both models)
    w_in_init = torch.randn(D, F)
    w_out_init = torch.randn(F, D)

    # Gradient from the "future" layer
    grad_upstream = torch.randn(B, D)

    # Custom model
    model_custom = Mlp(D, F)
    model_custom.load_checkpoint({'w_in': w_in_init, 'w_out': w_out_init})
    x_custom = x.clone().detach().requires_grad_(True)
    out_custom = model_custom(x_custom)
    out_custom.backward(grad_upstream)

    # Reference model
    class ReferenceMlp(torch.nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_ff, bias=False), # We implemented bias=False
                torch.nn.ReLU(),
                torch.nn.Linear(d_ff, d_model, bias=False)
            )
        def forward(self, x):
            return self.net(x)

    model_ref = ReferenceMlp(D, F)
    # Load weights (Transposed because nn.Linear stores as [Out, In])
    with torch.no_grad():
        model_ref.net[0].weight.copy_(w_in_init.T)
        model_ref.net[2].weight.copy_(w_out_init.T)

    x_ref = x.clone().detach().requires_grad_(True)
    out_ref = model_ref(x_ref)
    out_ref.backward(grad_upstream)

    # Comparison 
    print(f"{'='*20} COMPARISON RESULTS {'='*20}")

    # Check Forward Pass
    diff_out = (out_custom - out_ref).abs().max().item()
    print(f"Forward Output Max Diff:  {diff_out:.2e}  [{'OK' if diff_out < 1e-6 else 'FAIL'}]")

    # Check Input Gradients (dL/dX)
    diff_dx = (x_custom.grad - x_ref.grad).abs().max().item()
    print(f"Input Gradient Max Diff:  {diff_dx:.2e}  [{'OK' if diff_dx < 1e-6 else 'FAIL'}]")

    # Check Weight Gradients
    grad_w_in_ref = model_ref.net[0].weight.grad.T
    grad_w_out_ref = model_ref.net[2].weight.grad.T

    diff_w_in = (model_custom.w_in.grad - grad_w_in_ref).abs().max().item()
    print(f"W_in Gradient Max Diff:   {diff_w_in:.2e}  [{'OK' if diff_w_in < 1e-6 else 'FAIL'}]")

    diff_w_out = (model_custom.w_out.grad - grad_w_out_ref).abs().max().item()
    print(f"W_out Gradient Max Diff:  {diff_w_out:.2e}  [{'OK' if diff_w_out < 1e-6 else 'FAIL'}]")