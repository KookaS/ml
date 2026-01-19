import torch

class _LinearFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, b):
        """
        Forward pass for Linear layer with bias.

        Formula:
            y = x[B, D] @ w[D, F] + b[F]
        
        :param x: Input
        :param w: Weight
        :param b: Bias
        """
        ctx.save_for_backward(x, w, b)
        out = torch.einsum('...d,df->...f', x, w)

        if b is not None:
            out += b # broadcast b to match other dimensions

        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for Linear layer with bias.

        Formulas:
        dL/dX = dL/dy * dy/dX
        grad_x[B,D] = grad_out[B, F] @ W.T[F, D]

        dL/dW = dL/dy * dy/dW
        grad_w[D, F] = X.T[D, B] @ grad_out[B, F]

        dL/dB = dL/dy * dy/dB
        grad_w[F] = mean_B(grad_out[B, F])
        
        
        :param ctx: Description
        :param grad_out: Description
        """
        x, w, b = ctx.saved_tensors
        grad_x = torch.einsum('...f,df->...d', grad_out, w)
        grad_w = torch.einsum('...d,...f->df', x, grad_out)
        grad_b = None
        if b is not None:
            grad_b = torch.sum(grad_out, dim=tuple(range(x.ndim-1)))

        return grad_x, grad_w, grad_b

class Linear(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((dim_in, dim_out)))
        torch.nn.init.xavier_normal_(self.weight)

        self.bias = torch.nn.Parameter(torch.zeros((dim_out,))) # best set at zeros
    
    def forward(self, x):
        return _LinearFn.apply(x, self.weight, self.bias)
    

if __name__ == "__main__":

    # Create random input
    B, S, D, F = 2, 5, 16, 32
    x = torch.randn(B, S, D, requires_grad=True)

    # 1. Apply your custom function
    model = Linear(D, F)
    output = model.forward(x)

    # 2. Create a dummy loss to trigger backprop
    loss = output.sum()

    # 3. Compute gradients
    loss.backward()

    print("Custom Gradient shape:", x.grad.shape)
    
    # --- Verification ---
    # Let's compare against PyTorch's built-in softmax to ensure correctness
    x_ref = x.clone().detach().requires_grad_(True)
    model_ref = torch.nn.Linear(D, F, bias=True)
    with torch.no_grad():
        model_ref.weight.copy_(model.weight.T) # they do [Out, In] for the weights
        model_ref.bias.copy_(model.bias)
    out_ref = model_ref(x_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad, x_ref.grad)

    diff = torch.max(torch.abs(x.grad - x_ref.grad))
    print(f"Both implementations are nearly identical: {diff.item()}")