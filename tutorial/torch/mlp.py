import torch
from torch import einsum

class Mlp:

    def __init__(self, d_model, d_ff):
        self.activations = []
        # init the weights for the optimizer
        self.w_in = torch.zeros((d_model, d_ff), dtype=torch.float32)
        self.w_out = torch.zeros((d_ff, d_model), dtype=torch.float32)

    def load_checkpoint(self, params):
        # refill the empty tensors
        self.w_in[...] = params['layer_in/weights'][...]
        self.w_out[...] = params['layer_out/weights'][...]
    
    def forward(self, x):
        """
        Z[B, F] = X[B, D] @ Win[D, F]
        A[B, F] = Activation (Z)
        Out [B, D] = A[B, F] @ Wout[F, D]
        """
        self.activations = []
        self.activations.append(x)
        z = einsum('bd,df->bf', x, self.w_in.to(dtype=torch.bfloat16)) # weights bf16 could be stored in activations
        self.activations.append(z)
        a = torch.nn.functional.relu(z)
        out = einsum('bf,fd->bd', a, self.w_out.to(dtype=torch.bfloat16)) # weights bf16 could be stored in activations
        return out

    def backward(self, out_grad):
        """
        d is partial derivative, go layer by layer

        dWout[F, D] = A.T[F, B] @ dOut[B, D]
        
        dA[B, F] = dOut[B, D] @ Wout.T[D, F]
        dZ[B, F] = dA[B, F] * Act'(Z)[B, F] (same dimension, dot-product, order does not matter)
        dWin[D, F] = X.T[D, B] @ dZ[B, F]

        Relu derivative = d (x * (x>0)) /dx = x > 0

        To compute the input gradient for deeper networks (not required for the first layer)
        dX[B, D] = dZ[B, F] @ Win.T[F, D]
        """
        z = self.activations.pop()
        a = torch.nn.functional.relu(z)
        w_out_grad = einsum('bf,bd->fd', a, out_grad)

        a_grad = einsum('bd,fd->bf', out_grad, self.w_out.to(dtype=torch.bfloat16))
        z_grad = a_grad * (z > 0)
        x = self.activations.pop()
        w_in_grad = einsum('bd,bf->df', x, z_grad)

        x_grad = einsum('bf,df->bd', z_grad, self.w_in.to(dtype=torch.bfloat16))

        return {'layer_out/weights': w_out_grad, 'layer_in/weights': w_in_grad, 'input': x_grad}


if __name__ == "__main__":
    B, D, F = 8, 64, 256

    torch.manual_seed(42)
    x = torch.randn(B, D, dtype=torch.bfloat16)
    y = torch.randn(B, D, dtype=torch.bfloat16)
    params = {
        'layer_in/weights': torch.randn(D, F, dtype=torch.float32),
        'layer_out/weights': torch.randn(F, D, dtype=torch.float32),
    }

    model = Mlp(D, F)
    model.load_checkpoint(params)

    out = model.forward(x)

    # # Calculate Loss here
    # loss = (out - target).pow(2).mean()
    # print(f"Loss: {loss.item()}")
    # grad_out = 2 * (out - target) / B

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.bfloat16)
    grads = model.backward(grad_out)

    print(f"Gradient W_in shape: {grads['layer_in/weights'].shape}")
    print(f"Gradient W_out shape: {grads['layer_out/weights'].shape}")
    print(f"Gradient X shape: {grads['input'].shape}")
