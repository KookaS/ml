import torch
from torch import einsum

class Mlp:

    def __init__(self):
        self.params = {}
        self.activations = []

    def load_checkpoint(self, params):
        self.params = {
            'layer_in/weights': params['layer_in/weights'],
            'layer_out/weights': params['layer_out/weights'],
        }
    
    def forward(self, x):
        """
        Z[B, F] = In[B, D] @ Win[D, F]
        A[B, F] = Activation (Z)
        Out [B, D] = A[B, F] @ Wout[F, D]
        """
        
        self.activations.append(x)
        z = einsum('bd,df->bf', x, self.params['layer_in/weights'])
        self.activations.append(z)
        a = torch.nn.functional.relu(z)
        out = einsum('bf,fd->bd', a, self.params['layer_out/weights'])
        return out

    def backward(self, out_grad):
        """
        d is partial derivative, go layer by layer

        dWout[F, D] = A.T[F, B] @ dOut[B, D]
        
        dA[B, F] = dOut[B, D] @ Wout.T[D, F]
        dZ[B, F] = dA[B, F] * Act'(Z)[B, F] (same dimension, dot-product, order does not matter)
        dWin[D, F] = In.T[D, B] @ dZ[B, F]

        Relu derivative = d (x * (x>0)) /dx = x > 0
        """
        z = self.activations.pop()
        a = torch.nn.functional.relu(z)
        w_out_grad = einsum('bf,bd->fd', a, out_grad)

        a_grad = einsum('bd,fd->bf', out_grad, self.params['layer_out/weights'])
        
        z_grad = a_grad * (z > 0)
        x = self.activations.pop()
        w_in_grad = einsum('bd,bf->df', x, z_grad)

        return {'layer_out/weights': w_out_grad, 'layer_in/weights': w_in_grad}


B, D, F = 8, 64, 256

if __name__ == "__main__":

    torch.manual_seed(42)
    x = torch.randn(B, D)
    y = torch.randn(B, D)
    params = {
        'layer_in/weights': torch.randn(D, F, requires_grad=True),
        'layer_out/weights': torch.randn(F, D, requires_grad=True),
    }

    model = Mlp()
    model.load_checkpoint(params)

    y_hat = model.forward(x)
    out_grad = y-y_hat
    grads = model.backward(out_grad)

    