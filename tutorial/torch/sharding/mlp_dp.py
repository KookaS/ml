import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os
class MlpDp:

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
        Z[Bx, F] = In[Bx, D] @ Win[D, F]
        A[Bx, F] = Activation (Z)
        Out [Bx, D] = A[Bx, F] @ Wout[F, D]
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

        dWout[F, D] = A.T[F, Bx] @x dOut[Bx, D]
        
        dA[Bx, F] = dOut[Bx, D] @ Wout.T[D, F]
        dZ[Bx, F] = dA[Bx, F] * Act'(Z)[Bx, F] (same dimension, order does not matter)
        dWin[D, F] = In.T[D, Bx] @x dZ[Bx, F]

        Relu derivative = d (x * (x>0)) /dx = x > 0
        """
        z = self.activations.pop()
        a = torch.nn.functional.relu(z)
        w_out_grad = einsum('bf,bd->fd', a, out_grad)
        dist.all_reduce(w_out_grad, op=dist.ReduceOp.SUM)

        a_grad = einsum('bd,fd->bf', out_grad, self.params['layer_out/weights'])
        
        z_grad = a_grad * (z > 0)
        x = self.activations.pop()
        w_in_grad = einsum('bd,bf->df', x, z_grad)
        dist.all_reduce(w_in_grad, op=dist.ReduceOp.SUM)

        return {'layer_out/weights': w_out_grad, 'layer_in/weights': w_in_grad}


B, D, F = 8, 64, 256

# --- The Runner ---
def worker_fn(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # Force Gloo to use the specific Windows Loopback interface.
    # This is the standard name for localhost networking on Windows.
    os.environ['GLOO_SOCKET_IFNAME'] = 'Loopback Pseudo-Interface 1'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Model Init
    torch.manual_seed(42)
    params = {
        'layer_in/weights': torch.randn(D, F, requires_grad=True),
        'layer_out/weights': torch.randn(F, D, requires_grad=True),
    }
    model = MlpDp()
    model.load_checkpoint(params)

    # Data Sharding
    B_local = B // world_size
    torch.manual_seed(rank) # Different data per rank
    x_local = torch.randn(B_local, D)
    y_local = torch.randn(B_local, D)

    out_local = model.forward(x_local)
    # simulated loss gradient (dLoss/dOut)
    grad_out_local = (out_local - y_local)
    grads = model.backward(grad_out_local)

    # Verification
    if rank == 0:
        print(f"Rank {rank}: Manual Backward Complete.")
        print(f"Gradient W_in shape: {grads['layer_in/weights'].shape}")
        # If this runs without hanging, the all_reduce worked.

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.spawn(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    

    