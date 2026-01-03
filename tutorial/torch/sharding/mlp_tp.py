import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from tutorial.torch.mlp import Mlp

class MlpTp(Mlp):

    def __init__(self, rank, world_size, d_model, d_ff):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.d_model = d_model
        self.d_ff = d_ff

    def load_checkpoint(self, params):
        """
        In[B, D] @ Win[D, Fy] @y Wout[Fy, D]
        """
        local_d_ff = self.d_ff // self.world_size
        start = local_d_ff * self.rank
        end = start + local_d_ff
        self.params = {
            'layer_in/weights': params['layer_in/weights'][:, start:end],
            'layer_out/weights': params['layer_out/weights'][start:end, :],
        }
    
    def forward(self, x):
        """
        Z[B, Fy] = In[B, D] @ Win[D, Fy]
        A[B, Fy] = Activation (Z)
        Out[B, D]{Uy} = A[B, Fy] @y Wout[Fy, D]
        Out[B, D] = AllReduce_y(Out[B, D]{Uy})
        """
        
        out = super().forward(x)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def backward(self, out_grad):
        """
        dWout[Fy, D] = A.T[Fy, B] @ dOut[B, D]
        
        dA[B, Fy] = dOut[B, D] @ Wout.T[D, Fy]
        dZ[B, Fy] = dA[B, Fy] * Act'(Z)[B, Fy]
        dWin[D, Fy] = In.T[D, B] @ dZ[B, Fy]
        """
        grads = super().backward(out_grad)

        # If this MLP is part of a deeper network, we must 
        # All-Reduce the input gradient so the previous layer 
        # gets the correct signal.

        return grads


B, D, F = 8, 64, 256

# --- The Runner ---
def worker_fn(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Model Init
    torch.manual_seed(42)
    params = {
        'layer_in/weights': torch.randn(D, F, requires_grad=True),
        'layer_out/weights': torch.randn(F, D, requires_grad=True),
    }
    model = MlpTp(rank, world_size, D, F)
    model.load_checkpoint(params)

    x = torch.randn(B, D)
    out = model.forward(x)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D)
    grads = model.backward(grad_out)
    # For weight updates we return sharded gradients

    # Verification
    if rank == 0:
        print(f"Rank {rank}: Manual Backward Complete.")
        print(f"Gradient W_in shape: {grads['layer_in/weights'].shape}")
        # If this runs without hanging, the all_reduce worked.

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.spawn(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    

    