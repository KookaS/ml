import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from tutorial.torch.mlp import Mlp

class MlpTp(Mlp):

    def __init__(self, rank, d_model, d_ff):
        super().__init__()
        self.rank = rank
        self.d_model = d_model
        self.d_ff = d_ff

    def load_checkpoint(self, params):
        """
        X[B, D] @ Win[D, Fy] @y Wout[Fy, D]
        """
        local_d_ff = self.d_ff // dist.get_world_size()
        start = local_d_ff * self.rank
        end = start + local_d_ff
        self.w_in = params['layer_in/weights'][:, start:end].detach().clone().to(f"cpu:{self.rank}")
        self.w_out = params['layer_out/weights'][start:end, :].detach().clone().to(f"cpu:{self.rank}")
    
    def forward(self, x):
        """
        Z[B, Fy] = X[B, D] @ Win[D, Fy]
        A[B, Fy] = Activation (Z)
        Out[B, D]{Uy} = A[B, Fy] @y Wout[Fy, D]
        Out[B, D] = AllReduce(Out[B, D]{Uy})
        """
        
        out = super().forward(x)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def backward(self, out_grad):
        """
        dWout[Fy, D] = A.T[Fy, B] @ dOut[B, D]
        
        dA[B, Fy] = dOut[B, D] @ Wout.T[D, Fy]
        dZ[B, Fy] = dA[B, Fy] * Act'(Z)[B, Fy]
        dWin[D, Fy] = X.T[D, B] @ dZ[B, Fy]

        dX[B, D]{Uy} = dZ[B, Fy] @ Win.T[Fy, D]
        dX[B, D] = AllReduce(dX[B, D]{Uy})
        """
        grads = super().backward(out_grad)

        x_grad = grads['input']
        dist.all_reduce(x_grad, op=dist.ReduceOp.AVG)
        grads['input'] = x_grad

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
        'layer_in/weights': torch.randn(D, F, dtype=torch.float32, device=f"cpu:{rank}"),
        'layer_out/weights': torch.randn(F, D, dtype=torch.float32, device=f"cpu:{rank}"),
    }
    model = MlpTp(rank, D, F)
    model.load_checkpoint(params)

    x = torch.randn(B, D, dtype=torch.float32, device=f"cpu:{rank}")
    out = model.forward(x)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.float32, device=f"cpu:{rank}")
    grads = model.backward(grad_out)
    # For weight updates we return sharded gradients

    # Verification
    if rank == 0:
        print(f"Rank {rank}: Manual Backward Complete.")
        print(f"Gradient W_in shape: {grads['layer_in/weights'].shape}")
        print(f"Gradient W_out shape: {grads['layer_out/weights'].shape}")
        print(f"Gradient X shape: {grads['input'].shape}")
        # It should work without hanging

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.spawn(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    

    