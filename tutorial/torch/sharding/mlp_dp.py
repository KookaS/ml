import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from tutorial.torch.mlp import Mlp

class MlpDp(Mlp):

    def forward(self, x):
        """
        Z[Bx, F] = In[Bx, D] @ Win[D, F]
        A[Bx, F] = Activation (Z)
        Out [Bx, D] = A[Bx, F] @ Wout[F, D]
        """
        super().forward(x)

    def backward(self, out_grad):
        """
        dWout[F, D]{Ux} = A.T[F, Bx] @x dOut[Bx, D]
        dWout[F, D] = AllReduce_x(dWout[F, D]{Ux})
        
        dA[Bx, F] = dOut[Bx, D] @ Wout.T[D, F]
        dZ[Bx, F] = dA[Bx, F] * Act'(Z)[Bx, F]
        dWin[D, F]{Ux} = In.T[D, Bx] @x dZ[Bx, F]
        dWin[D, F] = AllReduce_x(dWin[D, F]{Ux})
        """
        grads = super().backward(out_grad)

        w_out_grad = grads['layer_in/weights']
        w_in_grad = grads['layer_out/weights']

        # average the gradients over all the batch
        dist.all_reduce(w_out_grad, op=dist.ReduceOp.AVG)
        dist.all_reduce(w_in_grad, op=dist.ReduceOp.AVG)

        return {'layer_out/weights': w_out_grad, 'layer_in/weights': w_in_grad}


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
    model = MlpDp()
    model.load_checkpoint(params)

    # Data Sharding
    B_local = B // world_size
    start = rank * B_local
    end = start + B_local

    x = torch.randn(B, D)
    x_local = x[start:end, :]
    out_local = model.forward(x_local)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D)
    grad_out_local = grad_out[start:end, :]
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

    

    