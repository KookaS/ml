import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from tutorial.torch.mlp import Mlp

class MlpDp(Mlp):

    def forward(self, x):
        """
        Z[Bx, F] = X[Bx, D] @ Win[D, F]
        A[Bx, F] = Activation (Z)
        Out [Bx, D] = A[Bx, F] @ Wout[F, D]
        """
        self.activations = []
        return super().forward(x)

    def backward(self, out_grad):
        """
        dWout[F, D]{Ux} = A.T[F, Bx] @x dOut[Bx, D]
        dWout[F, D] = AllReduce_x(dWout[F, D]{Ux})
        
        dA[Bx, F] = dOut[Bx, D] @ Wout.T[D, F]
        dZ[Bx, F] = dA[Bx, F] * Act'(Z)[Bx, F]
        dWin[D, F]{Ux} = X.T[D, Bx] @x dZ[Bx, F]
        dWin[D, F] = AllReduce_x(dWin[D, F]{Ux})

        dX[Bx, F] = dZ[Bx, F] @ Win.T[F, D]
        """
        grads = super().backward(out_grad)

        w_in_grad = grads['layer_in/weights']
        w_out_grad = grads['layer_out/weights']

        # average the gradients over all the batches
        # each device has a chunk of dimension [F,D]
        # but only holds parts of the end result
        w_out_grad = w_out_grad.contiguous()
        w_in_grad = w_in_grad.contiguous()
        dist.all_reduce(w_out_grad, op=dist.ReduceOp.AVG)
        dist.all_reduce(w_in_grad, op=dist.ReduceOp.AVG)
        grads['layer_in/weights'] = w_in_grad
        grads['layer_out/weights'] = w_out_grad

        return grads

# --- The Runner ---
def worker_fn(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    B, D, F = 8, 64, 256

    device_type = "cpu"
    device = f"{device_type}:{rank}"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Model Init
    torch.manual_seed(42)
    params = {
        'layer_in/weights': torch.randn(D, F, dtype=torch.float32),
        'layer_out/weights': torch.randn(F, D, dtype=torch.float32),
    }
    model = MlpDp()
    model.load_checkpoint(params, device=device)

    # Data Sharding
    B_local = B // world_size
    start = rank * B_local
    end = start + B_local

    x = torch.randn(B, D, dtype=torch.bfloat16)
    x_local = x[start:end, :].detach().clone().to(device)
    out_local = model.forward(x_local)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.bfloat16)
    grad_out_local = grad_out[start:end, :].detach().clone().to(device)
    grads = model.backward(grad_out_local)

    # Verification
    if rank == 0:
        print(f"--- Simulation on {device_type.upper()} ---")
        print(f"Rank {rank}: TP Backward Complete.")
        # Check Shapes
        print(f"Grad Win: {grads['layer_in/weights'].shape} (Expected: {D}, {F})")
        print(f"Grad Wout: {grads['layer_out/weights'].shape} (Expected: {F}, {D})")
        print(f"Grad X:   {grads['input'].shape} (Expected: {B_local}, {D})")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.spawn(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
