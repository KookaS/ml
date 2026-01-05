import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from tutorial.torch.mlp import Mlp

class MlpDp(Mlp):

    def __init__(self, rank, d_model, d_ff, device):
        self.activations = []
        self.rank = rank
        self.d_model = d_model
        self.d_ff = d_ff
        # init the weights for the optimizer
        self.w_in = torch.zeros((d_model, d_ff), dtype=torch.float32, device=device)
        self.w_out = torch.zeros((d_ff, d_model), dtype=torch.float32, device=device)

    def load_checkpoint(self, params):
        # refill the empty tensors
        self.w_in[...] = params['layer_in/weights'][...]
        self.w_out[...] = params['layer_out/weights'][...]

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
        dWout[F, D]{Ux} = A.T[F, Bx] @ dOut[Bx, D]
        dWout[F, D] = AllReduce_x(dWout[F, D]{Ux})
        
        dA[Bx, F] = dOut[Bx, D] @ Wout.T[D, F]
        dZ[Bx, F] = dA[Bx, F] * Act'(Z)[Bx, F]
        dWin[D, F]{Ux} = X.T[D, Bx] @ dZ[Bx, F]
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

    device_type = "cpu"
    device = f"{device_type}:{rank}"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    B, D, F = 8, 64, 256

    # Model Init
    torch.manual_seed(42)
    params = {
        'layer_in/weights': torch.randn(D, F, dtype=torch.float32),
        'layer_out/weights': torch.randn(F, D, dtype=torch.float32),
    }
    model = MlpDp(rank, D, F, device)
    model.load_checkpoint(params)

    # Data Sharding
    B_local = B // world_size
    start = rank * B_local
    end = start + B_local

    x = torch.randn(B, D, dtype=torch.bfloat16) # global unsharded input
    x_local = torch.zeros((B_local, D), dtype=torch.bfloat16, device=device)
    x_local[...] = x[start:end, :]
    out_local = model.forward(x_local)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.bfloat16) # global unsharded gradients
    grad_out_local = torch.zeros((B_local, D), dtype=torch.bfloat16, device=device)
    grad_out_local[...] = grad_out[start:end, :]
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
    mp.start_processes(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True, start_method="fork")
