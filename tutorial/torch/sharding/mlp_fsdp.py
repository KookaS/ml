import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from tutorial.torch.mlp import Mlp

class MlpFsdp(Mlp):

    def __init__(self, rank, world_size, d_model, d_ff):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.d_model = d_model
        self.d_ff = d_ff

    def load_checkpoint(self, params):
        """
        X[Bx, D] @ Win[Dx, F] @ Wout[F, Dx]
        """
        local_d_model = self.d_model // dist.get_world_size()
        start = local_d_model * self.rank
        end = start + local_d_model
        self.w_in = params['layer_in/weights'][start:end, :].detach().clone().to(f"cpu:{self.rank}")
        self.w_out = params['layer_out/weights'][:, start:end].detach().clone().to(f"cpu:{self.rank}")

    def forward(self, x):
        """
        Win[D, F] = AllGather_Dx(Win[Dx, F])
        Z[Bx, F] = X[Bx, D] @ Win[D, F]
        A[Bx, F] = Activation (Z)
        Wout[F, D] = AllGather_Dx(Wout[F, Dx])
        Out[Bx, D] = A[Bx, F] @ Wout[F, D]
        """
        F, D = self.d_ff, self.d_model

        # ALL-GATHER
        w_in_global = torch.zeros((D, F), dtype=self.w_in.dtype, device=self.w_in.device)
        dist.all_gather_into_tensor(w_in_global, self.w_in)

        self.activations.append(x)
        z = einsum('bd,df->bf', x, w_in_global)
        del w_in_global

        # ALL-GATHER
        w_out_global = torch.zeros((D, F),  dtype=self.w_out.dtype, device=self.w_out.device)
        w_out_local = self.w_out.t().contiguous()
        dist.all_gather_into_tensor(w_out_global, w_out_local)
        del w_out_local
        w_out_global = w_out_global.t() # [F, D]

        self.activations.append(z)
        a = torch.nn.functional.relu(z)
        out = einsum('bf,fd->bd', a, w_out_global)
        del w_out_global

        return out

    def backward(self, out_grad):
        """
        dWout[F, D]{Ux} = A.T[F, Bx] @x dOut[Bx, D]
        dWout[F, Dx] = ReduceScatter_Dx(dWout[F, D]{Ux})
        
        Wout[F, D] = AllGather_Dx(Wout[F, Dx])
        dA[Bx, F] = dOut[Bx, D] @ Wout.T[D, F]
        dZ[Bx, F] = dA[Bx, F] * Act'(Z)[Bx, F]
        dWin[D, F]{Ux} = X.T[D, Bx] @x dZ[Bx, F]
        dWin[Dx, F] = ReduceScatter_Dx(dWin[D, F]{Ux})

        Win.T[D, F] = AllGather_Dx(Win.T[Dx, F])
        dX[Bx, D] = dZ[Bx, F] @ Win.T[F, D]
        """
        F, Dx = self.w_out.shape
        D = self.d_model

        z = self.activations.pop()
        a = torch.nn.functional.relu(z)
        w_out_grad = einsum('bf,bd->fd', a, out_grad)

        # REDUCE-SCATTER dWout
        # torch always reduces along the primary dimension        
        w_out_grad_local = torch.zeros((Dx, F), device=self.w_out.device, dtype=self.w_out.dtype) # [Dx, F]
        # contiguous is needed for NCCL that needs sequential data X memory
        w_out_grad = w_out_grad.transpose(0, 1).contiguous() # [D, F]{Ux}
        dist.reduce_scatter_tensor(w_out_grad_local, w_out_grad, op=dist.ReduceOp.AVG)
        del w_out_grad
        w_out_grad_local = w_out_grad_local.transpose(0, 1) # [F, Dx]

        # ALL-GATHER Wout
        w_out = torch.zeros((D, F), device=self.w_out.device, dtype=self.w_out.dtype) # [D, F]
        w_out_local = self.w_out.t().contiguous() # [Dx, F]
        dist.all_gather_into_tensor(w_out, w_out_local)
        del w_out_local
        w_out = w_out.t() # [F, D]

        a_grad = einsum('bd,fd->bf', out_grad, w_out)
        z_grad = a_grad * (z > 0)
        x = self.activations.pop()
        w_in_grad = einsum('bd,bf->df', x, z_grad) # [D, F]{Ux}

        # REDUCE-SCATTER dWin
        w_in_grad_local = torch.zeros((Dx, F), device=self.w_in.device, dtype=self.w_in.dtype) # [Dx, F]
        w_in_grad = w_in_grad.contiguous()
        dist.reduce_scatter_tensor(w_in_grad_local, w_in_grad, op=dist.ReduceOp.AVG)
        del w_in_grad

        # AL-GATHER Win
        w_in = torch.zeros((D, F), dtype=self.w_in.dtype, device=self.w_in.device)
        w_in_local = self.w_in.contiguous()
        dist.all_gather_into_tensor(w_in, w_in_local)
        del w_in_local

        x_grad = einsum('bf,df->bd', z_grad, w_in)

        return {'layer_out/weights': w_out_grad_local, 'layer_in/weights': w_in_grad_local, 'input': x_grad}


B, D, F = 8, 64, 256

# --- The Runner ---
def worker_fn(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Model Init
    torch.manual_seed(42)
    params = {
        'layer_in/weights': torch.randn(D, F, dtype=torch.float32),
        'layer_out/weights': torch.randn(F, D, dtype=torch.float32),
    }
    model = MlpFsdp(rank, world_size, D, F)
    model.load_checkpoint(params)

    # Data Sharding
    B_local = B // world_size
    start = rank * B_local
    end = start + B_local

    x = torch.randn(B, D, dtype=torch.float32)
    x_local = x[start:end, :].detach().clone().to(f"cpu:{rank}")
    out_local = model.forward(x_local)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.float32)
    grad_out_local = grad_out[start:end, :].detach().clone().to(f"cpu:{rank}")
    grads = model.backward(grad_out_local)

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

    

    