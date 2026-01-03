# ml

## Run

```bash
uv run python -m tutorial.jax.sharding.mlp_dp
```

To view the Perfetto trace:
  1. Go to https://ui.perfetto.dev/
  2. Click "Open trace file"
  3. Select ./trace/plugins/profile/2025_12_27_18_46_31/perfetto_trace.json.gz

## Tutorial JAX


1. [JIT compilation](tutorial\jax\jit.py)
2. [Sharding](tutorial\jax\sharding.py)
3. [MLP class with JIT compilation](tutorial\jax\mlp_jit.py)

## Tutorial Torch

https://docs.pytorch.org/docs/stable/distributed.html

torch mesh + fsdp
torch manual all-to-all for MoE
