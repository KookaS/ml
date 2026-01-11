# Machine Learning

## Core

- Softmax in [PyTorch](tutorial/torch/softmax.py) and in [NumPy](tutorial/numpy/softmax.py)

## Positional Encoder
- Positional Encoder Sinusoidal in [NumPy](tutorial/numpy/positional_encoding_sinusoidal.py)
- RoPE in [NumPy](tutorial/numpy/positional_encoding_rope.py)
- RoPE GPT-NeoX in [NumPy](tutorial/numpy/positional_encoding_rope_neox.py)

## Sharding strategies

- MLP Data Parallelism(DP) in [PyTorch](tutorial/torch/sharding/mlp_dp.py), in [JAX](tutorial/jax/sharding/mlp_dp.py)
- MLP Tensor Parallelism(TP) in [PyTorch](tutorial/torch/sharding/mlp_tp.py), in [JAX](tutorial/jax/sharding/mlp_tp.py)
- MLP Fully Sharded Data Parallelism(FSDP) in [PyTorch](tutorial/torch/sharding/mlp_fsdp.py), in [JAX](tutorial/jax/sharding/mlp_fsdp.py)
- MLP Pipelining in [PyTorch](tutorial/torch/sharding/mlp_pp.py)

## NumPy Tutorial

- [Masking NumPy](tutorial/numpy/masking.py)
- [Normalization](tutorial/numpy/normalization.py)

## JAX Tutorial

- [JIT](tutorial/jax/jit.py)
- [Condition](tutorial/jax/condition.py)
- [Shading and Mesh](tutorial/jax/sharding.py)

## Notes Torch

- [Torch distributed API](https://docs.pytorch.org/docs/stable/distributed.html).
- don't use the old primitives, instead use in-place ones like `dist.all_gather_into_tensor` and `dist.all_reduce_tensor` that aggregate along the primary dimension.
- custom class for training requires `torch.autograd.Function`, `@staticmethod` and [`ctx.save_for_backward`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)