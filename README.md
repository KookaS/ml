# Machine Learning

Here are the modern implementations of LLM architecture, sharding strategies and kernel optimizations.

## Core

- Softmax in [PyTorch with autograd](tutorial/torch/softmax.py) and in [NumPy](tutorial/numpy/softmax.py)
- Linear projection in [PyTorch with autograd](tutorial/torch/linear.py)

## Positional Encoder

- Positional Encoder Sinusoidal in [NumPy](tutorial/numpy/positional_encoding_sinusoidal.py)
- RoPE in [NumPy](tutorial/numpy/positional_encoding_rope.py)
- RoPE GPT-NeoX in [NumPy](tutorial/numpy/positional_encoding_rope_neox.py)

## Transformer

- Multihead Attention in [PyTorch with autograd](tutorial/torch/attention.py)
- Multi Layer Perceptron in [PyTorch with autograd](tutorial/torch/mlp.py)
- Norm RMS in [PyTorch with autograd](tutorial/torch/normalization_rms.py) and in [NumPy](tutorial/numpy/normalization_rms.py)
- Transformer in [PyTorch with autograd](tutorial/torch/transformer.py)

## Sharding strategies

- MLP Data Parallelism(DP) in [PyTorch](tutorial/torch/sharding/mlp_dp.py), in [JAX](tutorial/jax/sharding/mlp_dp.py)
- MLP Tensor Parallelism(TP) in [PyTorch](tutorial/torch/sharding/mlp_tp.py), in [JAX](tutorial/jax/sharding/mlp_tp.py)
- MLP Fully Sharded Data Parallelism(FSDP) in [PyTorch](tutorial/torch/sharding/mlp_fsdp.py), in [JAX](tutorial/jax/sharding/mlp_fsdp.py)
- MLP Pipelining in [PyTorch](tutorial/torch/sharding/mlp_pp.py)

## NumPy Tutorial

- Masking in [NumPy](tutorial/numpy/masking.py)

## JAX Tutorial

- [JIT](tutorial/jax/jit.py)
- [Condition](tutorial/jax/condition.py)
- [Shading and Mesh](tutorial/jax/sharding.py)

## PyTorch Notes

- [Torch distributed API](https://docs.pytorch.org/docs/stable/distributed.html).
- don't use the old primitives, instead use in-place ones like `dist.all_gather_into_tensor` and `dist.all_reduce_tensor` that aggregate along the primary dimension.
- custom classes for training requires `torch.autograd.Function`, `@staticmethod` and [`ctx.save_for_backward`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)