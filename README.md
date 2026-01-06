# ml

## NumPy

Numpy basics can be found under [there](tutorial/numpy)

## Torch

```bash
uv run python -m tutorial.torch.sharding.mlp_dp
```

#### Tutorial Torch

1. [Torch distributed API](https://docs.pytorch.org/docs/stable/distributed.html)
2. [Sharding strategies](tutorial/torch/sharding)

#### Notes Torch

- don't use the old primitives, instead use in-place ones like `dist.all_gather_into_tensor` and `dist.all_reduce_tensor` that aggregate along the primary dimension.

## JAX

```bash
uv run python -m tutorial.jax.sharding.mlp_dp
```

#### Tutorial JAX


1. [JIT compilation](tutorial/jax/jit.py)
2. [Sharding](tutorial/jax/sharding.py)
3. [Sharding strategies](tutorial/jax/sharding)
