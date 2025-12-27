import torch


def LossL2(desired: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (expected - desired)**2