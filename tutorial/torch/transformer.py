import torch

from tutorial.torch.attention import Attention
from tutorial.torch.mlp import Mlp
from tutorial.torch.normalization_rms import NormRms

class Transformer(torch.nn.Module):

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            head_dim: int,
            d_ff: int,
            masking: bool = True,
        ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_ff = d_ff
        self.masking = masking

         # [B, T, D], norm done on D dimension
        self.norm_att = NormRms(d_model)
        self.attention = Attention(d_model, n_heads, head_dim, masking)
        self.norm_mlp = NormRms(d_model)
        self.mlp = Mlp(d_model, d_ff)

    def forward(self, x):
        """
        A basic transformer architecture with pre-norm on both Attention and Mlp.  
        """
        
        x_norm_att = self.norm_att(x)
        x_residual_att = x + self.attention(x_norm_att) # Residual() does the same as the + with autograd

        x_normed_mlp = self.norm_mlp(x_residual_att)
        x_residual_mlp = x_residual_att + self.mlp(x_normed_mlp)

        return x_residual_mlp