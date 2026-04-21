import torch
from torch import nn

import math
from typing import Tuple, List, Union, Optional, Self


class LoTR3(nn.Module):
    def __init__(self,
                 in_dim: int, num_heads: int, head_dim: int,
                 rank: Union[int, Tuple[int, int, int]],
                 device: Optional[torch.device] = None, dtype = torch.float,
    ):
        super().__init__()

        if isinstance(rank, int):
            rank = (rank, rank, rank)
        rank1, rank2, rank3 = rank

        self.dim = (in_dim, num_heads, head_dim)
        self.rank = rank

        self.core = nn.Parameter(torch.empty((rank1, rank2 * rank3), device=device, dtype=dtype), requires_grad=True)
        self.factor1 = nn.Parameter(torch.empty((in_dim, rank1), device=device, dtype=dtype), requires_grad=True)
        self.factor2 = nn.Parameter(torch.empty((rank2, num_heads), device=device, dtype=dtype), requires_grad=True)
        self.factor3 = nn.Parameter(torch.empty((rank3, head_dim), device=device, dtype=dtype), requires_grad=True)

        nn.init.zeros_(self.core)    
        nn.init.normal_(self.factor1)
        nn.init.normal_(self.factor2)
        nn.init.normal_(self.factor3)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(({self.dim}) -> ({self.rank}))'
    
    @property
    def device(self):
        return self.core.device
    
    @property
    def dtype(self):
        return self.core.dtype
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            inputs (Tensor): shape (..., dim1)
        
        Returns:
            Tensor: shape (..., dim2 * dim3)
        '''
        broadcast = inputs.shape[:-1]
        dim1, dim2, dim3 = self.dim
        rank1, rank2, rank3 = self.rank
        proj1 = torch.matmul(inputs, self.factor1)                  # (..., rank1)
        projr_flat = torch.matmul(proj1, self.core)                 # (..., rank2 * rank3)
        projr = projr_flat.reshape(*broadcast, rank2, rank3)        # (..., rank2, rank3)
        proj2 = torch.matmul(projr.transpose(-1, -2), self.factor2) # (..., rank3, dim2)
        proj3 = torch.matmul(proj2.transpose(-1, -2), self.factor3) # (..., dim2, dim3)
        return proj3.reshape(*broadcast, dim2 * dim3)               # (..., dim2 * dim3)

    @classmethod
    def from_lotr3(cls, other: 'LoTR3') -> Self:
        '''Construct a new container from existing one with projection matrix reused.'''
        self = cls(other.dim[0], other.dim[1], other.dim[2], other.rank, other.device, other.dtype)
        self.factor1 = other.factor1
        self.factor2 = other.factor2
        self.factor3 = other.factor3
        return self

class LoTR3Linear(nn.Module):
    def __init__(self,
                 in_dim: int, num_heads: int, head_dim: int, rank: Union[int, Tuple[int, int, int]],
                 device: Optional[torch.device] = None, dtype = torch.float,
                 bias: bool = True, scale: float = 1.0,
    ):
        super().__init__()

        if isinstance(rank, int):
            mean_rank = float(rank)
            rank = (rank, rank, rank)
        else:
            mean_rank = math.cbrt(math.prod([float(r) for r in rank]))

        self.dim = (in_dim, num_heads, head_dim)
        self.rank = rank

        self.scale = scale / mean_rank
        self.linear = nn.Linear(in_dim, num_heads * head_dim, bias=bias, device=device, dtype=dtype)
        self.linear.weight.requires_grad_(False)
        self.lotr3 = LoTR3(in_dim, num_heads, head_dim, rank, device, dtype)
    
    @property
    def device(self):
        return self.lotr3.device

    @property
    def dtype(self):
        return self.lotr3.dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # return self.linear(inputs) + self.lotr3(inputs)
        return self.linear(inputs) + self.scale * self.lotr3(inputs)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, num_heads: int, lotr3: LoTR3, scale: float = 1.0, device: Optional[torch.device] = None, dtype = torch.float) -> Self:
        bias = linear.bias is not None
        self = LoTR3Linear(linear.in_features, num_heads, linear.out_features // num_heads, lotr3.rank, device, dtype, bias, scale)
        self.linear = linear.requires_grad_(False)
        self.lotr3 = LoTR3.from_lotr3(lotr3).requires_grad_(True)
        return self


