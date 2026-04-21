import torch
from torch import nn

from LoTR4.lotr4 import LoTR3, LoTR3Linear
from LoTR4.utils import reshape_heads

from typing import Tuple, Union, Literal, List, Callable, Optional
from functools import partial
from dataclasses import dataclass

try:
    from numpy.lib.array_utils import normalize_axis_index
except ImportError:
    from numpy.core.multiarray import normalize_axis_index


def assign_(ten: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ten.data = val.type(val.dtype)
        return ten

@dataclass
class Tucker3:
    rank: tuple[int, int, int]                                  # (R1, R2, R3)
    core: torch.Tensor                                          # (R1, R2 * R3, L)
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]    # (D1, R1), (R2, D2), (R3, D3)
    shape: Tuple[int, int, int, int]                            # (D1, D2, D3, D4)

def tucker3(tensor: torch.Tensor, rank: Tuple[int, int, int]) -> Tucker3:
    """A native implementation of Tucker2 algorithm for building corresponding tensor decomposition.
    """
    assert tensor.ndim == 4, 'Only 4-tensors are accepted.'
    D1, D2, D3, D4 = tensor.shape
    R1, R2, R3 = rank

    X1 = tensor.reshape(D1, -1)                             # (D1, D2 * D3 * D4)
    X2 = tensor.permute(1, 0, 2, 3).reshape(D2, -1)         # (D2, D1 * D3 * D4)
    X3 = tensor.permute(2, 0, 1, 3).reshape(D3, -1)         # (D3, D1 * D2 * D4)
    
    factor1 = torch.svd_lowrank(X1, R1)[0]                  # (D1, R1)
    factor2 = torch.svd_lowrank(X2, R2)[0].T                # (R2, D2)
    factor3 = torch.svd_lowrank(X3, R3)[0].T                # (R3, D3)

    temp1 = (factor1.T @ X1).reshape(R1, D2, D3, D4)        # (R1, D2, D3, D4)
    temp2 = temp1.permute(1, 0, 2, 3).reshape(D2, -1)       # (D2, R1 * D3 * D4)
    temp2 = (factor2 @ temp2).reshape(R2, R1, D3, D4)       # (R2, R1, D3, D4)
    temp3 = temp2.permute(2, 1, 0, 3).reshape(D3, -1)       # (D3, R1 * R2 * D4)
    temp3 = (factor3 @ temp3).reshape(R3, R1, R2, D4)       # (R3, R1, R2, D4)
    core = temp3.permute(1, 2, 0, 3).reshape(R1, -1, D4)    # (R1, R2 * R3, D4)

    return Tucker3((R1, R2, R3), core, (factor1, factor2, factor3), (D1, D2, D3, D4))


def make_factor_init(factor_init: Literal['svd', 'trivial', 'normal'], decomp: Tucker3) -> Callable[[torch.Tensor, int], torch.Tensor]:
    if factor_init == 'normal':
        return lambda factor, idx: nn.init.normal_(factor)
    elif factor_init == 'trivial':
        return lambda factor, idx: nn.init.zeros_(factor)
    elif factor_init == 'svd':
        return lambda factor, idx: assign_(factor, decomp.factors[idx])
    else:
        raise ValueError(f'Unknown factor init: {factor_init}')

def make_core_init(core_init: Literal['neutral', 'svd', 'trivial'], decomp: Tucker3) -> Callable[[torch.Tensor, int], torch.Tensor]:
    if core_init == 'neutral':
        return lambda core, idx: nn.init.eye_(core)
    elif core_init == 'trivial':
        return lambda core, idx: nn.init.zeros_(core)
    elif core_init == 'svd':
        return lambda core, idx: assign_(core, decomp.core[..., idx])
    else:
        raise ValueError(f'Unknown core init: {core_init}')

def make_lotr4_init(num_heads: int, core_init: Literal['neutral', 'svd', 'trivial'], factor_init: Literal['svd', 'trivial', 'normal']) -> Callable[[List[LoTR3Linear]], None]:
    return partial(init_lotr4, num_heads=num_heads, core_init=core_init, factor_init=factor_init)

def init_lotr4(layers: List[LoTR3Linear], num_heads: int, core_init: Literal['neutral', 'svd', 'trivial'], factor_init: Literal['svd', 'trivial', 'normal']) -> None:
    layers = [*layers]
    if not layers:
        return
    
    rank: Optional[Tuple[int, int, int]] = None
    for ix, layer in enumerate(layers):
        if not isinstance(layer, LoTR3Linear):
            raise ValueError(f'Each layer in a sequence must be of LoTR3Linear type or derived from it but actual type of layer #{ix} is {type(layer)}.')

        if rank is None:
            rank = layer.rank
        elif any(rank[i] != layer.rank[i] for i in range(len(rank))):
            raise ValueError(f'Each layer in a sequence must have the same Tucker rank but layer #{ix} has rank {layer.rank} instead of {rank}.')

    decomp: Optional[Tucker3] = None
    if 'svd' in (core_init, factor_init):
        ten = torch.stack([reshape_heads(el.linear.weight, num_heads) for el in layers], dim=-1)
        decomp = tucker3(ten, rank)

    layer: LoTR3Linear = layers[0]
    dtype = layer.linear.weight.dtype
    device = layer.linear.weight.device
    lotr3 = LoTR3(*layer.dim, layer.rank, device, dtype)

    factor_fn = make_factor_init(factor_init, decomp)
    factor_fn(lotr3.factor1, 0)
    factor_fn(lotr3.factor2, 1)
    factor_fn(lotr3.factor3, 2)

    core_fn = make_core_init(core_init, decomp)
    for idx, layer in enumerate(layers):
        layer.lotr3 = LoTR3.from_lotr3(lotr3)
        core_fn(layer.lotr3.core, idx)


