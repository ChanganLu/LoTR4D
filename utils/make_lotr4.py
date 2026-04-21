from transformers import RobertaForSequenceClassification as roberta

from typing import Tuple, Union, Literal
import re

from LoTR4.lotr4 import LoTR3, LoTR3Linear
from LoTR4.init import init_lotr4
from utils.set_attribute import attrsetter

RE_FILTER = re.compile(
    r'roberta.encoder.layer.\d+.attention.self.(value|query)'
)

def make_lotr4(model: roberta, rank: Union[int, Tuple[int, int, int]] = (64, 4, 16), core_init: Literal['neutral', 'svd', 'trivial'] = 'trivial', factor_init: Literal['svd', 'trivial', 'normal'] = 'svd'):
    '''
    768 = 12 * 64
    '''
    modules = {k: m for k,m in model.named_modules() if RE_FILTER.match(k)}
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    lotr3 = LoTR3(hidden_size, num_heads, hidden_size // num_heads, rank=rank)
    layers = []
    for path, module in modules.items():
        layer = LoTR3Linear.from_linear(module, num_heads, lotr3)
        layers.append(layer)
        attrsetter(model, path, layer)

    init_lotr4(layers, num_heads, core_init, factor_init)
    return model
