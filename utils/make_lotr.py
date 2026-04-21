from transformers import RobertaForSequenceClassification as roberta

import re

from lotr.lotr import LoTR, LoTRLinear
from lotr.init import make_lotr_init
from utils.set_attribute import attrsetter

RE_FILTER = re.compile(
    r'roberta.encoder.layer.\d+.attention.self.(value|query)'
)

def make_lotr(model: roberta, rank: int = 8, use_svd_init=True):
    """
    将模型转换为 LoTR 结构，并应用初始化。
    """
    modules = {k: m for k,m in model.named_modules() if RE_FILTER.match(k)}
    lotr = LoTR(model.config.hidden_size, model.config.hidden_size, rank)
    layers = []
    for path, module in modules.items():
        layer = LoTRLinear.from_linear(module, lotr)
        # print(f'Trainable Checking 01: {layer.lotr.mid.requires_grad}')
        layers.append(layer)
        attrsetter(model, path, layer)

    # 2. 应用联合初始化 (SVD)
    # 这一步会计算所有收集到的层的 Tucker 分解，实现参数共享
    if use_svd_init:
        # 左=SVD, 中=Identity(neutral), 右=SVD
        # 这种组合通常比较强
        # initializer = make_lotr_init('svd', 'neutral', 'svd')
        initializer = make_lotr_init('svd', 'trivial', 'svd')
    else:
        initializer = make_lotr_init('normal', 'trivial', 'normal')
        
    initializer(layers)
    # for layer in layers:
    #     print(f'Trainable Checking 02: {layer.lotr.mid.requires_grad}')
    return model
