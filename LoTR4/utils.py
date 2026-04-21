import torch

def reshape_heads(weight_matrix: torch.Tensor, num_heads: int = 12) -> torch.Tensor:
    '''
    将注意力头的 query 和 value 投影矩阵的注意力头数量拿出来单独作为一个维度,
    重塑为 (input_dim, num_heads, head_dim) .
    RoBERTa 的实现中将 query_layer/value_layer 重塑为形状 (batch_size, sequence_length, num_heads, head_size) ,
    因此此处将 out_dim 解释为 num_heads * head_dim 而非 head_dim * num_heads .

    Args:
        weight_matrix (Tensor): 权重矩阵, LoTR 中是 query 和 value 投影矩阵, 形状为 (input_dim, output_dim) , 在 RoBERTa 中是 (768, 768) .
        num_heads (int): 注意力头的数量, 需要被输出维度整除, 在 RoBERTa 中是 12 .

    Returns:
        Tensor: 重塑后的权重矩阵, 形状为 (input_dim, num_heads, head_dim) .
    '''
    input_dim, output_dim = weight_matrix.shape
    head_dim = output_dim // num_heads
    assert num_heads * head_dim == output_dim, f'输出维度必须整除注意力头的数量: {num_heads} | {weight_matrix.shape}'
    return weight_matrix.view(input_dim, num_heads, head_dim).contiguous()
