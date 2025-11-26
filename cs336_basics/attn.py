import math

import torch
from jaxtyping import Float, Bool, Int
from torch import Tensor
from cs336_basics import run_softmax
from cs336_basics.other import run_rope


def run_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... dec_dim d_k"]): Query tensor
        K (Float[Tensor, " ... enc_dim d_k"]): Key tensor
        V (Float[Tensor, " ... enc_dim d_v"]): Values tensor
        mask (Bool[Tensor, " ... dec_dim enc_dim"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... dec d_v"]: Output of SDPA
    """

    d_k = Q.shape[-1]
    inner = (Q @ K.transpose(-1, -2)) / (d_k ** 0.5)
    scores = inner.masked_fill(~mask, float("-inf"))  # B x queries x keys

    ret = run_softmax(scores, dim=-1)
    ret = ret @ V

    return ret


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    q = in_features @ q_proj_weight.transpose(-1, -2)  # ... seq_len d_k
    k = in_features @ k_proj_weight.transpose(-1, -2)  # ... seq_len d_k
    v = in_features @ v_proj_weight.transpose(-1, -2)  # ... seq_len d_k

    q = q.reshape(*q.shape[:-1], num_heads, d_model // num_heads)
    k = k.reshape(*k.shape[:-1], num_heads, d_model // num_heads)
    v = v.reshape(*v.shape[:-1], num_heads, d_model // num_heads)

    q = q.permute(*range(q.ndim - 3), -2, -3, -1)  # ... heads seq_len emb
    k = k.permute(*range(k.ndim - 3), -2, -3, -1)  # ... heads seq_len emb
    v = v.permute(*range(v.ndim - 3), -2, -3, -1)  # ... heads seq_len emb

    affinity = q @ k.transpose(-1, -2)  # ...  heads seq_len seq_len
    affinity = affinity / math.sqrt(d_model // num_heads)

    seq_len = in_features.shape[-2]
    triangular_mask = torch.tril(torch.ones((seq_len, seq_len)))
    affinity = affinity.masked_fill(triangular_mask == 0, float("-inf"))
    scores = run_softmax(affinity, dim=-1)

    new_v = scores @ v  # ... heads seq_len emb

    new_v = new_v.permute(*range(new_v.ndim - 3), -2, -3, -1)  # ... seq_len heads emb
    new_v = new_v.reshape(*new_v.shape[:-2], d_model)  # ... seq_len d_model

    return new_v @ o_proj_weight.transpose(-1, -2)  # ... seq_len d_model


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    q = in_features @ q_proj_weight.transpose(-1, -2)  # ... seq_len d_k
    k = in_features @ k_proj_weight.transpose(-1, -2)  # ... seq_len d_k
    v = in_features @ v_proj_weight.transpose(-1, -2)  # ... seq_len d_k

    q = q.reshape(*q.shape[:-1], num_heads, d_model // num_heads)
    k = k.reshape(*k.shape[:-1], num_heads, d_model // num_heads)
    v = v.reshape(*v.shape[:-1], num_heads, d_model // num_heads)

    q = q.permute(*range(q.ndim - 3), -2, -3, -1)  # ... heads seq_len emb
    k = k.permute(*range(k.ndim - 3), -2, -3, -1)  # ... heads seq_len emb
    v = v.permute(*range(v.ndim - 3), -2, -3, -1)  # ... heads seq_len emb

    q = run_rope(d_model // num_heads, theta, max_seq_len, q, token_positions)
    k = run_rope(d_model // num_heads, theta, max_seq_len, k, token_positions)

    affinity = q @ k.transpose(-1, -2)  # ...  heads seq_len seq_len
    affinity = affinity / math.sqrt(d_model // num_heads)

    device = in_features.device

    seq_len = in_features.shape[-2]
    triangular_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    affinity = affinity.masked_fill(triangular_mask == 0, float("-inf"))
    scores = run_softmax(affinity, dim=-1)

    new_v = scores @ v  # ... heads seq_len emb

    new_v = new_v.permute(*range(new_v.ndim - 3), -2, -3, -1)  # ... seq_len heads emb
    new_v = new_v.reshape(*new_v.shape[:-2], d_model)  # ... seq_len d_model

    return new_v @ o_proj_weight.transpose(-1, -2)  # ... seq_len d_model
