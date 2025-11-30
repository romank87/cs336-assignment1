import math
from typing import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, " d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    assert weights.shape == (d_model,)

    in_dtype = in_features.dtype
    x = in_features.to(torch.float32)

    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    x = x / rms

    return (x * weights).to(in_dtype)


def run_swiglu(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    w1x = in_features @ w1_weight.transpose(-1, -2)  # (... d_model) x  (d_ff d_model)^T =  ... d_ff
    w3x = in_features @ w3_weight.transpose(-1, -2)  # ... d_ff

    silu = w1x * torch.sigmoid(w1x)  # ... d_ff

    inside = silu * w3x  # ... d_off

    output = inside @ w2_weight.transpose(-1, -2)  # ... d_model

    return output


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """

    ids = torch.arange(1, d_k // 2 + 1, device=in_query_or_key.device)
    x = 1 / (theta ** ((2 * ids - 2) / d_k))

    token_positions = token_positions.unsqueeze(-1)
    freq = token_positions * x.view(1, -1)

    sin_t = torch.sin(freq)
    cos_t = torch.cos(freq)

    even = in_query_or_key[..., 0::2]
    odd = in_query_or_key[..., 1::2]

    rotated_even = cos_t * even - sin_t * odd
    rotated_odd = sin_t * even + cos_t * odd

    rotated = torch.zeros_like(in_query_or_key)
    rotated[..., 0::2] = rotated_even
    rotated[..., 1::2] = rotated_odd
    return rotated


def run_softmax_classic(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    denominator = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    return torch.exp(x) / denominator


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    denominator = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    return torch.exp(x) / denominator


def run_softmax_with_temperature(in_features: Float[Tensor, " ..."], dim: int, temperature: float) -> Float[
    Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.
        temperature (float): Temperature to use in softmax.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    x = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    if temperature < 1e-5:
        argmax_indices = torch.argmax(in_features, dim=dim, keepdim=True)

        result = torch.zeros_like(in_features)

        result.scatter_(dim, argmax_indices, 1.0)
        return result
    else:
        x = x / temperature
        exp_x = torch.exp(x)
        denominator = torch.sum(exp_x, dim=dim, keepdim=True)
        return exp_x / denominator


def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    cosine_term = 1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * cosine_term


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> float:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    summa = sum(((p.grad.data ** 2).sum() for p in parameters if p.grad is not None), start=0.0)
    g_norm = math.sqrt(summa)

    if g_norm > max_l2_norm:
        clip_coef = max_l2_norm / (g_norm + 1e-6)
        for p in parameters:

            if p.grad is None:
                continue

            p.grad.data.mul_(clip_coef)

    return g_norm
