import torch


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         rope_scaling_factor:int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs/rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # NOTE: in original code, complex tensors were used.
    # However, ExecuTorch does not support operations on complex tensors - therefore
    # we use 2 real tensors instead
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    # x shape: [batch_size, seq_len, num_heads, head_dim]
    # The original logic splits the head_dim into two halves (real and imaginary parts)
    x_real, x_imag = torch.chunk(x.transpose(1, 2).float(), 2, dim=-1)
    
    # Complex multiplication equivalent using real numbers:
    # (x_real + i*x_imag) * (cos + i*sin) = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    
    # Concatenate back and restore shape
    x_out = torch.cat([out_real, out_imag], dim=-1).type_as(x)
    return x_out.transpose(1, 2)