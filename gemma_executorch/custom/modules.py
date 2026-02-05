from .utils import compute_rope_params, apply_rope
from ..config import GemmaConfig
from ..types import AttentionType
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dim, config.hidden_dim, dtype=config.get_dtype(), bias=False)
        self.fc2 = nn.Linear(config.embedding_dim, config.hidden_dim, dtype=config.get_dtype(), bias=False)
        self.fc3 = nn.Linear(config.hidden_dim, config.embedding_dim, dtype=config.get_dtype(), bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)
    

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
         
        if self.shift is not None:
            out = out + self.shift.float()
         
        return out.to(input_dtype)
    

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        attn_type,
        d_in, 
        num_heads, 
        num_kv_groups,
        max_seq_len,
        sliding_window_size,
        head_dim=None, 
        qk_norm=False,
        query_pre_attn_scalar=None, 
        dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.attn_type = attn_type
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.sliding_window_size = sliding_window_size

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5

        # Pre-allocate KV cache buffers
        # NOTE: does not support more than 1 batch
        kv_cache_shape = (1, max_seq_len, num_kv_groups, head_dim)
        self.register_buffer("k_cache", torch.zeros(kv_cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(kv_cache_shape, dtype=dtype), persistent=False)

    def forward(self, x, input_positions, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE to queries and the NEW keys only
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Update cache for new tokens
        self.k_cache[:b, input_positions] = keys.transpose(1, 2)
        self.v_cache[:b, input_positions] = values.transpose(1, 2)

        # Use full history from cache for attention
        # For sliding window, you'd slice the last N positions here
        last_pos = input_positions[-1].item()
        keys = self.k_cache[:b, :last_pos+1].transpose(1, 2)
        values = self.v_cache[:b, :last_pos+1].transpose(1, 2)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Scale queries
        queries = queries * self.scaling

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).contiguous().reshape(b, num_tokens, self.d_out)

        return self.out_proj(context)


class TransformerBlock(nn.Module):

    def __init__(self, config: GemmaConfig, attn_type: AttentionType):
        super().__init__()
        self.attn_type = attn_type 

        self.att = GroupedQueryAttention(
            attn_type=attn_type,
            d_in=config.embedding_dim,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.n_kv_groups,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_length,
            sliding_window_size=config.sliding_window_size,
            qk_norm=config.use_qk_norm,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            dtype=config.get_dtype(),
        )
        self.ff = FeedForward(config)
        self.input_layernorm = RMSNorm(config.embedding_dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(config.embedding_dim, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(config.embedding_dim, eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(config.embedding_dim, eps=1e-6)

    def forward(
        self,
        x,
        input_positions,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == AttentionType.LOCAL_SLIDING:
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global
        
        x_attn = self.att(x, input_positions, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        
        return x