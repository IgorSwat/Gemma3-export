"""
This module contains core neural network building blocks for the Gemma model family,
including quantized linear and embedding layers, normalization, multi-layer perceptron,
attention mechanisms, decoder layers, and the main model container.

Each class implements a specific component used in transformer-based architectures,
with support for quantization and advanced normalization and attention techniques.

Module interactions:
- Linear and Embedding are used by higher-level modules for projections and embeddings.
- RMSNorm is used for normalization in attention, MLP, and decoder layers.
- MLP and GemmaAttention are core blocks in Gemma2DecoderLayer.
- Gemma2DecoderLayer is stacked in GemmaModel to form the full transformer.
"""

import torch
import torch.nn.functional as F

from .. import config as gemma_config
from .embedding_utils import apply_rotary_emb
from torch import nn
from typing import List, Tuple, Mapping


class Linear(nn.Module):
    """
    A linear (fully-connected) layer supporting optional quantization.
    If quantized, weights are stored as int8 and scaled by a learned parameter.
    Used for projections in attention and MLP blocks.

    Interactions:
    - Used by MLP for gate, up, and down projections.
    - Used by GemmaAttention for QKV and output projections.
    """

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output
    

class Embedding(nn.Module):
    """
    An embedding layer supporting optional quantization.
    Maps input indices to learned embedding vectors, with quantized weights if enabled.
    Used for token and positional embeddings.

    Interactions:
    - Typically used outside this file for input token and positional embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output
    

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    Normalizes inputs based on their RMS value, with optional unit offset.
    Used for stabilizing activations in transformer layers.

    Interactions:
    - Used by GemmaAttention for query/key normalization.
    - Used by Gemma2DecoderLayer for input, post-attention, and feedforward normalization.
    - Used by GemmaModel for final output normalization.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # call__to_dim_order happening here a lot?

        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)
    

class MLP(nn.Module):
    """
    Multi-Layer Perceptron block for transformer architectures.
    Consists of gate, up, and down projections, with GELU activation and optional quantization.
    Used for feed-forward transformations after attention.

    Interactions:
    - Uses Linear for all projections.
    - Used by Gemma2DecoderLayer as the feed-forward block.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(embedding_dim, hidden_dim, quant)
        self.up_proj = Linear(embedding_dim, hidden_dim, quant)
        self.down_proj = Linear(hidden_dim, embedding_dim, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs
    

class GemmaAttention(nn.Module):
    """
    Implements multi-head attention for Gemma models.
    Supports rotary positional embeddings, quantized projections, key-value caching,
    local sliding window attention, and optional logit softcapping.
    Used for self-attention in decoder layers.

    Interactions:
    - Uses Linear for QKV and output projections.
    - Uses RMSNorm for query/key normalization.
    - Used by Gemma2DecoderLayer for self-attention.
    """

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.embedding_dim = config.embedding_dim
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.embedding_dim,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=config.quant)
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.embedding_dim, quant=config.quant
        )
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1])
        xk = apply_rotary_emb(xk, freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1])

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache

        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            # Fix: replacing repeat_interleave makes the model almost 2x as fast for single token inference
            key = key.unsqueeze(3).expand(-1, -1, -1, self.num_queries_per_kv, -1).reshape(batch_size, -1, self.num_heads, self.head_dim)
            value = value.unsqueeze(3).expand(-1, -1, -1, self.num_queries_per_kv, -1).reshape(batch_size, -1, self.num_heads, self.head_dim)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class Gemma2DecoderLayer(nn.Module):
    """
    A single decoder layer for Gemma-2 and Gemma-3 architectures.
    Composed of layer normalization, self-attention, and MLP blocks,
    with optional pre/post feed-forward normalization.
    Used as a building block for stacking in the main model.

    Interactions:
    - Uses GemmaAttention for self-attention.
    - Uses MLP for feed-forward transformation.
    - Uses RMSNorm for normalization at various stages.
    - Stacked by GemmaModel to form the full model.
    """

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = MLP(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.embedding_dim,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.embedding_dim,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.embedding_dim, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.embedding_dim, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        # return hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    

class GemmaModel(nn.Module):
    """
    The main container for the Gemma model.
    Stacks multiple decoder layers, applies final normalization,
    and manages attention types and quantization settings.
    Used for end-to-end forward passes in Gemma-based transformer models.

    Interactions:
    - Stacks multiple Gemma2DecoderLayer instances.
    - Uses RMSNorm for final output normalization.
    - Manages attention types and passes configuration to each layer.
    """

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture in (
                gemma_config.Architecture.GEMMA_2,
                gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unsupported architecture: {config.architecture}')
        self.norm = RMSNorm(config.embedding_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, Tuple[torch.Tensor, torch.Tensor]],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states