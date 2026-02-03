from .embedding_utils import precompute_freqs_cis
from .modules import GemmaModel, Embedding
from .. import config as gemma_config
from ..igemma import IGemma
from ..utils.checkpoint import load_checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union, override

# A casual gemma LLM implementation (no multimodality)
class GemmaForCausalLM(IGemma, nn.Module):

  def __init__(
        self,
        config: gemma_config.GemmaConfig
    ):
    nn.Module.__init__(self)
    IGemma.__init__(self, config)

    max_seq_len = config.max_position_embeddings
    head_dim = config.head_dim
    vocab_size = config.vocab_size

    self.embedder = Embedding(vocab_size, config.embedding_dim, config.quant)
    self.model = GemmaModel(config)

    for i in range(config.num_hidden_layers):
      size = (1, config.max_seq_length, config.num_key_value_heads, config.head_dim)
      # persistent=False keeps these out of saved checkpoints but keeps them in the module
      self.register_buffer(f"k_cache_{i}", torch.zeros(size, dtype=config.get_dtype()), persistent=False)
      self.register_buffer(f"v_cache_{i}", torch.zeros(size, dtype=config.get_dtype()), persistent=False)

    # Pre-compute rotary embedding table.
    if config.architecture == gemma_config.Architecture.GEMMA_3:
      if config.rope_wave_length is None:
        raise ValueError('rope_wave_length must be provided for Gemma3.')

      rope_lengths = config.rope_wave_length
      defaults = {
        gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
        gemma_config.AttentionType.GLOBAL: 1_000_000,
      }

      for attn_type, name in [
        (gemma_config.AttentionType.LOCAL_SLIDING, 'local_freqs_cis'),
        (gemma_config.AttentionType.GLOBAL, 'global_freqs_cis'),
      ]:
        theta = rope_lengths.get(
                    attn_type, defaults[attn_type]
                )
        self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

    else:
      self._register_freqs_cis('freqs_cis', head_dim, max_seq_len)

    # Precomputed global mask
    mask_size = (1, 1, self.config.max_seq_length, self.config.max_seq_length)
    mask_val = -2.3819763e38
    g_mask = torch.triu(torch.full(mask_size, mask_val, dtype=torch.float), diagonal=1)
    self.register_buffer("global_mask_buffer", g_mask, persistent=False)

    # Precomputed local mask
    if config.sliding_window_size:
      l_mask = g_mask + torch.tril(
        torch.full(mask_size, mask_val, dtype=torch.float),
        diagonal=-config.sliding_window_size
      )
      self.register_buffer("local_mask_buffer", l_mask, persistent=False)
    else:
        self.local_mask_buffer = None

    # Precomputed embedder normalization tensor
    self.register_buffer(
      "embed_normalizer", 
      torch.tensor(config.embedding_dim**0.5, dtype=config.get_dtype()), 
      persistent=False
    )

  def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000):
    cos, sin = precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta)
    self.register_buffer(f"{name}_cos", cos)
    self.register_buffer(f"{name}_sin", sin)

  # Initialize model weights
  def load(self, model_path: Union[Path, str]):
    load_checkpoint(self, model_path)
  
  # Model's main inference
  # Returns logit tensor
  @override
  @torch.no_grad()
  def forward(
    self,
    input_token_ids: torch.Tensor,  # A single prompt inference
    input_positions: torch.Tensor,  # In form of an increasing sequence [0, 1, 2, ..., n]
  ) -> torch.Tensor:
    # ---------------------------------------
    # Initialization - basic input properties
    # ---------------------------------------

    device = input_token_ids.device
    
    # --------------------------
    # Initialization - freqs_cis
    # --------------------------

    freqs_cis = {}

    if self.config.architecture == gemma_config.Architecture.GEMMA_3:
      freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
        self.local_freqs_cis_cos.index_select(0, input_positions),
        self.local_freqs_cis_sin.index_select(0, input_positions)
			)
      freqs_cis[gemma_config.AttentionType.GLOBAL] = (
				self.global_freqs_cis_cos.index_select(0, input_positions),
				self.global_freqs_cis_sin.index_select(0, input_positions)
			)
    else:
      freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
        self.freqs_cis_cos.index_select(0, input_positions),
        self.freqs_cis_sin.index_select(0, input_positions)
			)
      freqs_cis[gemma_config.AttentionType.GLOBAL] = (
        self.freqs_cis_cos.index_select(0, input_positions),
        self.freqs_cis_sin.index_select(0, input_positions)
			)
    
    # ---------------------------------
    # Initialization - kv_write_indices 
    # ---------------------------------

    # The original code just passes None and then does this, so I guess it's enough
    kv_write_indices = input_positions

    # --------------------------
    # Initialization - kv_caches
    # --------------------------
    
    kv_caches = []
    for i in range(self.config.num_hidden_layers):
      k_cache = getattr(self, f"k_cache_{i}")
      v_cache = getattr(self, f"v_cache_{i}")
      kv_caches.append((k_cache, v_cache))

    # ---------------------
    # Initialization - mask
    # ---------------------

    mask = self.global_mask_buffer.index_select(2, input_positions)

    # ---------------------------
    # Initialization - local_mask
    # ---------------------------

    local_mask = self.local_mask_buffer.index_select(
          2, input_positions
      ) if self.local_mask_buffer is not None else None
    
    # ----------------------cle
    # Inference - embeddings
    # ----------------------
    
    hidden_states = self.embedder(input_token_ids)
    # return hidden_states

    # Gemma normalizes the embedding by sqrt(hidden_size).
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    # normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
    hidden_states = hidden_states * self.embed_normalizer

    # return hidden_states

    # ----------------------------------
    # Inference - core (attention & MLP)
    # ----------------------------------

    hidden_states = self.model(
      hidden_states=hidden_states,
      freqs_cis=freqs_cis,
      kv_write_indices=kv_write_indices,
      kv_caches=kv_caches,
      mask=mask,
      local_mask=local_mask,
    )

    # ------------------------------------------------
    # Inference - output (reverse embeddings & logits)
    # ------------------------------------------------

    # NOTE: an equantization trick if config.quant is enabled
    embedder_weight = self.embedder.weight
    if self.config.quant:
      embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
      
    # Select the last element for each sequence.
    # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
    hidden_states = hidden_states[:, -1, :]
    
    logits = F.linear(hidden_states, embedder_weight)

    if self.config.final_logit_softcapping is not None:
      logits = logits / self.config.final_logit_softcapping
      logits = torch.tanh(logits)
      logits = logits * self.config.final_logit_softcapping

    return logits
