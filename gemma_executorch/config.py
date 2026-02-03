from .types import Architecture, AttentionType
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import dataclasses
import torch


# -------------------------
# General config definition
# -------------------------

@dataclasses.dataclass
class GemmaConfig:
    # The architecture of the model.
    architecture: Architecture = Architecture.GEMMA_1
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    # = max_seq_len
    max_position_embeddings: int = 8192
    # The limited sequence length
    max_seq_length: int = 2048
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    embedding_dim: int = 3072
    # The dimension of the MLP representations.
    hidden_dim: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The number of kv groups
    n_kv_groups: int = 1,
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'bfloat16'
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The path to the model tokenizer.
    tokenizer: Union[str, Path] = (
    'tokenizer/tokenizer.model'
    )
    # The types of attention used in the layers of the model.
    attn_types: Optional[Sequence[AttentionType]] = None
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: Optional[float] = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: Optional[float] = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: Optional[int] = None
    # Whether to use pre mlp normalization.
    use_pre_ffw_norm: bool = False
    # Whether to use post mlp normalization.
    use_post_ffw_norm: bool = False
    # The wave length of the rotary embedding.
    rope_wave_length: dict[AttentionType, int] | None = None
    # Whether to use QK normalization in the attention blocks.
    use_qk_norm: bool = False
    # Vision model config.
    # TODO: add an appropriate type
    vision_config: Any = None
    # The factor by which the rope wave length is divided for global layers.
    rope_scaling_factor: int| None = None

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""

        _STR_DTYPE_TO_TORCH_DTYPE = dict({
            'float16': torch.float16,
            'float': torch.float32,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
        })

        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


# -----------------
# Gemma 270M config
# -----------------

def get_config_for_270m(dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      vocab_size=262_144,
      max_position_embeddings=32_768, # Effectively a context length
      max_seq_length=2048,
      num_hidden_layers=18,
      num_attention_heads=4,
      num_key_value_heads=1,
      embedding_dim=640,
      hidden_dim=2048,
      head_dim=256,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      sliding_window_size=512,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      query_pre_attn_scalar=256,
      use_qk_norm=True,
      vision_config=None,
  )


# ---------------
# Gemma 1b config
# ---------------

def get_config_for_1b(dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=26,
      num_attention_heads=4,
      num_key_value_heads=1,
      embedding_dim=1152,
      hidden_dim=6912,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=512,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      max_position_embeddings=32_768,
      max_seq_length=1024,
      tokenizer='tokenizer/gemma3_cleaned_262144_v2.spiece.model',
      use_qk_norm=True,
      vision_config=None,
  )


# ---------------
# Gemma 4b config
# ---------------

def get_config_for_4b(dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=34,
      num_attention_heads=8,
      num_key_value_heads=4,
      embedding_dim=2560,
      hidden_dim=10240,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=1024,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      tokenizer='tokenizer/gemma3_cleaned_262144_v2.spiece.model',
      use_qk_norm=True,
      # TODO: add vision config
      vision_config=None,
      rope_scaling_factor=8,
  )


# ----------------
# Config switching
# ----------------

def get_model_config(variant: str, dtype: str = 'bfloat16') -> GemmaConfig:
  """Gets the GemmaConfig for the diresired variant and dtype."""
  # Gemma1 variants
  # ...
  # Gemma2 variants
  # ..
  # Gemma3 variants
  if variant == '270m':
    return get_config_for_270m(dtype)
  elif variant == '1b':
    return get_config_for_1b(dtype)
  elif variant == '4b':
    return get_config_for_4b(dtype)
  # Invalid variants
  else:
    raise ValueError(
        f'Invalid variant {variant}. Supported variants are "270m", "1b", "4b"'
    )
