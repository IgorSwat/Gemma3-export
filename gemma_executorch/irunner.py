from .config import GemmaConfig
from .igemma import IGemma
from .itokenizer import ITokenizer
from abc import ABC, abstractmethod
from executorch.runtime import Runtime
from pathlib import Path
from typing import Any, Optional, Union
import torch

# An interface for Gemma model runner
class IRunner(ABC):

  def __init__(self, config: GemmaConfig, model: Union[IGemma, Path], tokenizer: ITokenizer, 
               use_pte: bool = False):
    if not use_pte:
      assert isinstance(model, IGemma)
      self.model = model
      self.et_runtime = None
    else:
      # In case of a .pte model, we want to load a model from a filepath
      assert isinstance(model, Path)
      self.et_runtime = Runtime.get()
      self.model = self.et_runtime.load_program(model).load_method("forward")

    self.tokenizer = tokenizer

    self.config = config
    self.use_pte = use_pte

  @abstractmethod
  def generate(
    self,
    prompt: str,
    output_len: int = 100,
    temperature: float | None = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    device: Any = torch.device("cpu")
  ):
    pass

  def run_model(self, input_tokens: torch.LongTensor, 
                      positions: torch.LongTensor) -> torch.Tensor:
    if not self.use_pte:
      return self.model(input_tokens, positions)
    else:
      return self.model.execute((input_tokens, positions))[0]
    
  def sample(
    self, 
    logits: torch.Tensor,
    temperatures: torch.Tensor | None,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    embedding_bias: Optional[torch.Tensor] = None,
  ):
    if temperatures is None:
      return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

    # Apply temperature scaling.
    logits.div_(temperatures.unsqueeze(dim=1))

    # Calculate probabilities with softmax.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # Apply top-p, top-k.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
    probs_sort = torch.where(top_ps_mask, 0, probs_sort)

    top_ks_mask = torch.arange(probs_idx.shape[-1],
                                device=probs_idx.device)
    top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
    top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
    probs_sort = torch.where(top_ks_mask, 0, probs_sort)

    # Re-normalization.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs = torch.gather(probs_sort,
                          dim=-1,
                          index=torch.argsort(probs_idx, dim=-1))

    next_token_ids = torch.multinomial(probs,
                                        num_samples=1,
                                        replacement=True).squeeze(dim=-1)

    return next_token_ids, logits