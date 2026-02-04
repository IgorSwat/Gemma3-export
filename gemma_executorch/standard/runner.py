from .gemma_casual import GemmaForCausalLM
from .tokenizer import Tokenizer
from ..config import GemmaConfig
from ..irunner import IRunner
from ..utils.chat_template import apply_chat_template
from pathlib import Path
from typing import Any, Optional, Union
import torch


class Runner(IRunner):

  def __init__(self, config: GemmaConfig, model: Union[GemmaForCausalLM, Path], tokenizer: Tokenizer, 
               use_pte: bool = False, use_instruct: bool = True):
    super().__init__(config, model, tokenizer, use_pte)
    self.use_instruct = use_instruct

  def generate(
    self,
    prompt: str,
    output_len: int = 100,
    temperature: float | None = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    device: Any = torch.device("cpu")
  ):
    if self.use_instruct:
      prompt = apply_chat_template(prompt)

    prompt_tokens = self.tokenizer.encode(prompt)

    prompt_length = len(prompt_tokens)
    max_seq_len = prompt_length + output_len
    assert max_seq_len <= self.config.max_position_embeddings

    token_ids_tensor = torch.full((1, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.long)
    input_token_ids_tensor = torch.full((1, prompt_length),
                                        self.tokenizer.pad_id,
                                        dtype=torch.long)
    token_ids_tensor[:, :prompt_length] = torch.tensor(prompt_tokens)
    input_token_ids_tensor[:, :prompt_length] = torch.tensor(prompt_tokens[:prompt_length])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    input_positions_tensor = torch.arange(0, prompt_length,
                                          dtype=torch.long).to(device)
    prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
    
    temperatures_tensor = None if not temperature else torch.FloatTensor([temperature]).to(device)
    top_ps_tensor = torch.FloatTensor([top_p]).to(device)
    top_ks_tensor = torch.LongTensor([top_k]).to(device)
    output_index = torch.tensor(prompt_length, dtype=torch.long).to(device)

    for _ in range(output_len):
      # INFERENCE
      logits = self.run_model(input_token_ids_tensor, input_positions_tensor)

      next_token_ids, _ = self.sample(logits, temperatures_tensor, top_ps_tensor, top_ks_tensor)

      curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
      curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
      output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
      token_ids_tensor.index_copy_(1, output_index, output_token_ids)

      input_token_ids_tensor = output_token_ids
      input_positions_tensor = output_index.unsqueeze(dim=-1)

      output_index = output_index + 1
    
    # Detokenization.
    token_ids = token_ids_tensor.squeeze().tolist()

    trimmed_output = token_ids[prompt_length:prompt_length + output_len]
    # This is incorrect - EOS ID could be both <eos> and <end_of_turn>
    if self.tokenizer.eos_id in trimmed_output:
      eos_index = trimmed_output.index(self.tokenizer.eos_id)
      trimmed_output = trimmed_output[:eos_index]
    return self.tokenizer.decode(trimmed_output)
