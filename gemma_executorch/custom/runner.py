from .gemma_casual import GemmaForCasualLM
from .tokenizer import Tokenizer
from ..config import GemmaConfig
from ..irunner import IRunner
from ..utils.chat_template import apply_chat_template
from pathlib import Path
from typing import Any, Union, override
import torch


class Runner(IRunner):

  def __init__(self, config: GemmaConfig, model: Union[GemmaForCasualLM, Path], tokenizer: Tokenizer, 
               use_pte: bool = False, use_instruct: bool = True):
    super().__init__(config, model, tokenizer, use_pte)
    self.use_instruct = use_instruct

  @override
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
    
    input_token_ids = self.tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_positions_tensor = torch.arange(0, input_token_ids_tensor.shape[-1], dtype=torch.long).to(device=device)

    temperature_tensor = torch.FloatTensor([temperature]).to(device)
    top_p_tensor = torch.FloatTensor([top_p]).to(device)
    top_k_tensor = torch.LongTensor([top_k]).to(device)

    generated_tokens = []
    for token in self.__generate_text_basic_stream(
        token_ids=input_token_ids_tensor,
        positions=input_positions_tensor,
        max_new_tokens=output_len,
        eos_token_id=self.tokenizer.encode("<end_of_turn>")[-1],
        temperature=temperature_tensor,
        top_p=top_p_tensor,
        top_k=top_k_tensor,
        device=device
    ):
        token_id = token.squeeze(0).tolist()
        generated_tokens.extend(token_id)

    return self.tokenizer.decode(generated_tokens)
  
  def __generate_text_basic_stream(self, token_ids: torch.LongTensor, positions: torch.LongTensor,
                                   max_new_tokens: int, eos_token_id: int,
                                   temperature: torch.FloatTensor, top_p: torch.FloatTensor, top_k: torch.LongTensor,
                                   device: Any):
    with torch.no_grad():
      for _ in range(max_new_tokens):
          logits = self.run_model(token_ids, positions)[:, -1]
          next_token, _ = self.sample(logits, temperature, top_p, top_k)
          next_token = next_token.unsqueeze(dim=0)
          # next_token = torch.argmax(logits, dim=-1, keepdim=True)

          if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
              break

          yield next_token
          
          token_ids = torch.cat([token_ids, next_token], dim=1)
          positions = torch.cat([positions, torch.tensor([positions[-1] + 1], device=device)], dim=0)
