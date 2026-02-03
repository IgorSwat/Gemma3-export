from ..itokenizer import ITokenizer
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer
from typing import Union


class Tokenizer:
    def __init__(self, model_path: Union[str, Path]):
        if isinstance(model_path, str):
            model_path = Path(model_path)

        self._tok = HFTokenizer.from_file(str(model_path))

        # Attempt to identify EOS and padding tokens
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)