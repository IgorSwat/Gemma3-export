from ..itokenizer import ITokenizer
from pathlib import Path
from typing import List, Union, override
import sentencepiece

# Standard tokenizer - a Sentencepiece wrapper
class Tokenizer(ITokenizer):
    def __init__(self, model_path: Union[str, Path]):
        if isinstance(model_path, Path):
            model_path = str(model_path)

        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.boi_id: int = 255999
        self.eoi_id: int = 256000
        self.image_token_placeholder_id: int = self.sp_model.pad_id()

    @override
    def encode(self, s: str) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.EncodeAsIds(s)
        return [self.bos_id] + t

    @override
    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.DecodeIds(t)
