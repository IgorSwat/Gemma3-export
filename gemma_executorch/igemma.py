from .config import GemmaConfig
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import torch

# Gemma3 model interface
# A unified interface for all gemma3 family models.
class IGemma(ABC):
    def __init__(self, config: GemmaConfig):
        self.config = config

        assert config.embedding_dim % config.num_attention_heads == 0

    @abstractmethod
    def forward(self, input_tokens: torch.LongTensor, positions: torch.LongTensor):
        """
        Generate output tokens given input tokens and positions.

        Args:
            input_tokens (torch.LongTensor): Input token IDs.
            positions (torch.LongTensor): Position IDs.

        Returns:
            torch.Tensor: Generated output.
        """
        pass

    @abstractmethod
    def load(self, source: Union[Path, str]):
        """
        Load model weights from a local path or HuggingFace repo ID.

        Args:
            source (Union[Path, str]): Filesystem path or HuggingFace repo ID.

        Returns:
            None
        """
        pass
