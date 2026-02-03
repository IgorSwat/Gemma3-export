from abc import ABC, abstractmethod
from typing import List

class ITokenizer(ABC):
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of integer token IDs.

        Args:
            text (str): The input string.

        Returns:
            List[int]: The encoded token IDs.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of integer token IDs back into a string.

        Args:
            tokens (List[int]): The list of token IDs.

        Returns:
            str: The decoded string.
        """
        pass
