from ...itokenizer import ITokenizer
from ...utils.chat_template import apply_chat_template
from torch.utils.data import Dataset
import json
import torch


class StandardInstructDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: ITokenizer, apply_bos: bool = True):
        """
        Args:
            json_path: Path to the .json file containing the entries.
            tokenizer: An ITokenizer object.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

        self.apply_bos = apply_bos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]
        messages = entry["messages"]
        
        # Extract user and model content
        user_content = messages[0]["content"]
        model_content = messages[1]["content"]

        # Apply chat template
        templated_content = apply_chat_template(user_content, model_content, apply_bos=self.apply_bos)
        user_content_templated, sep, _ = templated_content.partition("model")
        user_content_templated = user_content_templated + sep

        # Tokenize prompt and response
        # Using bos=True for user and eos=True for model to mark sequence bounds
        full_tokens = self.tokenizer.encode(templated_content)
        prompt_tokens = self.tokenizer.encode(user_content_templated)
        # Ensure alignment
        prompt_tokens_length = len(prompt_tokens)
        mask = [0] * prompt_tokens_length + [1] * (len(full_tokens) - prompt_tokens_length)
        
        return (
            torch.tensor(full_tokens, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool)
        )
