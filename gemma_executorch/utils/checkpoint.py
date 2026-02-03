import gc
import json
import os
import torch
from pathlib import Path
from torch import nn
from typing import List, Set


# ANSI Color Codes for stylized logging
CLR_G = "\033[92m" # Green
CLR_Y = "\033[93m" # Yellow
CLR_B = "\033[94m" # Blue
CLR_CYAN = "\033[96m" # Light Blue
CLR_R = "\033[91m" # Red
CLR_END = "\033[0m"


def report_mismatches(missing_keys: List[str], unexpected_keys: List[str]):
    """
    Prints a detailed report of the discrepancy between the checkpoint 
    and the target model state.
    """
    print(f"\n{CLR_B}{'='*60}{CLR_END}")
    print(f"{CLR_CYAN}WEIGHT LOADING REPORT{CLR_END}")
    print(f"{CLR_B}{'='*60}{CLR_END}")

    if not missing_keys and not unexpected_keys:
        print(f"{CLR_G}All weights matched perfectly!{CLR_END}")
    else:
        if missing_keys:
            print(f"{CLR_R}Missing keys (in model but not in checkpoint):{CLR_END}")
            for k in sorted(missing_keys):
                print(f"  - {k}")
        
        if unexpected_keys:
            print(f"\n{CLR_Y}Unexpected keys (in checkpoint but not in model):{CLR_END}")
            for k in sorted(unexpected_keys):
                print(f"  - {k}")
    
    print(f"{CLR_B}{'='*60}{CLR_END}\n")


def load_checkpoint(model: nn.Module, model_path: Path):
    """
    Loads checkpoint weights into the given model.
    Handles both single-file (.ckpt) and sharded directories.
    """
    if os.path.isfile(model_path):
        # Single file load (standard for gemma_pytorch)
        state_dict = torch.load(
            model_path, mmap=True, weights_only=True,
        )['model_state_dict']
        incompatible = model.load_state_dict(state_dict, strict=False)
        report_mismatches(incompatible.missing_keys, incompatible.unexpected_keys)
    else:
        # Sharded load (searching for index file)
        index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
        if not os.path.exists(index_path):
             raise FileNotFoundError(f"Checkpoint path {model_path} is not a file and contains no index.")
             
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        shard_files = list(set(index["weight_map"].values()))
        all_checkpoint_keys: Set[str] = set()
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
            all_checkpoint_keys.update(state_dict.keys())
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()

        # Manual calculation of mismatches for sharded loading
        model_keys = set(model.state_dict().keys())
        missing_keys = list(model_keys - all_checkpoint_keys)
        unexpected_keys = list(all_checkpoint_keys - model_keys)
        report_mismatches(missing_keys, unexpected_keys)
