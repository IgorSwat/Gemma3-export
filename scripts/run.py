import argparse
import contextlib
import time
import torch
from gemma_executorch.config import get_model_config
from gemma_executorch.standard.runner import Runner as StandardRunner
from gemma_executorch.standard.gemma_casual import GemmaForCausalLM as StandardGemma
from gemma_executorch.standard.tokenizer import Tokenizer as StandardTokenizer
from gemma_executorch.custom.runner import Runner as CustomRunner
from gemma_executorch.custom.gemma_casual import GemmaForCasualLM as CustomGemma
from gemma_executorch.custom.tokenizer import Tokenizer as CustomTokenizer
from pathlib import Path


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main():
    parser = argparse.ArgumentParser(description="Run Gemma 3 model inference.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the model file (.ckpt for standard, .pte for executorch)")
    parser.add_argument("--model-type", type=str, required=True, choices=["270m", "1b", "4b"], help="Model variant")
    parser.add_argument("--impl", type=str, default="standard", choices=["standard", "custom"], help="Implementation to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type for weights")
    parser.add_argument("--prompt", type=str, default="Write a hello world program in python", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Path to the tokenizer file (optional)")
    parser.add_argument("--instruct", type=bool, default=True, help="Use instruction/chat template mode (default: True)")

    args = parser.parse_args()

    config = get_model_config(args.model_type, args.dtype)
    if args.tokenizer:
        config.tokenizer = args.tokenizer
    else:
        config.tokenizer = args.model.parent / "tokenizer.model"
    
    # Check if the tokenizer path is valid
    tokenizer_path = Path(config.tokenizer)
    if not tokenizer_path:
        print(f"\033[1;31m[ERROR]\033[0m Tokenizer not found.")
        return
    
    # Check if we are using ExecuTorch format
    use_pte = args.model.suffix == ".pte"
    
    device = torch.device("cpu") # Default to CPU

    print(f"\033[1;34m[INFO]\033[0m Model Type: {args.model_type}")
    print(f"\033[1;34m[INFO]\033[0m Implementation: {args.impl}")
    print(f"\033[1;34m[INFO]\033[0m Format: {'ExecuTorch (.pte)' if use_pte else 'PyTorch (.ckpt)'}")
    print(f"\033[1;34m[INFO]\033[0m DType: {args.dtype}")

    model = None
    tokenizer = None
    runner = None

    if args.impl == "standard":
        if use_pte:
            # For PTE, we pass the path directly to the runner instead of a model instance
            # We still need tokenizer though. The tokenizer path is in config.
            # Assuming tokenizer file is relative to project root or provided in config
            # Here we try to find it relative to current script/project structure
            tokenizer = StandardTokenizer(str(tokenizer_path))
            model = args.model # Runner takes Path for PTE
        
        else:
            with _set_default_tensor_type(config.get_dtype()):
                model = StandardGemma(config)
                print(f"\033[1;34m[INFO]\033[0m Loading weights from {args.model}...")
                model.load(args.model)
                model.eval()
            tokenizer = StandardTokenizer(str(tokenizer_path))

        runner = StandardRunner(config, model, tokenizer, use_pte=use_pte)

    elif args.impl == "custom":
        if use_pte:
            tokenizer = CustomTokenizer(str(tokenizer_path))
            model = args.model
        else:
            with _set_default_tensor_type(config.get_dtype()):
                model = CustomGemma(config)
                print(f"\033[1;34m[INFO]\033[0m Loading weights from {args.model}...")
                model.load(args.model)
                model.eval()
            tokenizer = CustomTokenizer(str(tokenizer_path))

        runner = CustomRunner(config, model, tokenizer, use_pte=use_pte, use_instruct=args.instruct)

    print(f"\033[1;32m[START]\033[0m Generating response for prompt:\n\033[3m\"{args.prompt}\"\033[0m\n")

    start_time = time.time()
    
    response = runner.generate(
        prompt=args.prompt,
        output_len=args.max_tokens,
        device=device,
        top_k=1
    )
    
    end_time = time.time()
    duration = end_time - start_time
    tokens_per_sec = args.max_tokens / duration if duration > 0 else 0

    print(f"\033[1;36m[OUTPUT]\033[0m{response}\n")
    print(f"\033[1;33m[STATS]\033[0m Time allowed: {duration:.2f}s | Speed: {tokens_per_sec:.2f} tokens/s")

if __name__ == "__main__":
    main()
