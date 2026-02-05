import argparse
import contextlib
import sys
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from gemma_executorch.config import get_model_config
from gemma_executorch.standard.gemma_casual import GemmaForCausalLM as StandardGemma
from gemma_executorch.standard.tokenizer import Tokenizer as StandardTokenizer
from gemma_executorch.custom.gemma_casual import GemmaForCasualLM as CustomGemma
from gemma_executorch.custom.tokenizer import Tokenizer as CustomTokenizer
from gemma_executorch.quant.ptq import run_ptq
from gemma_executorch.quant.qat import run_qat
from gemma_executorch.quant.datasets.standard_instruct import StandardInstructDataset
from pathlib import Path
from torch.export import Dim


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main():
    parser = argparse.ArgumentParser(description="Export Gemma 3 model to ExecuTorch.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the source .ckpt file")
    parser.add_argument("--model-type", type=str, required=True, choices=["270m", "1b", "4b"], help="Model variant")
    parser.add_argument("--impl", type=str, default="standard", choices=["standard", "custom"], help="Implementation to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type for weights")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the .pte file")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length for export (default: 1024)")
    parser.add_argument("--quant", type=str, default=None, choices=["ptq", "qat"], help="Quantization method")
    parser.add_argument("--quant-group-size", type=int, default=32, help="Group size for quantization (default: 32, use 0 for per-channel)")
    parser.add_argument("--data", type=Path, help="Path to the dataset (required for QAT)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs for QAT (default: 1)")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Path to the tokenizer file (required for QAT)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use (default: cpu)")

    args = parser.parse_args()

    if args.quant == "qat":
        if not args.data:
            parser.error("--data is required when --quant=qat")
        
        # Resolve tokenizer path if not provided
        if not args.tokenizer:
            args.tokenizer = args.model.parent / "tokenizer.model"
            
        if not args.tokenizer.exists():
            parser.error(f"--tokenizer path not found: {args.tokenizer}. Required for QAT.")

    config = get_model_config(args.model_type, args.dtype)
    if args.max_seq_len:
        config.max_seq_length = args.max_seq_len

    assert config.max_seq_length >= 128
    
    print(f"\033[1;34m[INFO]\033[0m Model Type: {args.model_type}")
    print(f"\033[1;34m[INFO]\033[0m Implementation: {args.impl}")
    print(f"\033[1;34m[INFO]\033[0m DType: {args.dtype}")

    group_size = args.quant_group_size if args.quant_group_size >= 0 else None

    with _set_default_tensor_type(config.get_dtype()):
        if args.impl == "standard":
            model = StandardGemma(config)
        else:
            model = CustomGemma(config)
            
        print(f"\033[1;34m[INFO]\033[0m Loading weights from {args.model}...")
        model.load(args.model)
        model.eval()

        if args.quant == "ptq":
            print(f"\033[1;34m[INFO]\033[0m Applying PTQ (group_size={group_size})...")
            model = run_ptq(model, group_size=group_size)
        elif args.quant == "qat":
            print(f"\033[1;34m[INFO]\033[0m Starting QAT (group_size={group_size}, epochs={args.epochs})...")
            
            # Load teacher model (original float model)
            if args.impl == "standard":
                teacher_model = StandardGemma(config)
                tokenizer = StandardTokenizer(str(args.tokenizer))
            else:
                teacher_model = CustomGemma(config)
                tokenizer = CustomTokenizer(str(args.tokenizer))
            
            teacher_model.load(args.model)
            
            # Load dataset
            dataset = StandardInstructDataset(args.data, tokenizer=tokenizer, apply_bos=False)
            
            model = run_qat(
                model=model,
                teacher_model=teacher_model,
                dataset=dataset,
                group_size=group_size,
                num_epochs=args.epochs,
                device=args.device
            )
            # Ensure model is back on CPU for export
            model = model.to("cpu")

    print(f"\033[1;34m[INFO]\033[0m Starting export process...")

    # Define export conditions
    t = Dim("t", min=1, max=config.max_seq_length)
    dynamic_shapes = (
        {1: t},
        {0: t}
    )

    # Inputs: (tokens, pos)
    inputs = (
        torch.randint(0, config.vocab_size, size=(1, 128), dtype=torch.long),
        torch.arange(0, 128, dtype=torch.long),
    )
    
    exported_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)

    print(f"\033[1;34m[INFO]\033[0m Lowering to ExecuTorch with XNNPACK...")
    
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    ).to_executorch()

    print("\033[1;32m[SUCCESS]\033[0m Exporting succeed!")

    with open(args.output, "wb") as file:
        executorch_program.write_to_file(file)

    print(f"\033[1;34m[INFO]\033[0m Saved to file: {args.output}")

if __name__ == "__main__":
    main()
