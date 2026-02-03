import argparse
import contextlib
import sys
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from gemma_executorch.config import get_model_config
from gemma_executorch.standard.gemma_casual import GemmaForCausalLM as StandardGemma
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

    args = parser.parse_args()

    config = get_model_config(args.model_type, args.dtype)
    if args.max_seq_len:
        config.max_seq_length = args.max_seq_len

    assert config.max_seq_length >= 128
    
    print(f"\033[1;34m[INFO]\033[0m Model Type: {args.model_type}")
    print(f"\033[1;34m[INFO]\033[0m Implementation: {args.impl}")
    print(f"\033[1;34m[INFO]\033[0m DType: {args.dtype}")

    if args.impl != "standard":
        print(f"\033[1;31m[ERROR]\033[0m Export currently only supports 'standard' implementation.")
        sys.exit(1)

    with _set_default_tensor_type(config.get_dtype()):
        model = StandardGemma(config)
        print(f"\033[1;34m[INFO]\033[0m Loading weights from {args.model}...")
        model.load(args.model)
        model.eval()

    print(f"\033[1;34m[INFO]\033[0m Starting export process...")

    # Define export conditions
    t = Dim("t", min=1, max=config.max_seq_length)
    dynamic_shapes = (
        {1: t},
        {0: t}
    )

    # Inputs: (tokens, pos)
    inputs = (
        torch.randint(0, config.vocab_size, size=(1, 128), dtype=torch.int64),
        torch.arange(0, 128, dtype=torch.int64),
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
