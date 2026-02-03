# Gemma 3 ExecuTorch Export

This toolkit provides a streamlined workflow for exporting **Gemma 3** models to the **ExecuTorch** (`.pte`) format. It leverages PyTorch's export capabilities and the XNNPACK backend to enable efficient, on-device inference.

---

## Key Features

*   **Multi-size Support**: Compatible with Gemma 3 `270m`, `1b`, and `4b` variants.
*   **Performance Optimized**: Designed for ExecuTorch with XNNPACK integration for CPU execution.
*   **Modular Architecture**: Separated `standard` and `custom` implementations for flexible deployment.
*   **Validation Tools**: Integrated CLI for immediate inference verification.

---

## Project Structure

*   `gemma_executorch/`: Core library containing architecture definitions and utilities.
    *   `standard/`: Reference implementation of Gemma 3.
    *   `custom/`: Extension point for experimental optimizations.
    *   `config.py` & `types.py`: Configuration and core type definitions.
*   `scripts/`:
    *   `export.py`: Script for lowering PyTorch checkpoints to `.pte`.
    *   `run.py`: Inference runner for both `.ckpt` and `.pte` formats.
*   `models/`: Recommended directory for checkpoints and exported binaries.

---

## Usage Guide

### 1. Exporting a Model

Lower a Gemma 3 checkpoint to ExecuTorch using the `XnnpackPartitioner`.

```bash
python -m scripts.export \
    --model models/gemma3-1b/model.ckpt \
    --model-type 1b \
    --dtype float16 \
    --output models/gemma3-1b/model.pte \
    --max-seq-len 2048
```

### 2. Running Inference

The runner automatically detects the model format based on the file extension.

**ExecuTorch (.pte):**
```bash
python -m scripts.run \
    --model models/gemma3-1b/model.pte \
    --model-type 1b \
    --prompt "Explain quantum computing in three sentences."
```

**PyTorch (.ckpt):**
```bash
python -m scripts.run \
    --model models/gemma3-1b/model.ckpt \
    --model-type 1b \
    --dtype bfloat16 \
    --prompt "What is the capital of France?"
```

---

## Requirements

*   PyTorch (Nightly recommended)
*   [ExecuTorch](https://github.com/pytorch/executorch)
*   `sentencepiece`

---

## Acknowledgments

Developed based on Google's Gemma 3 release and the PyTorch ExecuTorch ecosystem.
