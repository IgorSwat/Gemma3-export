from torchao.quantization.quant_api import quantize_, Int8DynamicActivationInt4WeightConfig, Int4WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_ptq(
    model: nn.Module,
    group_size: int | None = 32,
):
  print(group_size)
  """
    Performs a standard PTQ on the provided model with torchao package.
  """

  # Define PQT configs
  embedding_config = Int4WeightOnlyConfig(group_size=group_size)
  linear_config = Int8DynamicActivationInt4WeightConfig(group_size=group_size)

  # Quantize embeddings
  # quantize_(
  #     model, 
  #     embedding_config, 
  #     filter_fn=lambda m, _: isinstance(m, nn.Embedding)
  # )

  # Quantize linear layers
  quantize_(model, linear_config)

  return model
