from torch.utils.data import DataLoader, Dataset
from torchao.quantization.quant_api import quantize_, Int8DynamicActivationInt4WeightConfig, Int4WeightOnlyConfig
from torchao.quantization.qat import QATConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def run_qat(
    model: nn.Module,
    teacher_model: nn.Module,
    dataset: Dataset,
    group_size: int = 32,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_epochs: int = 1,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    alpha: float = 0.5,  # Weight for soft loss (the lower it gets, the more we value responses from dataset over teacher model)
    temperature: float = 0.8, # Temperature for distillation
):
    """
    Performs QAT on the provided model using a teacher model for distillation.
    """
    model = model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Define QAT configs
    base_config = Int8DynamicActivationInt4WeightConfig(group_size=group_size)
    
    # Prepare step: swaps torch.nn.Linear -> FakeQuantizedLinear
    print("Preparing model for QAT...")
    quantize_(model, QATConfig(base_config, step="prepare"))

    # Apply weight-only quantization to embeddings (activation fake quant not supported)
    # embedding_weight_config = Int4WeightOnlyConfig(group_size=group_size)
    # emb_qat_config = QATConfig(weight_config=embedding_weight_config, step="prepare")
    # quantize_(
    #     model, 
    #     emb_qat_config, 
    #     filter_fn=lambda m, _: isinstance(m, nn.Embedding)
    # )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    print(f"Starting QAT training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (tokens, mask) in enumerate(dataloader):
            tokens = tokens.to(device)
            mask = mask.to(device)

            # Gemma/Llama usually expect inputs [B, T]. Output logits [B, T, V].
            # Shift tokens for next-token prediction
            # input: tokens[:, :-1]
            # target: tokens[:, 1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            target_mask = mask[:, 1:] # Mask corresponding to the prediction positions

            optimizer.zero_grad()

            # Student forward pass
            student_output = model(input_tokens)
            student_logits = student_output.logits if hasattr(student_output, "logits") else student_output

            # Teacher forward pass
            with torch.no_grad():
                teacher_output = teacher_model(input_tokens)
                teacher_logits = teacher_output.logits if hasattr(teacher_output, "logits") else teacher_output

            # Flatten for loss calculation based on mask
            # We only care about positions where mask == 1 (model response)
            active_loss_pos = target_mask.view(-1) == 1
            
            if not active_loss_pos.any():
                continue

            active_student_logits = student_logits.view(-1, student_logits.size(-1))[active_loss_pos]
            active_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))[active_loss_pos]
            active_targets = target_tokens.reshape(-1)[active_loss_pos]

            # Hard Loss (Label Smoothing could be added)
            loss_hard = F.cross_entropy(active_student_logits, active_targets)

            # Soft Loss (Distillation)
            loss_soft = F.kl_div(
                F.log_softmax(active_student_logits / temperature, dim=-1),
                F.softmax(active_teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)

            loss = (1.0 - alpha) * loss_hard + alpha * loss_soft

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f} (Hard: {loss_hard.item():.4f}, Soft: {loss_soft.item():.4f})")

    print("Converting QAT model to quantized model...")
    # Convert: swap FakeQuantizedLinear -> torch.nn.Linear, then quantize using base_config
    # This finalizes the weights to their int representation effectively
    quantize_(model, QATConfig(base_config, step="convert"))
    
    return model

def pad_collate(batch):
    """
    Simple collate function to pad sequences to the max length in the batch.
    Assumes padding token id is 0 (or specific to tokenizer, but usually handled here).
    """
    # batch is a list of tuples (tokens, mask)
    tokens_list, mask_list = zip(*batch)
    
    # Pad tokens
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens_list, batch_first=True, padding_value=0)
    # Pad masks (0 is ignored anyway)
    mask_padded = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=0)
    
    return tokens_padded, mask_padded
