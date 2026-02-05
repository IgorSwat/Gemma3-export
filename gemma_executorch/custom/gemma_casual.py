from .modules import RMSNorm, TransformerBlock
from .utils import compute_rope_params, load_weights_into_gemma
from ..config import GemmaConfig
from ..igemma import IGemma
from ..types import AttentionType
from huggingface_hub import hf_hub_download
from pathlib import Path
from safetensors.torch import load_file
from typing import override, Union
import torch
import torch.nn as nn


class GemmaForCasualLM(IGemma, nn.Module):
    def __init__(self, config: GemmaConfig):
        nn.Module.__init__(self)
        IGemma.__init__(self, config)

        # Ensure a correct number of attention layers
        assert config.attn_types is not None and len(config.attn_types) == config.num_hidden_layers
        
        # Main model parameters
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim, dtype=config.get_dtype())

        self.blocks = nn.ModuleList([
            TransformerBlock(config, attn_type) for attn_type in config.attn_types
        ])

        self.final_norm = RMSNorm(config.embedding_dim, eps=1e-6)
        self.out_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False, dtype=config.get_dtype())
        self.config = config

        # Reusable utilities    
        cos_local, sin_local = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_wave_length[AttentionType.LOCAL_SLIDING],
            context_length=config.max_position_embeddings,
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_wave_length[AttentionType.GLOBAL],
            context_length=config.max_position_embeddings,
            dtype=torch.float32,
        )
        
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)
    
    # Initialize model weights
    @override
    def load(self, model_src: Union[Path, str]):
        """
        Loads model weights from a HuggingFace Hub repo or a local directory.

        Args:
            model_src (Union[Path, str]): Either a HuggingFace repo_id (str) or a local directory (Path or str)
                containing 'model.safetensors'.
        """
        model_src = str(model_src)
        if Path(model_src).is_dir():
            weights_file = Path(model_src) / "model.safetensors"
        else:
            weights_file = hf_hub_download(
                repo_id=model_src,
                filename="model.safetensors",
                local_dir=".",
            )
        weights_dict = load_file(str(weights_file))
        load_weights_into_gemma(self, self.config, weights_dict)
    
    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    
        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)
    
        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.config.sliding_window_size).T
    
        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        return mask_global, mask_local

    @override
    def forward(self, input_ids: torch.LongTensor, input_positions: torch.LongTensor):
        # NOTE: we pass input positions for compatibility with standard RNE LLM API,
        # but this implementation does not use it.

        # Forward pass
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.config.embedding_dim ** 0.5)
        # mask_global, mask_local = None, None
        # if seq_len > 1:
        #     mask_global, mask_local = self._create_masks(seq_len, x.device)
        mask_global, mask_local = self._create_masks(input_positions[-1].item() + 1, x.device)
        mask_global = mask_global[input_positions, :]
        mask_local = mask_local[input_positions, :]
        # print(mask_global.shape, mask_local.shape)

        for block in self.blocks:
            x = block(
                x,
                input_positions,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)

        logits = self.out_head(x.to(self.config.get_dtype()))

        return logits