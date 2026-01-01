from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal

from x_transformers.x_transformers import RotaryEmbedding
from .utils import (
    default,
    exists,
    lens_to_mask,
)
from .modules import (
    MelSpec,
    ConvPositionEmbedding,
    SpeedPredictorLayer,
    GaussianCrossEntropyLoss,
)

class SpeedTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth=6,
        heads=8,
        dropout=0.1,
        ff_mult=4,
        qk_norm=None,
        pe_attn_head=None,
        mel_dim=100,
        num_classes=32,
    ):
        super().__init__()
        self.dim_head = dim // heads
        self.num_classes = num_classes
        self.mel_proj = nn.Linear(mel_dim, dim)
        self.conv_layer = ConvPositionEmbedding(dim=dim)
        self.rotary_embed = RotaryEmbedding(self.dim_head)
        self.transformer_blocks = nn.ModuleList([
                SpeedPredictorLayer(
                    dim=dim,
                    heads=heads,
                    dim_head = self.dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head
                ) for _ in range(depth)
            ])
        self.pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),  # nn.ReLU()
            nn.Linear(dim, num_classes)
        )
        # self.initialize_weights()
        
    # def initialize_weights(self):
        
    def forward(self, x, lens):                     # x.shape = [b, seq_len, d_mel]
        seq_len = x.shape[1]
        mask = lens_to_mask(lens, length=seq_len)   # shape = [b, seq_len]
        
        x = self.mel_proj(x)                        # shape = [b, seq_len, h]
        x = self.conv_layer(x, mask)                # shape = [b, seq_len, h]
        
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope=rope)      # shape = [b, seq_len, h]
            
        # sequence pooling
        weights = self.pool(x)                      # shape = [b, seq_len, 1]
        # 将 padding 位置的 weights 设为 -inf
        weights.masked_fill_(~mask.unsqueeze(-1), -torch.finfo(weights.dtype).max)
        weights = F.softmax(weights, dim=1)         # shape = [b, seq_len, 1]
        x = (x * weights).sum(dim=1)                # shape = [b, h]
        
        output = self.classifier(x)                 # shape: [b, num_classes] 
        return output
    
class SpeedMapper:
    def __init__(
        self, 
        num_classes: Literal[32, 72],
        delta: float = 0.25
    ):
        self.num_classes = num_classes
        self.delta = delta
        
        self.max_speed = float(num_classes) * delta
            
        self.speed_values = torch.arange(0.25, self.max_speed + self.delta, self.delta)
        assert len(self.speed_values) == num_classes, f"Generated {len(self.speed_values)} classes, expected {num_classes}"
    
    def label_to_speed(self, label: torch.Tensor) -> torch.Tensor:
        return self.speed_values.to(label.device)[label] # label * 0.25 + 0.25
    
class SpeedPredictor(nn.Module):
    def __init__(
        self,
        mel_spec_kwargs: dict = dict(),
        loss_type: Literal["GCE", "CE"] = "CE",
        arch_kwargs: dict | None = None,
        sigma_factor: int = 2,
        mel_spec_module: nn.Module | None = None,
        num_channels: int = 100,
    ):
        super().__init__()
        self.num_classes = 32
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels
        self.speed_transformer = SpeedTransformer(**arch_kwargs, num_classes=self.num_classes)
        if loss_type == "GCE":
            self.loss = GaussianCrossEntropyLoss(num_classes=self.num_classes, sigma_factor=sigma_factor)
        elif loss_type == "CE":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.speed_mapper = SpeedMapper(self.num_classes)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def predict_speed(self, audio: torch.Tensor, lens: torch.Tensor | None = None):
        # raw wave
        if audio.ndim == 2:
            audio = self.mel_spec(audio).permute(0, 2, 1)

        batch, seq_len, device = *audio.shape[:2], audio.device
        
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)

        logits = self.speed_transformer(audio, lens)
        probs = F.softmax(logits, dim=-1)
        
        pred_class = torch.argmax(probs, dim=-1)
        pred_speed = self.speed_mapper.label_to_speed(pred_class)
        return pred_speed

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        speed: float["b"],      # speed groundtruth
        lens: int["b"] | None = None,  # noqa: F821
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        device = self.device
        pred = self.speed_transformer(inp, lens)
        if isinstance(self.loss, GaussianCrossEntropyLoss):
            loss = self.loss(pred, speed, self.device)  # Pass device for GCE
        else:
            loss = self.loss(pred, speed)
        
        return loss