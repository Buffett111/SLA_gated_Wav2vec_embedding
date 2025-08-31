from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from einops import reduce
from transformers import AutoModel


@dataclass
class Wav2Vec2BackboneConfig:
    model_name: str = "facebook/wav2vec2-base"
    output_hidden_states: bool = True
    layer: Optional[int] = None  # single layer index
    last_k_layers: int = 4  # if layer is None, average last-k hidden states
    trainable: bool = False


class Wav2Vec2Backbone(nn.Module):
    """Wrapper to produce frame-level features from Wav2Vec2-like models."""

    def __init__(self, config: Wav2Vec2BackboneConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name, output_hidden_states=config.output_hidden_states
        )
        if not config.trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def _get_encoder_layers(self):
        # Handle common wav2vec2 model structures
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
            return self.model.encoder.layers
        if (
            hasattr(self.model, "wav2vec2")
            and hasattr(self.model.wav2vec2, "encoder")
            and hasattr(self.model.wav2vec2.encoder, "layers")
        ):
            return self.model.wav2vec2.encoder.layers
        return None

    def _get_feature_extractor(self):
        if hasattr(self.model, "feature_extractor"):
            return self.model.feature_extractor
        if hasattr(self.model, "wav2vec2") and hasattr(self.model.wav2vec2, "feature_extractor"):
            return self.model.wav2vec2.feature_extractor
        return None

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_last_encoder_layers(self, num_layers: int, freeze_feature_extractor: bool = True) -> None:
        """Unfreeze the last N transformer encoder layers. CNN feature extractor stays frozen by default."""
        self.freeze_all()
        fe = self._get_feature_extractor()
        if fe is not None:
            if freeze_feature_extractor:
                for p in fe.parameters():
                    p.requires_grad = False
            else:
                for p in fe.parameters():
                    p.requires_grad = True
        layers = self._get_encoder_layers()
        if layers is not None and num_layers > 0:
            take = min(num_layers, len(layers))
            for layer in list(layers)[-take:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_values=input_values, attention_mask=attention_mask)
        if self.config.output_hidden_states:
            if self.config.layer is not None:
                feats = out.hidden_states[self.config.layer]
            else:
                hs: List[torch.Tensor] = list(out.hidden_states)
                take = hs[-self.config.last_k_layers :]
                feats = torch.stack(take, dim=0).mean(dim=0)
        else:
            feats = out.last_hidden_state
        return feats  # (B, T, D)

    @torch.no_grad()
    def make_frame_mask(self, input_attention_mask: Optional[torch.Tensor], frames: torch.Tensor) -> torch.Tensor:
        """Build frame-level mask (B,T) from input sample-level attention mask (B,L).

        For Wav2Vec2, feature frames T are shorter than input length L due to striding.
        This uses the model's internal feature-extractor length computation when available,
        and falls back to proportional scaling otherwise.
        """
        B, T = frames.shape[:2]
        if input_attention_mask is None:
            return torch.ones(B, T, device=frames.device, dtype=torch.long)
        input_attention_mask = input_attention_mask.to(frames.device)
        input_lengths = input_attention_mask.long().sum(dim=1)
        try:
            feat_lengths = self.model._get_feat_extract_output_lengths(input_lengths).to(frames.device)
        except Exception:
            # Fallback: scale proportionally to the produced sequence length
            L = input_attention_mask.shape[1]
            feat_lengths = (input_lengths.float() * (T / max(1, L))).round().long()
        feat_lengths = torch.clamp(feat_lengths, min=0, max=T)
        arange_t = torch.arange(T, device=frames.device).unsqueeze(0)
        frame_mask = (arange_t < feat_lengths.unsqueeze(1)).long()
        return frame_mask


class GatedLinearAdapter(nn.Module):
    """Feature gates: y = x * sigmoid(Wx + b) with optional bottleneck.

    This implements a simple per-feature gating mechanism that can learn to pass or suppress
    dimensions relevant to SLA scoring.
    """

    def __init__(self, feature_dim: int, bottleneck_dim: Optional[int] = None):
        super().__init__()
        hidden = feature_dim if bottleneck_dim is None else bottleneck_dim
        self.proj = (
            nn.Identity()
            if bottleneck_dim is None
            else nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, feature_dim))
        )
        self.gate = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        g = torch.sigmoid(self.gate(z))
        return x * g


class TemporalAggregator(nn.Module):
    """Simple aggregation from frames to utterance: mean or attentive mean."""

    def __init__(self, feature_dim: int, attentive: bool = False):
        super().__init__()
        self.attentive = attentive
        if attentive:
            self.score = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        if attention_mask is not None:
            # Convert mask to (B, T, 1)
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
        else:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype).unsqueeze(-1)

        if self.attentive:
            weights = self.score(x)  # (B, T, 1)
            weights = torch.softmax(weights.masked_fill(mask == 0, -1e9), dim=1)
            e = (weights * x * mask).sum(dim=1)  # (B, D)
            denom = (weights * mask).sum(dim=1).clamp(min=1e-6)
            e = e / denom
            return e
        else:
            # masked mean
            x_masked = x * mask
            sum_vec = x_masked.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-6)
            return sum_vec / lengths


class Regressor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            self.head = nn.Linear(feature_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.head(e).squeeze(-1)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=eps))


def triplet_loss(emb_anchor: torch.Tensor, emb_pos: torch.Tensor, emb_neg: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    a = l2_normalize(emb_anchor)
    p = l2_normalize(emb_pos)
    n = l2_normalize(emb_neg)
    d_ap = (a - p).pow(2).sum(dim=-1)
    d_an = (a - n).pow(2).sum(dim=-1)
    return torch.relu(d_ap - d_an + margin).mean()


