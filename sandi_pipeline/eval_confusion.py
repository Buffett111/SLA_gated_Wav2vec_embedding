import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

from .data import (
    AudioCollator,
    CollatorConfig,
    AudioScoreDataset,
    autodetect_score_column,
    load_sandi_dataset,
)
from .model import (
    GatedLinearAdapter,
    TemporalAggregator,
    Wav2Vec2Backbone,
    Wav2Vec2BackboneConfig,
    Regressor,
)


@dataclass
class EvalConfig:
    dataset_name: str = "ntnu-smil/sandi-corpus-2025"
    split: str = "dev"
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    out_dir: str = "./outputs"
    adapter_path: str = "./outputs/adapter.pt"
    aggregator_path: str = "./outputs/aggregator.pt"
    regressor_path: str = "./outputs/regressor.pt"
    batch_size: int = 8
    max_seconds: Optional[float] = 30.0
    chunk_seconds: Optional[float] = 30.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Prediction-to-class mapping
    rounding_classes: bool = True  # if True, round to nearest integer classes


@torch.no_grad()
def _aggregate_chunked_embeddings(e_chunks: torch.Tensor, owner_idx: torch.Tensor, num_examples: int) -> torch.Tensor:
    device = e_chunks.device
    dim = e_chunks.size(-1)
    sums = torch.zeros(num_examples, dim, device=device)
    counts = torch.zeros(num_examples, 1, device=device)
    sums.index_add_(0, owner_idx, e_chunks)
    ones = torch.ones(e_chunks.size(0), 1, device=device)
    counts.index_add_(0, owner_idx, ones)
    return sums / counts.clamp(min=1e-6)


def evaluate_and_plot(cfg: EvalConfig) -> str:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    os.makedirs(cfg.out_dir, exist_ok=True)

    raw = load_sandi_dataset(cfg.dataset_name)
    d = raw[cfg.split] if hasattr(raw, "keys") and cfg.split in raw.keys() else raw
    score_col = autodetect_score_column(d)
    ds = AudioScoreDataset(d, score_col)

    collator = AudioCollator(
        CollatorConfig(model_name=cfg.model_name, max_seconds=cfg.max_seconds, chunk_seconds=cfg.chunk_seconds)
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    backbone = Wav2Vec2Backbone(
        Wav2Vec2BackboneConfig(model_name=cfg.model_name, trainable=False)
    ).to(cfg.device)
    feature_dim = backbone.model.config.hidden_size
    adapter = GatedLinearAdapter(feature_dim=feature_dim, bottleneck_dim=None).to(cfg.device)
    aggregator = TemporalAggregator(feature_dim=feature_dim, attentive=True).to(cfg.device)
    regressor = Regressor(feature_dim, hidden_dim=512, dropout=0.1).to(cfg.device)

    adapter.load_state_dict(torch.load(cfg.adapter_path, map_location=cfg.device))
    aggregator.load_state_dict(torch.load(cfg.aggregator_path, map_location=cfg.device))
    regressor.load_state_dict(torch.load(cfg.regressor_path, map_location=cfg.device))
    adapter.eval(); aggregator.eval(); regressor.eval();

    preds: List[torch.Tensor] = []
    targs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
            frame_mask = backbone.make_frame_mask(batch.get("attention_mask"), frames)
            gated = adapter(frames)
            e = aggregator(gated, attention_mask=frame_mask)
            if "owner_idx" in batch and "num_examples" in batch:
                num_ex = batch["num_examples"]
                num_ex_int = int(num_ex.item()) if isinstance(num_ex, torch.Tensor) else int(num_ex)
                e = _aggregate_chunked_embeddings(e, batch["owner_idx"].to(e.device), num_ex_int)
            y = regressor(e)
            preds.append(y.detach().cpu())
            targs.append(batch["scores"].detach().cpu())

    y_pred = torch.cat(preds).numpy() if preds else np.array([])
    y_true = torch.cat(targs).numpy() if targs else np.array([])
    if y_pred.size == 0:
        raise RuntimeError("No predictions produced. Did the dataset load correctly?")

    if cfg.rounding_classes:
        y_pred_cls = np.rint(y_pred)
        y_true_cls = np.rint(y_true)
    else:
        # fallback: 10 bins across observed range
        smin, smax = float(np.min(y_true)), float(np.max(y_true))
        if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
            smin, smax = 0.0, 10.0
        bins = np.linspace(smin, smax, num=11)
        y_pred_cls = np.digitize(y_pred, bins, right=True)
        y_true_cls = np.digitize(y_true, bins, right=True)

    labels_sorted = sorted(list({int(v) for v in np.concatenate([y_true_cls, y_pred_cls])}))
    cm = confusion_matrix(y_true_cls.astype(int), y_pred_cls.astype(int), labels=labels_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels_sorted)),
        yticks=range(len(labels_sorted)),
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({cfg.split})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    out_path = os.path.join(cfg.out_dir, f"confusion_matrix_{cfg.split}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


if __name__ == "__main__":
    path = evaluate_and_plot(EvalConfig())
    print(f"Saved confusion matrix to {path}")


