import os
import json
from dataclasses import dataclass
from typing import Optional

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
)


@dataclass
class ExportConfig:
    dataset_name: str = "ntnu-smil/sandi-corpus-2025"
    split: str = "dev"
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    out_dir: str = "./embeddings"
    adapter_path: str = "./outputs/adapter.pt"
    aggregator_path: str = "./outputs/aggregator.pt"
    batch_size: int = 8
    max_seconds: Optional[float] = 30.0
    chunk_seconds: Optional[float] = 30.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def export_embeddings(cfg: ExportConfig) -> None:
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
    adapter.load_state_dict(torch.load(cfg.adapter_path, map_location=cfg.device))
    aggregator.load_state_dict(torch.load(cfg.aggregator_path, map_location=cfg.device))
    adapter.eval(); aggregator.eval();

    import numpy as np
    import pathlib
    meta = []
    for batch in loader:
        with torch.no_grad():
            batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
            frame_mask = backbone.make_frame_mask(batch.get("attention_mask"), frames)
            gated = adapter(frames)
            e = aggregator(gated, attention_mask=frame_mask)
            if "owner_idx" in batch and "num_examples" in batch:
                # re-average chunks to utterance-level
                from .train import _aggregate_chunked_embeddings
                num_ex = batch["num_examples"]
                num_ex_int = int(num_ex.item()) if isinstance(num_ex, torch.Tensor) else int(num_ex)
                e = _aggregate_chunked_embeddings(e, batch["owner_idx"].to(e.device), num_ex_int)

        e_np = e.detach().cpu().numpy().astype("float32")
        for i in range(e_np.shape[0]):
            uid = str(batch["ids"][i]) if i < len(batch["ids"]) else f"idx_{len(meta)}"
            out_path = os.path.join(cfg.out_dir, f"{uid}.npy")
            np.save(out_path, e_np[i])
            meta.append({
                "id": uid,
                "score": float(batch["scores"][i].item()),
                "path": out_path,
            })

    with open(os.path.join(cfg.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(meta)} embeddings to {cfg.out_dir}")


if __name__ == "__main__":
    export_embeddings(ExportConfig())


