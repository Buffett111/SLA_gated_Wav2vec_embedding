import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import DatasetDict
from scipy.stats import pearsonr, spearmanr
from transformers import get_linear_schedule_with_warmup

from .data import (
    AudioCollator,
    CollatorConfig,
    AudioScoreDataset,
    autodetect_score_column,
    build_triplets,
    load_sandi_dataset,
    stratified_train_val_indices,
)
from .model import (
    GatedLinearAdapter,
    Regressor,
    TemporalAggregator,
    Wav2Vec2Backbone,
    Wav2Vec2BackboneConfig,
    triplet_loss,
)
from .data import CollatorConfig as _CollatorConfig
from .data import AudioCollator as _AudioCollator


@dataclass
class TrainConfig:
    dataset_name: str = "ntnu-smil/sandi-corpus-2025"
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    batch_size: int = 8
    max_seconds: Optional[float] = 30.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs_adapter: int = 5
    epochs_regressor: int = 30
    margin: float = 0.2
    attentive_pooling: bool = True
    last_k_layers: int = 4
    chunk_seconds: float = 30.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./outputs"
    warmup_steps: int = 600
    seed: int = 42
    # Stage-2 finetuning options
    finetune_last_n_layers: int = 4
    freeze_feature_extractor: bool = True
    train_adapter_stage2: bool = True
    train_aggregator_stage2: bool = True
    backbone_lr: float = 1e-5
    adapter_lr: float = 1e-4
    aggregator_lr: float = 1e-4
    regressor_lr: float = 3e-4
    regressor_hidden_dim: Optional[int] = 512
    regressor_dropout: float = 0.1
    save_finetuned_backbone: bool = False


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _aggregate_chunked_embeddings(e_chunks: torch.Tensor, owner_idx: torch.Tensor, num_examples: int) -> torch.Tensor:
    # Average chunks that belong to the same original example
    device = e_chunks.device
    dim = e_chunks.size(-1)
    sums = torch.zeros(num_examples, dim, device=device)
    counts = torch.zeros(num_examples, 1, device=device)
    sums.index_add_(0, owner_idx, e_chunks)
    ones = torch.ones(e_chunks.size(0), 1, device=device)
    counts.index_add_(0, owner_idx, ones)
    return sums / counts.clamp(min=1e-6)


def compute_embeddings(backbone: Wav2Vec2Backbone, adapter: GatedLinearAdapter, aggregator: TemporalAggregator, batch):
    with torch.no_grad():
        frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
        frame_mask = backbone.make_frame_mask(batch.get("attention_mask"), frames)
    gated = adapter(frames)
    e_chunks = aggregator(gated, attention_mask=frame_mask)
    if "owner_idx" in batch and "num_examples" in batch:
        num_ex = batch["num_examples"]
        num_ex_int = int(num_ex.item()) if isinstance(num_ex, torch.Tensor) else int(num_ex)
        e = _aggregate_chunked_embeddings(e_chunks, batch["owner_idx"].to(e_chunks.device), num_ex_int)
    else:
        e = e_chunks
    return e


def stage1_train_adapter(cfg: TrainConfig) -> tuple:
    os.makedirs(cfg.out_dir, exist_ok=True)
    raw = load_sandi_dataset(cfg.dataset_name)
    # Enforce official splits (simpler and matches S&I Challenge)
    if not (isinstance(raw, DatasetDict) and ("train" in raw) and ("dev" in raw or "validation" in raw)):
        raise ValueError("Expected official splits ('train' and 'dev'/'validation') in the dataset.")

    d_train = raw["train"]
    d_dev = raw["dev"] if "dev" in raw else raw["validation"]
    sc_train = autodetect_score_column(d_train)
    sc_dev = autodetect_score_column(d_dev)
    if sc_train is None and sc_dev is None:
        raise ValueError("Could not detect a numeric SLA score column in the dataset.")
    ds_train = AudioScoreDataset(d_train, sc_train or sc_dev)
    ds_dev = AudioScoreDataset(d_dev, sc_dev or sc_train)
    scores = [float(ds_train[i]["score"]) for i in range(len(ds_train))]
    train_idx = list(range(len(ds_train)))
    val_idx = list(range(len(ds_dev)))
    collator = AudioCollator(
        CollatorConfig(model_name=cfg.model_name, max_seconds=cfg.max_seconds, chunk_seconds=cfg.chunk_seconds)
    )
    # Build loaders
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(ds_dev, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
    backing_ds_for_pack = ds_train

    # Models
    backbone = Wav2Vec2Backbone(
        Wav2Vec2BackboneConfig(model_name=cfg.model_name, trainable=False, last_k_layers=cfg.last_k_layers)
    ).to(cfg.device)
    # Avoid a GPU forward for feature_dim; use hidden size from HF config
    feature_dim = backbone.model.config.hidden_size
    adapter = GatedLinearAdapter(feature_dim=feature_dim, bottleneck_dim=None).to(cfg.device)
    aggregator = TemporalAggregator(feature_dim=feature_dim, attentive=cfg.attentive_pooling).to(cfg.device)

    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Ensure trainable params for adapter/aggregator
    for p in adapter.parameters():
        p.requires_grad = True
    for p in aggregator.parameters():
        p.requires_grad = True

    # Helper to make triplets within a batch using binning over scores
    def _batch_triplets(scores_tensor: torch.Tensor, bins: int = 10, margin_bins: int = 2):
        s = scores_tensor.detach().cpu().tolist()
        if len(s) < 3:
            return []
        smin, smax = float(min(s)), float(max(s))
        if smax <= smin:
            smax = smin + 1.0
        def to_bin(v: float) -> int:
            pos = (v - smin) / (smax - smin + 1e-8)
            return min(bins - 1, max(0, int(pos * bins)))
        bin_to_idx = {b: [] for b in range(bins)}
        for i, v in enumerate(s):
            bin_to_idx[to_bin(v)].append(i)
        trips = []
        all_bins = list(range(bins))
        import random
        for a in range(len(s)):
            ba = to_bin(s[a])
            pos_pool = [i for i in bin_to_idx[ba] if i != a]
            if len(pos_pool) == 0:
                continue
            p = random.choice(pos_pool)
            far_bins = [b for b in all_bins if abs(b - ba) >= margin_bins and len(bin_to_idx[b]) > 0]
            if not far_bins:
                continue
            bn = random.choice(far_bins)
            n = random.choice(bin_to_idx[bn])
            trips.append((a, p, n))
        return trips

    # Stage-1 training over shuffled batches
    for epoch in range(cfg.epochs_adapter):
        adapter.train()
        total_loss = 0.0
        count = 0
        did_check = False
        for batch in train_loader:
            batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Backbone is frozen; run without grad to save memory
            with torch.no_grad():
                frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
                frame_mask = backbone.make_frame_mask(batch.get("attention_mask"), frames)
            # Adapter + aggregator with grad
            gated = adapter(frames)
            e = aggregator(gated, attention_mask=frame_mask)
            if not did_check:
                if not any(p.requires_grad for p in adapter.parameters()):
                    raise RuntimeError("Adapter parameters are not requiring grad; training would be no-op.")
                if not e.requires_grad:
                    raise RuntimeError(
                        "Sanity check failed: utterance embedding E does not require grad. "
                        "Ensure no torch.no_grad wraps adapter/aggregator and parameters require grad."
                    )
                did_check = True
            trips = _batch_triplets(batch["scores"])  # list of (a,p,n) within batch
            if not trips:
                continue
            a_idx = torch.tensor([t[0] for t in trips], device=e.device)
            p_idx = torch.tensor([t[1] for t in trips], device=e.device)
            n_idx = torch.tensor([t[2] for t in trips], device=e.device)
            loss = triplet_loss(e[a_idx], e[p_idx], e[n_idx], margin=cfg.margin)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * len(trips)
            count += len(trips)
        print(f"[Stage1][Epoch {epoch+1}/{cfg.epochs_adapter}] triplet_loss={total_loss / max(1, count):.4f}")

    # Save adapter + aggregator
    torch.save(adapter.state_dict(), os.path.join(cfg.out_dir, "adapter.pt"))
    torch.save(aggregator.state_dict(), os.path.join(cfg.out_dir, "aggregator.pt"))
    # In the official split path, return train and dev indices lengths for later reference
    return backbone, adapter, aggregator, (backing_ds_for_pack, train_idx, val_idx, scores)


def stage2_train_regressor(cfg: TrainConfig, backbone: Wav2Vec2Backbone, adapter: GatedLinearAdapter, aggregator: TemporalAggregator, ds_pack):
    ds, train_idx, val_idx, scores = ds_pack
    collator = AudioCollator(
        CollatorConfig(model_name=cfg.model_name, max_seconds=cfg.max_seconds, chunk_seconds=cfg.chunk_seconds)
    )

    # If we received official splits, train_idx/val_idx are full ranges of ds_train and ds_dev respectively.
    if isinstance(ds, Subset):
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
    else:
        # ds is the training dataset object; need to reload the dev split to build val loader
        raw = load_sandi_dataset(cfg.dataset_name)
        if isinstance(raw, DatasetDict) and ("dev" in raw or "validation" in raw):
            d_val = raw["dev"] if "dev" in raw else raw["validation"]
            sc_val = autodetect_score_column(d_val)
            ds_val = AudioScoreDataset(d_val, sc_val)
            train_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
        else:
            train_loader = DataLoader(Subset(ds, train_idx), batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(Subset(ds, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    # Optionally unfreeze last N encoder layers for finetuning
    if cfg.finetune_last_n_layers and cfg.finetune_last_n_layers > 0:
        try:
            backbone.unfreeze_last_encoder_layers(cfg.finetune_last_n_layers, cfg.freeze_feature_extractor)
        except Exception as e:
            print(f"[Warn] Failed to unfreeze backbone layers: {e}")

    # Determine feature dim from backbone config to avoid synthetic forward shape issues
    feature_dim = backbone.model.config.hidden_size

    reg = Regressor(feature_dim, hidden_dim=cfg.regressor_hidden_dim, dropout=cfg.regressor_dropout).to(cfg.device)
    criterion = nn.MSELoss()
    # Build parameter groups with per-module learning rates
    param_groups = []
    if cfg.finetune_last_n_layers and cfg.finetune_last_n_layers > 0:
        pg_backbone = {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay}
        if len(pg_backbone["params"]) > 0:
            param_groups.append(pg_backbone)
    if cfg.train_adapter_stage2:
        for p in adapter.parameters():
            p.requires_grad = True
        param_groups.append({"params": adapter.parameters(), "lr": cfg.adapter_lr, "weight_decay": cfg.weight_decay})
    else:
        for p in adapter.parameters():
            p.requires_grad = False
    if cfg.train_aggregator_stage2:
        for p in aggregator.parameters():
            p.requires_grad = True
        param_groups.append({"params": aggregator.parameters(), "lr": cfg.aggregator_lr, "weight_decay": cfg.weight_decay})
    else:
        for p in aggregator.parameters():
            p.requires_grad = False
    param_groups.append({"params": reg.parameters(), "lr": cfg.regressor_lr, "weight_decay": cfg.weight_decay})

    opt = torch.optim.AdamW(param_groups)

    steps_per_epoch = max(1, math.ceil(len(train_loader)))
    total_steps = steps_per_epoch * cfg.epochs_regressor
    warmup = min(cfg.warmup_steps, total_steps - 1) if total_steps > 1 else 0
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    for epoch in range(cfg.epochs_regressor):
        reg.train()
        running = 0.0
        denom = 0
        for batch in train_loader:
            batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
            frame_mask = backbone.make_frame_mask(batch.get("attention_mask"), frames)
            gated = adapter(frames)
            e = aggregator(gated, attention_mask=frame_mask)
            if "owner_idx" in batch and "num_examples" in batch:
                num_ex = batch["num_examples"]
                num_ex_int = int(num_ex.item()) if isinstance(num_ex, torch.Tensor) else int(num_ex)
                e = _aggregate_chunked_embeddings(
                    e, batch["owner_idx"].to(e.device), num_ex_int
                )
            if e.size(0) != batch["scores"].size(0):
                raise RuntimeError(
                    f"Size mismatch: embeddings={e.size(0)} vs scores={batch['scores'].size(0)}"
                )
            pred = reg(e)
            loss = criterion(pred, batch["scores"]) 
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reg.parameters(), 5.0)
            opt.step()
            sched.step()

            running += loss.item() * e.size(0)
            denom += e.size(0)

        # validation
        reg.eval()
        val_loss = 0.0
        val_n = 0
        preds = []
        targs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
                frame_mask = backbone.make_frame_mask(batch.get("attention_mask"), frames)
                gated = adapter(frames)
                e = aggregator(gated, attention_mask=frame_mask)
                if "owner_idx" in batch and "num_examples" in batch:
                    num_ex = batch["num_examples"]
                    num_ex_int = int(num_ex.item()) if isinstance(num_ex, torch.Tensor) else int(num_ex)
                    e = _aggregate_chunked_embeddings(
                        e, batch["owner_idx"].to(e.device), num_ex_int
                    )
                pred = reg(e)
                loss = criterion(pred, batch["scores"]) 
                val_loss += loss.item() * e.size(0)
                val_n += e.size(0)
                preds.append(pred.detach().cpu())
                targs.append(batch["scores"].detach().cpu())

        # metrics
        import numpy as np
        y_pred = torch.cat(preds).numpy() if preds else np.array([])
        y_true = torch.cat(targs).numpy() if targs else np.array([])
        if y_pred.size > 0:
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            pcc = float(pearsonr(y_pred, y_true)[0]) if len(y_pred) > 1 else float("nan")
            src = float(spearmanr(y_pred, y_true)[0]) if len(y_pred) > 1 else float("nan")
            acc_05 = float(np.mean(np.abs(y_pred - y_true) <= 0.5))
            acc_10 = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
            print(
                f"[Stage2][Epoch {epoch+1}/{cfg.epochs_regressor}] train_mse={running/max(1,denom):.4f} "
                f"val_mse={val_loss/max(1,val_n):.4f} RMSE={rmse:.4f} PCC={pcc:.4f} SRC={src:.4f} "
                f"ACC@0.5={acc_05:.3f} ACC@1.0={acc_10:.3f}"
            )
        else:
            print(
                f"[Stage2][Epoch {epoch+1}/{cfg.epochs_regressor}] train_mse={running/max(1,denom):.4f} val_mse={val_loss/max(1,val_n):.4f}"
            )

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(reg.state_dict(), os.path.join(cfg.out_dir, "regressor.pt"))
    if cfg.save_finetuned_backbone and any(p.requires_grad for p in backbone.parameters()):
        try:
            torch.save(backbone.model.state_dict(), os.path.join(cfg.out_dir, "backbone_finetuned.pt"))
        except Exception as e:
            print(f"[Warn] Failed to save finetuned backbone: {e}")


def _export_embeddings_after_stage1(cfg: TrainConfig, backbone: Wav2Vec2Backbone, adapter: GatedLinearAdapter, aggregator: TemporalAggregator) -> None:
    os.makedirs(os.path.join(cfg.out_dir, "embeddings_stage1"), exist_ok=True)
    raw = load_sandi_dataset(cfg.dataset_name)
    splits = ["train", "dev"] if isinstance(raw, DatasetDict) else [None]
    for split in splits:
        d = raw[split] if split is not None else raw
        sc = autodetect_score_column(d)
        ds = AudioScoreDataset(d, sc)
        collator = _AudioCollator(_CollatorConfig(model_name=cfg.model_name, max_seconds=cfg.max_seconds, chunk_seconds=cfg.chunk_seconds))
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
        out_dir = os.path.join(cfg.out_dir, "embeddings_stage1", split or "all")
        os.makedirs(out_dir, exist_ok=True)
        import numpy as np
        meta = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                frames = backbone(batch["input_values"], attention_mask=batch.get("attention_mask"))
                gated = adapter(frames)
                e = aggregator(gated, attention_mask=batch.get("attention_mask"))
                if "owner_idx" in batch and "num_examples" in batch:
                    e = _aggregate_chunked_embeddings(e, batch["owner_idx"].to(e.device), int(batch["num_examples"]))
                e_np = e.detach().cpu().numpy().astype("float32")
                for i in range(e_np.shape[0]):
                    uid = str(batch["ids"][i]) if i < len(batch["ids"]) else f"idx_{len(meta)}"
                    p = os.path.join(out_dir, f"{uid}.npy")
                    np.save(p, e_np[i])
                    meta.append({"id": uid, "path": p, "score": float(batch["scores"][i].item())})
        import json
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[Export] Saved {len(meta)} embeddings to {out_dir}")


def train_all(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    backbone, adapter, aggregator, ds_pack = stage1_train_adapter(cfg)

    # Export utterance-level embeddings E for reuse before training the regressor
    try:
        _export_embeddings_after_stage1(cfg, backbone, adapter, aggregator)
    except Exception as e:
        print(f"[Warn] Export embeddings failed: {e}")
    stage2_train_regressor(cfg, backbone, adapter, aggregator, ds_pack)


if __name__ == "__main__":
    train_all()


