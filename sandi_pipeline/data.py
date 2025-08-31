import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import AutoFeatureExtractor, AutoProcessor


DEFAULT_DATASET = "ntnu-smil/sandi-corpus-2025"


def load_sandi_dataset(
    dataset_name: str = DEFAULT_DATASET,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    sampling_rate: int = 16000,
) -> Dataset | DatasetDict:
    """Load the S&I SANDI dataset with an `audio` column decoded.

    When `split` is None, returns a `DatasetDict` if available; otherwise returns a single `Dataset`.
    The `audio` column is cast to datasets.Audio with the target sampling rate for consistent processing.
    """

    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    def _cast_audio(d: Dataset) -> Dataset:
        if "audio" in d.features and isinstance(d.features["audio"], Audio):
            # Recast to ensure sampling rate is as expected
            return d.cast_column("audio", Audio(sampling_rate=sampling_rate))
        elif "audio" in d.column_names:
            return d.cast_column("audio", Audio(sampling_rate=sampling_rate))
        return d

    if isinstance(ds, DatasetDict):
        return DatasetDict({k: _cast_audio(v) for k, v in ds.items()})
    else:
        return _cast_audio(ds)


SCORE_NAME_CANDIDATES = [
    "sla_score",
    "score",
    "overall_score",
    "si_score",
    "rating",
    "total_score",
]


def autodetect_score_column(dataset: Dataset) -> Optional[str]:
    """Heuristically find an SLA score column name.

    Looks for numeric columns whose names contain any of the candidate substrings.
    Returns the first match or None if nothing found.
    """

    numeric_like = []
    for name, feature in dataset.features.items():
        is_numeric = getattr(feature, "dtype", None) in {"float64", "float32", "int64", "int32"}
        if is_numeric:
            numeric_like.append(name)

    lowered = {n: n.lower() for n in numeric_like}
    for cand in SCORE_NAME_CANDIDATES:
        for name, low in lowered.items():
            if cand in low:
                return name
    return None


class AudioScoreDataset(torch.utils.data.Dataset):
    """Dataset returning {waveform, sampling_rate, score, id, speaker_id, test_part}.

    - Uses dynamic score selection: for each row, score is taken from column `sla_{test_part}_score` if present,
      otherwise falls back to the provided `score_column`.
    """

    def __init__(self, dataset: Dataset, score_column: Optional[str]):
        self.dataset = dataset
        self.score_column = score_column

    def __len__(self) -> int:  # noqa: D401
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        row = self.dataset[idx]
        audio = row["audio"]
        if isinstance(audio, dict):
            wav = audio["array"]
            sr = audio["sampling_rate"]
        else:  # already array
            wav = audio
            sr = None

        # Resolve score by test_part mapping if available
        score: Optional[float] = None
        test_part = row.get("test_part")
        if isinstance(test_part, str):
            cand = f"sla_{test_part}_score"
            if cand in row and row[cand] is not None:
                score = float(row[cand])
        if score is None and self.score_column is not None:
            score = float(row[self.score_column])
        if score is None:
            raise KeyError("Could not resolve SLA score. Expected `sla_{test_part}_score` or fallback score column.")

        return {
            "waveform": np.asarray(wav, dtype=np.float32),
            "sampling_rate": sr,
            "score": score,
            "id": row.get("file_id", idx),
            "speaker_id": row.get("speaker_id"),
            "test_part": test_part,
        }


@dataclass
class CollatorConfig:
    model_name: str = "facebook/wav2vec2-base"
    max_seconds: Optional[float] = None
    target_sampling_rate: int = 16000
    chunk_seconds: Optional[float] = None


class AudioCollator:
    """Pads raw waveforms using the model processor.

    Returns a dict with keys compatible with Wav2Vec2-like models: `input_values`, `attention_mask`, plus `scores` and `ids`.
    """

    def __init__(self, config: CollatorConfig):
        # Some wav2vec2 checkpoints (e.g., wav2vec2-large-xlsr-53) ship only a feature extractor,
        # not a tokenizer. Prefer AutoFeatureExtractor and fall back to AutoProcessor.
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(config.model_name)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.max_samples = (
            int(config.max_seconds * config.target_sampling_rate) if config.max_seconds else None
        )
        self.target_sr = config.target_sampling_rate
        self.chunk_samples = (
            int(config.chunk_seconds * config.target_sampling_rate) if config.chunk_seconds else None
        )

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Build chunked list while tracking owner example indices
        chunk_waves: List[np.ndarray] = []
        owner_idx: List[int] = []
        ids = [ex.get("id") for ex in batch]
        scores_per_example = [float(ex["score"]) for ex in batch]

        for ei, ex in enumerate(batch):
            wav = ex["waveform"]
            if self.chunk_samples is not None and len(wav) > self.chunk_samples:
                # split into non-overlapping chunks of chunk_samples
                num_chunks = (len(wav) + self.chunk_samples - 1) // self.chunk_samples
                for c in range(num_chunks):
                    start = c * self.chunk_samples
                    end = min((c + 1) * self.chunk_samples, len(wav))
                    w = wav[start:end]
                    if self.max_samples is not None:
                        w = w[: self.max_samples]
                    chunk_waves.append(w)
                    owner_idx.append(ei)
            else:
                if self.max_samples is not None:
                    wav = wav[: self.max_samples]
                chunk_waves.append(wav)
                owner_idx.append(ei)

        processed = self.processor(
            chunk_waves,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        processed["scores"] = torch.tensor(scores_per_example, dtype=torch.float32)
        processed["ids"] = ids
        processed["owner_idx"] = torch.tensor(owner_idx, dtype=torch.long)
        processed["num_examples"] = torch.tensor(len(batch), dtype=torch.long)
        return processed


def stratified_train_val_indices(scores: List[float], val_ratio: float = 0.1, bins: int = 10, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Create a simple stratified split using score bins.

    Returns lists of train_indices and val_indices.
    """
    rng = random.Random(seed)
    # Compute bin ids
    if len(scores) == 0:
        return [], []
    smin, smax = float(min(scores)), float(max(scores))
    if smax <= smin:
        smax = smin + 1.0
    def to_bin(s: float) -> int:
        pos = (s - smin) / (smax - smin + 1e-8)
        return min(bins - 1, max(0, int(pos * bins)))

    bin_to_indices: Dict[int, List[int]] = {b: [] for b in range(bins)}
    for i, s in enumerate(scores):
        bin_to_indices[to_bin(float(s))].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for b in range(bins):
        inds = bin_to_indices[b]
        rng.shuffle(inds)
        split = int(round(len(inds) * (1.0 - val_ratio)))
        train_idx.extend(inds[:split])
        val_idx.extend(inds[split:])
    return train_idx, val_idx


def build_triplets(indices: List[int], scores: List[float], margin_bins: int = 2, bins: int = 10, seed: int = 17) -> List[Tuple[int, int, int]]:
    """Build triplets (anchor, positive, negative) guided by score proximity.

    - Positive: same score bin as anchor
    - Negative: score bin at least `margin_bins` away
    """
    rng = random.Random(seed)
    if len(indices) < 3:
        return []

    smin, smax = float(min(scores)), float(max(scores))
    if smax <= smin:
        smax = smin + 1.0

    def to_bin(s: float) -> int:
        pos = (s - smin) / (smax - smin + 1e-8)
        return min(bins - 1, max(0, int(pos * bins)))

    bin_to_pool: Dict[int, List[int]] = {b: [] for b in range(bins)}
    for i in indices:
        bin_to_pool[to_bin(float(scores[i]))].append(i)

    triplets: List[Tuple[int, int, int]] = []
    bins_list = list(range(bins))
    for a in indices:
        ba = to_bin(float(scores[a]))
        pos_pool = bin_to_pool[ba]
        if len(pos_pool) < 2:
            continue
        p = rng.choice([i for i in pos_pool if i != a])

        far_bins = [b for b in bins_list if abs(b - ba) >= margin_bins and len(bin_to_pool[b]) > 0]
        if not far_bins:
            continue
        bn = rng.choice(far_bins)
        n = rng.choice(bin_to_pool[bn])
        triplets.append((a, p, n))

    rng.shuffle(triplets)
    return triplets


