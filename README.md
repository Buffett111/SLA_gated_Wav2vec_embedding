## S&I SANDI-2025 Triplet-Gated Pipeline

This repo provides a minimal, modular pipeline to:

- Load `ntnu-smil/sandi-corpus-2025` from Hugging Face Datasets
- Extract frame-level features with frozen Wav2Vec2
- Pass features through a gated-linear adapter (feature gates)
- Aggregate to an utterance embedding `E`
- Train in two stages:
  - Stage 1: Contrastive learning on `E` using triplet loss guided by S&I SLA scores
  - Stage 2: Train a lightweight linear regressor `E -> score`

### Setup

Ensure you have a working Python 3.11 environment (see `setupenv.sh`). Then:

```bash
pip install -r requirements.txt
```

Login to Hugging Face if the dataset requires authentication:

```bash
huggingface-cli login
```

### Train

```bash
python -m sandi_pipeline.train
```

By default, it uses `facebook/wav2vec2-base`. You can adjust configuration by editing `TrainConfig` in `sandi_pipeline/train.py` (batch size, max seconds, pooling, etc.). Models will be saved to `./outputs`.

### Notes

- Adapter: gated-linear as `y = x * sigmoid(Wx + b)` with an optional bottleneck.
- Contrastive: Triplet loss; positives are within same score bin, negatives from sufficiently distant bins.
- After Stage 1, a linear regressor is trained on frozen `E`.

### References

- Triplet learning and feature gating motivation (as requested): [`arxiv:2410.06675`](https://arxiv.org/pdf/2410.06675)
- Method overview link provided by user: [shared conversation](https://chatgpt.com/share/68b1f138-c074-800a-806e-79f9d5458453)


