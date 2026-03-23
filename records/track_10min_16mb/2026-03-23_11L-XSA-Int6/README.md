# Record: 11L XSA4 + Int6+zstd + EMA + GPTQ-lite + Tight SWA (mean val_bpb=1.1299)

## Summary
- Mean val_bpb **1.1299** (3-seed), best 1.1295 — beats SOTA 1.1428 by **-0.013**
- 11-layer transformer with XSA on last 4 layers, int6 quantization + zstd-22 compression
- Full technique stack: EMA, GPTQ-lite, Tight SWA, Late QAT, Partial RoPE, LN Scale
- No TTT. Training: ~5950 steps in 600s on 8xH100 (SDPA attention)

## Approach

11-layer d=512 transformer with MLP 3x ReLU², GQA (8 heads, 4 KV heads), tied embeddings (vocab 1024).

**Architecture:**
- Exclusive Self-Attention (XSA) on last 4 layers — removes self-value bias
- BigramHash(2048, dim=128) + SmearGate for bigram context
- Value Embedding (VE128) shared on layers 9-10 with per-layer scales
- Partial RoPE (16/64 dims) — rotary on 25% of head dims
- LN Scale depth damping: 1/sqrt(layer_idx+1)
- U-Net skip connections between encoder/decoder halves

**Quantization & Compression:**
- Int6 per-row quantization for attention + MLP weights
- GPTQ-lite: per-row optimal clip percentile search (5 candidates, best MSE)
- FP16 passthrough for small/control tensors
- zstd-22 compression (~15.5 MB artifact)

**Training:**
- Muon optimizer: lr=0.025, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- Adam for embeddings: lr=0.035, WD=0.04
- Warmdown: 3500 iters, Late QAT at threshold 0.15
- EMA decay=0.997
- Tight SWA: collect every 50 steps when lr_scale < 0.2, average up to 12 checkpoints
- Batch: 786K tokens, seq_len=2048
- Grad clip: 0.3

**Evaluation:**
- Sliding window evaluation with stride=64
- No test-time training

## Results (3-seed, sliding window stride=64)

| Seed | Steps | val_bpb |
|------|-------|---------|
| 1337 | 5956 | 1.1295 |
| 42   | 5936 | 1.1301 |
| 7    | 5952 | 1.1301 |
| **Mean±Std** | | **1.1299 ± 0.0003** |

## Comparison to prior SOTA

| Metric | Prior SOTA (thwu1 #180) | Ours |
|--------|------------------------|------|
| Mean BPB | 1.1428 | 1.1299 |
| Layers | 10 | 11 |
| Quantization | Int5-MLP + Int6-Attn | Int6 all |
| Compression | zstd | zstd-22 |
| XSA | No | Last 4 layers |
| EMA | No | 0.997 |
| TTT | No | No |

## Run command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in `train_gpt.py`.
