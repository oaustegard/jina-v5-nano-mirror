# Performance & the remax / remax_kb stack

How the ONNX variants of this mirror perform, and how the model fits the
remax (compression) and remax_kb (KB format + retrieval) libraries.

## The stack: where this model sits

```
text
  │  ── jina-v5-nano-mirror (THIS repo) ─────────────────────────────
  ▼     the embedder. fp32 or q4 retrieval ONNX → 768-d L2-normed float
768-d float
  │  ── remax ───────────────────────────────────────────────────────
  ▼     the compressor. center on corpus mean → SRHT rotate → sign
1-bit StackedSignBit codes  (rank-correct cosine LSH; dim×k bits/doc)
  │  ── remax_kb ─────────────────────────────────────────────────────
  ▼     the KB. .kb pack/read + Hamming search; its JinaONNXEmbedder /
ranked hits      JinaQ4ONNXEmbedder wrap THIS repo's ONNX as the runtime
```

- **jina-v5-nano-mirror (this repo)** — hosts the model weights and the
  reference loaders. The fp32 ONNX (`model.onnx`) and the quantized
  `model.q4.onnx` are *derived* retrieval-only exports (the safetensors are the
  unmodified upstream mirror; the ONNX exports are ours).
- **[remax](https://github.com/oaustegard/remax)** — rank-correct cosine LSH.
  `StackedSignBitQuantizer` turns the float embedding into 1-bit codes (center +
  random rotation + sign, k stacks). This is what makes the *vector* small.
- **[remax_kb](https://github.com/oaustegard/remax_kb)** — the `.kb` format +
  hybrid retrieval. Its `JinaONNXEmbedder` / `JinaQ4ONNXEmbedder` load this
  repo's ONNX to embed queries; the reader does Hamming search over the codes.

## ONNX variants (derived from `model.onnx`)

Measured on 4-vCPU CPU (onnxruntime), 256 muninn chunks for throughput; quality
on the 73-post muninn corpus, 5 acceptance queries (n=5 — directional), full-float
cosine R@5/R@10.

| variant | size | CPU throughput | R@5 / R@10 | fidelity to fp32 | hosted? |
|---|---:|---:|---|---|---|
| **fp32** (`model.onnx`) | 847 MB | 8.5 ch/s | 0.90 / 1.00 | 1.000 (ref) | ✅ release |
| int8 dynamic | 212 MB | **18.3 ch/s (2.2×)** | 0.83 / 1.00 | **0.445 OOD** | build-only |
| **q4** (`model.q4.onnx`) | **170 MB** | 8.9 ch/s | 0.90 / 1.00 | **0.975** | ✅ release |

- **q4 is the recommended quantized variant**: 5× smaller than fp32, retrieval-
  identical (0.975 per-doc cosine; same R@5/R@10), and **domain-robust**. Built
  by `scripts/quantize_onnx.py` (MatMulNBits 4-bit blockwise + int8 embedding
  mop-up — the mop-up is what gets it below int8's size; without it 4-bit-alone
  is ~465 MB because the EuroBERT embedding `Gather` stays fp32).
- **int8 is faster (2.2×) but domain-fragile**: per-tensor dynamic int8 collapsed
  to **0.445** per-doc cosine vs fp32 on out-of-domain (medical/NFCorpus) text,
  while q4 held at 0.975. Use int8 only for high-throughput *indexing* of
  in-domain corpora; it is not hosted (reproduce with `--int8`).
- q4 gives **no CPU speedup** over fp32 (MatMulNBits 4-bit kernel ≈ fp32
  throughput) — its win is **size**, not speed.

## End-to-end through remax (the 1-bit index)

q4 embeddings → remax 1-bit codes, muninn corpus, same gold:

| pipeline | bytes/doc | R@5 / R@10 |
|---|---:|---|
| q4, full-float vectors | 3072 | 0.90 / 1.00 |
| q4 → remax 1-bit, d=512/k=4 | **256** | **0.833 / 1.00** |
| q4 → remax 1-bit, d=256/k=8 (default) | 256 | 0.667 / 0.933 |
| q4 → remax 1-bit, d=768/k=2 | 192 | 0.800 / 1.00 |

**Full-stack compression ledger** (fp32 model + fp32 vector DB → q4 model + 1-bit
index): R@5 **0.90 → 0.833** (−0.067, ~7%), R@10 **1.00 → 1.00** (unchanged).
The loss decomposes as: model quantization **free** (0.90 → 0.90), vector
quantization the entire cost. Pick **d=512/k=4** (not the d=256/k=8 default —
dims beat stacks here).

Efficiency gained: **model 5× smaller** (847 → 170 MB), **vector 12× smaller**
(3072 → 256 B/doc; index @100k docs 307 → 26 MB).

## Caveats

- **Small, not fast, at small N.** The 1-bit *storage* win is real everywhere;
  the 1-bit *search-speed* win is a large-N phenomenon. At muninn scale a numpy
  BLAS float-cosine scan beats the current popcount Hamming scan (0.05 vs 0.50
  ms/query @600 docs). Closing that gap is tracked in
  [remax_kb#15](https://github.com/oaustegard/remax_kb/issues/15).
- **n=5 acceptance queries** — directional, not a leaderboard number. The R@5
  loss is concentrated; R@10 is robust.
- **In-vocabulary queries.** The residual a hosted embedder buys over pure
  lexical (BM25) is *vocabulary-divergent* (paraphrase) queries — not captured by
  these in-vocab acceptance queries.

## Reproduce

```bash
# fp32 export (retrieval adapter merged):
python scripts/export_onnx.py            # -> model.onnx
# quantized variants:
python scripts/quantize_onnx.py model.onnx model.q4.onnx   --bits 4   # 170 MB, recommended
python scripts/quantize_onnx.py model.onnx model.int8.onnx --int8     # 212 MB, fast/fragile
```

Methodology + raw runs: claude-workspace `experiments/{jina-int8-remax_kb,
muninn-embedder-bakeoff,recall-per-byte,rotation-decorrelation}`.
