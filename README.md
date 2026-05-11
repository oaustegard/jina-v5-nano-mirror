# jina-v5-nano-mirror

A pinned, GitHub-hosted mirror of [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) for reproducible offline use. Source files live in this repository; weight binaries are attached as GitHub Release assets.

## Pinned upstream

| | |
|---|---|
| Upstream repo | `jinaai/jina-embeddings-v5-text-nano` |
| Pinned commit SHA | `8a7f00aac812071b69403df470f1038ec85f8925` |
| Release tag | [`v5-nano-8a7f00aa`](https://github.com/oaustegard/jina-v5-nano-mirror/releases/tag/v5-nano-8a7f00aa) |
| Architecture | `JinaEmbeddingsV5Model` (EuroBERT-210M + task LoRA) |
| Parameters | 239M |
| Embedding dim | 768 (Matryoshka: 32 / 64 / 128 / 256 / 512 / 768) |
| Max sequence length | 8192 |
| Pooling | **last-token** (not mean) |
| Output | L2-normalized fp32 |

## License

Upstream model is licensed **[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/)** (see [`LICENSE`](LICENSE)). The license carries forward to this mirror: redistribution is permitted for non-commercial use with attribution. **Do not use for commercial purposes.** Attribution: Jina AI GmbH, [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) @ `8a7f00aa`.

This mirror is unmodified — source files are byte-identical to the pinned upstream commit, and weight assets are byte-identical to the upstream `safetensors` blobs (verified via SHA256, below).

## Why mirror

- **Reproducibility.** A specific commit SHA is pinned, so downstream `.kb`-style artifacts that reference this mirror lock to one set of weights forever.
- **Ephemeral-container friendly.** GitHub Release assets are stable, free, and reachable from environments where `huggingface.co` may not be on the allowlist (`release-assets.githubusercontent.com` typically is).
- **No HF cold-start jitter.** Single-asset `curl` is faster and more predictable than `huggingface_hub` snapshot resolution under load.

## Repository layout

```
LICENSE                       CC-BY-NC-4.0 legal code
README.md                     this file
model/                        source files (configs, custom modeling code, tokenizer)
  config.json
  config_sentence_transformers.json
  configuration_eurobert.py
  configuration_jina_embeddings_v5.py
  custom_st.py
  modeling_eurobert.py
  modeling_jina_embeddings_v5.py
  modules.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  adapters/{retrieval,text-matching,clustering,classification}/adapter_config.json
scripts/
  embed.py                    torch loader (all 4 tasks)
  smoke.py                    torch path cosine sanity check
  embed_onnx.py               torch-free ONNX loader (retrieval only)
  smoke_onnx.py               ONNX retrieval correctness + parity-vs-torch
  export_onnx.py              build-time: merge retrieval adapter, export ONNX
requirements.txt              torch path runtime deps
requirements-onnx.txt         ONNX path runtime deps (no torch)
```

Weight binaries (`model.safetensors`, 4 × `adapter_model.safetensors`) are **not** in the tree — see the Release.

## Release assets (`v5-nano-8a7f00aa`)

### Torch path

| Asset | Size | SHA256 |
|---|---:|---|
| `model.safetensors` | 423,543,680 | `f07de152e254b5d03b1273e0bca0a11dfade686508503500d460813428539651` |
| `adapter_retrieval.safetensors` | 13,586,992 | `bc1afe96601bbb5198cd5553fa17d74bd77df3e30b515e50139980dd3a933cba` |
| `adapter_text-matching.safetensors` | 13,586,992 | `16c3bb35e45cdbfd877988019815f5c9661940beeab90fea186d10a0df9e6cd8` |
| `adapter_clustering.safetensors` | 13,586,992 | `80dd657af859b61c286872f15af83cb3bfd3456e9b6201398fee2ef701d06e21` |
| `adapter_classification.safetensors` | 13,586,992 | `c106c3748866900ecd22edbd49ba124c6c50ae898a53f5cd3ce10a5c091f6f28` |

### ONNX path (retrieval-only)

| Asset | Size | SHA256 |
|---|---:|---|
| `model.onnx` | 847,354,038 | `9f45091f1a1bc0affdd89245ca56928c7cc7ffefa79403782e1323eec9513ae6` |

The ONNX export has the **retrieval LoRA adapter merged into the base weights** (encoder forward only — pooling, Matryoshka truncate, and L2-normalize live in the loader). Single-file fp32, opset 17. For the other tasks (text-matching / clustering / classification) use the torch path.

The loader verifies SHA256 of every downloaded asset before installing it into the cache.

## Install matrix

Two runtime options. Pick one based on the host environment:

| Path | Requirements | Tasks supported | Use when |
|---|---|---|---|
| **Torch** | `torch`, `transformers<5`, `peft`, `safetensors`, `huggingface_hub`, `numpy` (see `requirements.txt`) | retrieval, text-matching, clustering, classification | Local dev; CCotw; anywhere torch is already on the host. |
| **ONNX** | `onnxruntime`, `tokenizers`, `numpy` (see `requirements-onnx.txt`) | retrieval only | Ephemeral containers where torch is blocked (e.g. claude.ai project containers, ~2 GB torch eats the layer budget). |

The upper bound on `transformers<5` in `requirements.txt` is deliberate — transformers 5.x has a dynamic-module-cache regression that breaks loading this model's chained relative imports. Validated against `transformers==4.57.1`.

## Quick start

### Torch path

```bash
git clone https://github.com/oaustegard/jina-v5-nano-mirror.git
cd jina-v5-nano-mirror
pip install -r requirements.txt
python scripts/smoke.py
```

First run downloads ~478 MB of safetensors into `~/.cache/jina-v5-nano-mirror/sha-8a7f00aac8/` and verifies SHA256. Subsequent runs are zero-network.

### ONNX path (torch-free)

```bash
git clone https://github.com/oaustegard/jina-v5-nano-mirror.git
cd jina-v5-nano-mirror
pip install -r requirements-onnx.txt
python scripts/embed_onnx.py "test embedding"
```

First run downloads ~847 MB ONNX (with merged retrieval adapter) into `~/.cache/jina-v5-nano-mirror/sha-8a7f00aac8-onnx/` and verifies SHA256. The tokenizer is copied in from the source tree, no second download. Subsequent runs are zero-network.

To run both paths and verify they match (mean cos > 0.99 across 6 (query, relevant, unrelated) triples + Matryoshka 512/256), install both requirement sets and run `python scripts/smoke_onnx.py`.

## Programmatic use

### Torch path

```python
from scripts.embed import embed

# Document side (passages you'll search)
docs = embed(
    ["Vitamin D deficiency causes fatigue, bone pain, muscle weakness."],
    task="retrieval",
    prompt_name="document",
    max_length=512,
)  # → (N, 768) float32, L2-normalized

# Query side (asymmetric prompt)
q = embed(
    ["symptoms of vitamin D deficiency"],
    task="retrieval",
    prompt_name="query",
    max_length=512,
)
cosine = float(q[0] @ docs[0])
```

Matryoshka truncation is supported via `truncate_dim`:

```python
v256 = embed(["…"], task="retrieval", truncate_dim=256)  # → (1, 256)
```

### ONNX path

Same shape, retrieval-only, no `task=` argument (the retrieval adapter is baked into the export). `dim` replaces `truncate_dim`:

```python
from scripts.embed_onnx import embed

docs = embed(
    ["Vitamin D deficiency causes fatigue, bone pain, muscle weakness."],
    prompt_name="document",
    max_length=512,
)  # → (N, 768) float32, L2-normalized
q = embed(["symptoms of vitamin D deficiency"], prompt_name="query")
cosine = float(q[0] @ docs[0])

# Matryoshka:
v256 = embed(["…"], dim=256)  # → (1, 256), valid dims: 32/64/128/256/512/768
```

### Pooling decision (ONNX)

The ONNX graph is **encoder forward only** — last-token pooling, Matryoshka truncate, and L2-normalize are all numpy ops in `embed_onnx.py`. The truncate has to happen pre-normalize (otherwise Matryoshka dims aren't unit vectors), so baking pooling+normalize into the graph would foreclose Matryoshka without a separate per-dim export. Numpy ops are cheap; the graph stays general.

### Tasks

| `task=` | LoRA adapter | Use when |
|---|---|---|
| `retrieval` | retrieval | dense IR, RAG, semantic search (default) |
| `text-matching` | text-matching | symmetric similarity, dedup |
| `clustering` | clustering | unsupervised grouping |
| `classification` | classification | downstream linear classifiers |

### Prompts

The model expects asymmetric prefixes for retrieval:

| `prompt_name=` | Prefix |
|---|---|
| `query` | `"Query: "` |
| `document` | `"Document: "` (default) |

These are applied automatically by `embed()` and match `config_sentence_transformers.json` in this mirror.

### Cache location

Default: `~/.cache/jina-v5-nano-mirror/<short-sha>/`. Override via `JINA_V5_NANO_CACHE=/some/path` (the `<short-sha>` subdir is appended automatically).

## Feedstock for `.kb`

This mirror is the embedder dependency for a portable knowledgebase format (`.kb` — manifest + binary vectors + chunks) built on top of [remex](https://github.com/oaustegard/remex)/[remax](https://github.com/oaustegard/remax) centered SimHash binarization. The companion benchmark in [remax PR #44](https://github.com/oaustegard/remax/pull/44) validated this model end-to-end on BEIR/SciFact (1-bit retrieval Δ = −0.028 nDCG@10 vs fp32, zero-training).

The `.kb` format spec and reference packer/skill are tracked separately and will link back here once filed.

## What this mirror does *not* include

- **Per-task ONNX exports** for text-matching / clustering / classification. Only the retrieval adapter is merged into the current ONNX graph. If a torch-free consumer needs another task, file a follow-up — same export pipeline, different `set_adapter()` call.
- **INT8 / dynamic quantization** of the ONNX graph. Possible future asset, not now.
- **Pooling baked into the ONNX graph.** Encoder-only export, by design — see "Pooling decision" above.
- **Sibling models** (`jina-embeddings-v5-text-small`, future v6, etc.). Each gets its own mirror following the same pattern, not refactored into a multi-model repo until at least two siblings exist.

## Reproducing the mirror

To verify the source files match upstream:

```bash
huggingface-cli download jinaai/jina-embeddings-v5-text-nano \
    --revision 8a7f00aac812071b69403df470f1038ec85f8925 \
    --local-dir /tmp/upstream
diff -r /tmp/upstream/{config.json,*.py,modules.json,special_tokens_map.json,tokenizer_config.json} \
        model/
sha256sum /tmp/upstream/model.safetensors  # should match table above
```

To verify Release assets match the SHA256s in the table above, redownload and run `sha256sum` against each asset.
