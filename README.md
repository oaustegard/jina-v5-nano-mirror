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
  embed.py                    reference loader, exposes embed()
  smoke.py                    end-to-end cosine sanity check
```

Weight binaries (`model.safetensors`, 4 × `adapter_model.safetensors`) are **not** in the tree — see the Release.

## Release assets (`v5-nano-8a7f00aa`)

| Asset | Size | SHA256 |
|---|---:|---|
| `model.safetensors` | 423,543,680 | `f07de152e254b5d03b1273e0bca0a11dfade686508503500d460813428539651` |
| `adapter_retrieval.safetensors` | 13,586,992 | `bc1afe96601bbb5198cd5553fa17d74bd77df3e30b515e50139980dd3a933cba` |
| `adapter_text-matching.safetensors` | 13,586,992 | `16c3bb35e45cdbfd877988019815f5c9661940beeab90fea186d10a0df9e6cd8` |
| `adapter_clustering.safetensors` | 13,586,992 | `80dd657af859b61c286872f15af83cb3bfd3456e9b6201398fee2ef701d06e21` |
| `adapter_classification.safetensors` | 13,586,992 | `c106c3748866900ecd22edbd49ba124c6c50ae898a53f5cd3ce10a5c091f6f28` |

The loader verifies SHA256 of every downloaded asset before installing it into the cache.

## Quick start

```bash
git clone https://github.com/oaustegard/jina-v5-nano-mirror.git
cd jina-v5-nano-mirror
pip install -r requirements.txt
python scripts/smoke.py
```

The upper bound on `transformers<5` in `requirements.txt` is deliberate — transformers 5.x has a dynamic-module-cache regression that breaks loading this model's chained relative imports. Validated against `transformers==4.57.1`.

First run downloads ~478 MB of weights into `~/.cache/jina-v5-nano-mirror/8a7f00aac8/` and verifies SHA256. Subsequent runs are zero-network.

## Programmatic use

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

- **ONNX export.** Discussed in [`claude-workspace#73`](https://github.com/oaustegard/claude-workspace/issues/73); deferred because the primary consumer (CCotw) already ships torch and the export work (LoRA merge + last-token-pool wrapper around custom remote code) doesn't pay for itself yet. May appear in a future release tag if a torch-free consumer surfaces.
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
