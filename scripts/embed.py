"""Reference loader for jina-v5-nano-mirror.

Downloads weight assets from this repo's GitHub Release into a per-SHA cache
dir, materializes the upstream HF directory layout there, and exposes a
single ``embed()`` entry point that wraps the model's built-in ``.encode()``.

Pooling, prompt prefixing, adapter selection, and L2-normalization are
performed by the upstream custom modeling code (``last-token`` pool, *not*
mean-pool — the issue spec was wrong on this point). This module is a thin
materialization + ergonomics layer.

Dependencies: torch, transformers, peft, safetensors, numpy, huggingface_hub
(transitive). No onnxruntime. No tokenizers-direct.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import Request, urlopen

import numpy as np

UPSTREAM_REPO = "jinaai/jina-embeddings-v5-text-nano"
UPSTREAM_SHA = "8a7f00aac812071b69403df470f1038ec85f8925"
RELEASE_TAG = "v5-nano-8a7f00aa"

GH_OWNER = "oaustegard"
GH_REPO = "jina-v5-nano-mirror"
RELEASE_BASE = (
    f"https://github.com/{GH_OWNER}/{GH_REPO}/releases/download/{RELEASE_TAG}"
)

TASKS = ("retrieval", "text-matching", "clustering", "classification")
DIM = 768

WEIGHT_ASSETS = {
    "model.safetensors": (
        f"{RELEASE_BASE}/model.safetensors",
        "f07de152e254b5d03b1273e0bca0a11dfade686508503500d460813428539651",
    ),
    "adapters/retrieval/adapter_model.safetensors": (
        f"{RELEASE_BASE}/adapter_retrieval.safetensors",
        "bc1afe96601bbb5198cd5553fa17d74bd77df3e30b515e50139980dd3a933cba",
    ),
    "adapters/text-matching/adapter_model.safetensors": (
        f"{RELEASE_BASE}/adapter_text-matching.safetensors",
        "16c3bb35e45cdbfd877988019815f5c9661940beeab90fea186d10a0df9e6cd8",
    ),
    "adapters/clustering/adapter_model.safetensors": (
        f"{RELEASE_BASE}/adapter_clustering.safetensors",
        "80dd657af859b61c286872f15af83cb3bfd3456e9b6201398fee2ef701d06e21",
    ),
    "adapters/classification/adapter_model.safetensors": (
        f"{RELEASE_BASE}/adapter_classification.safetensors",
        "c106c3748866900ecd22edbd49ba124c6c50ae898a53f5cd3ce10a5c091f6f28",
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "model"
CACHE_ROOT = Path(
    os.environ.get("JINA_V5_NANO_CACHE")
    or Path.home() / ".cache" / "jina-v5-nano-mirror"
) / f"sha-{UPSTREAM_SHA[:10]}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, expected_sha256: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, headers={"User-Agent": "jina-v5-nano-mirror/1.0"})
    with urlopen(req) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        seen = 0
        next_log = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            seen += len(chunk)
            if total and seen >= next_log:
                pct = 100 * seen / total
                sys.stderr.write(f"  {dst.name}: {seen >> 20}/{total >> 20} MB ({pct:.0f}%)\n")
                next_log = seen + (50 << 20)
    got = _sha256(tmp)
    if got != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {dst.name}: got {got}, expected {expected_sha256}"
        )
    tmp.rename(dst)


def materialize(verbose: bool = True) -> Path:
    """Ensure the cache dir has source files + weight assets. Return its path."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    for src in SOURCE_DIR.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(SOURCE_DIR)
        dst = CACHE_ROOT / rel
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel_path, (url, sha) in WEIGHT_ASSETS.items():
        dst = CACHE_ROOT / rel_path
        if dst.exists():
            continue
        if verbose:
            sys.stderr.write(f"fetching {rel_path}\n")
        _download(url, dst, sha)

    return CACHE_ROOT


_model_cache: dict = {}


def _load_model():
    if "model" in _model_cache:
        return _model_cache["model"]
    cache_dir = materialize()
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(cache_dir), trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    _model_cache["model"] = model
    return model


def embed(
    texts: Iterable[str],
    task: str = "retrieval",
    prompt_name: str = "document",
    max_length: Optional[int] = None,
    truncate_dim: Optional[int] = None,
) -> np.ndarray:
    """Encode ``texts`` and return an ``(N, dim)`` ``float32`` numpy array.

    Output is last-token-pooled and L2-normalized per the upstream model
    (handled internally by ``model.encode``).

    Args:
        texts: input strings.
        task: one of ``retrieval``, ``text-matching``, ``clustering``,
            ``classification``. Selects the LoRA adapter.
        prompt_name: ``"query"`` for queries, ``"document"`` for documents/
            passages. Prepends ``"Query: "`` or ``"Document: "`` per
            ``config_sentence_transformers.json``.
        max_length: token cap. Defaults to model's ``max_position_embeddings``
            (8192). For retrieval, 512 is typical.
        truncate_dim: optional Matryoshka truncation (32/64/128/256/512).
            ``None`` returns full 768d.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; pick one of {TASKS}")
    if prompt_name not in ("query", "document"):
        raise ValueError(f"unknown prompt_name {prompt_name!r}; pick 'query' or 'document'")

    texts = list(texts)
    model = _load_model()
    out = model.encode(
        texts=texts,
        task=task,
        prompt_name=prompt_name,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    return out.detach().cpu().float().numpy()


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Embed a single text from the CLI.")
    p.add_argument("text", help="text to embed")
    p.add_argument("--task", default="retrieval", choices=TASKS)
    p.add_argument("--prompt-name", default="document", choices=("query", "document"))
    p.add_argument("--max-length", type=int, default=None)
    args = p.parse_args()

    vec = embed([args.text], task=args.task, prompt_name=args.prompt_name,
                max_length=args.max_length)[0]
    print(f"shape={vec.shape} dtype={vec.dtype} "
          f"norm={float(np.linalg.norm(vec)):.4f} "
          f"first8={vec[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
