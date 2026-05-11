"""Torch-free reference loader for jina-v5-nano-mirror.

Loads the retrieval-adapter-merged ONNX export and runs inference using only
``onnxruntime``, ``tokenizers``, ``numpy``, and the stdlib. No torch, no
transformers, no peft — suitable for ephemeral containers (e.g. claude.ai
project containers) where torch is blocked.

The ONNX graph is encoder-only (no pooling baked in). This module performs:

  1. Asymmetric prompt prefixing (``Query: `` / ``Document: ``)
  2. Tokenization via ``tokenizers`` (HF fast tokenizer, no transformers wrapper)
  3. Encoder forward via ``onnxruntime`` CPU provider
  4. **Last-token pool** (gather the index ``mask.sum(-1) - 1`` from each row)
  5. Optional Matryoshka truncation (``dim`` arg)
  6. L2-normalize

The ONNX export merges the **retrieval** LoRA adapter into the base weights,
so this loader supports retrieval only. For text-matching / clustering /
classification, use the torch path (``embed.py``).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import Request, urlopen

import numpy as np

# Hard sanity check: if torch is on path, importing this module is fine, but
# *using* torch would betray the "torch-free" premise. We can't ban it without
# also refusing perfectly valid environments where torch happens to be
# installed alongside (e.g. CCotw). So this is a soft signal only.
if "torch" in sys.modules:
    sys.stderr.write(
        "[embed_onnx] note: torch is already imported in this process; "
        "this module itself does not use it.\n"
    )

UPSTREAM_SHA = "8a7f00aac812071b69403df470f1038ec85f8925"
RELEASE_TAG = "v5-nano-8a7f00aa"
GH_OWNER = "oaustegard"
GH_REPO = "jina-v5-nano-mirror"
RELEASE_BASE = (
    f"https://github.com/{GH_OWNER}/{GH_REPO}/releases/download/{RELEASE_TAG}"
)

DIM = 768
ALLOWED_DIMS = (32, 64, 128, 256, 512, 768)

# ONNX release assets. ``model.onnx_data`` is optional — set ``sha256`` to None
# to indicate "fetch if a SHA256 is published in release notes, otherwise skip".
# Both SHA256s and the presence of the sidecar are determined at export time;
# the values below are filled in once the release assets are uploaded.
ONNX_ASSETS = {
    "model.onnx": {
        "url": f"{RELEASE_BASE}/model.onnx",
        "sha256": "9f45091f1a1bc0affdd89245ca56928c7cc7ffefa79403782e1323eec9513ae6",
        "required": True,
    },
    # External-data sidecar — only present if the export was forced over the
    # 2GB single-file boundary. Current release ships single-file (847 MB).
    "model.onnx_data": {
        "url": f"{RELEASE_BASE}/model.onnx_data",
        "sha256": None,
        "required": False,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "model"
CACHE_ROOT = Path(
    os.environ.get("JINA_V5_NANO_CACHE")
    or Path.home() / ".cache" / "jina-v5-nano-mirror"
) / f"sha-{UPSTREAM_SHA[:10]}-onnx"

TOKENIZER_FILENAME = "tokenizer.json"
PROMPT_PREFIXES = {"query": "Query: ", "document": "Document: "}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, expected_sha256: Optional[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, headers={"User-Agent": f"{GH_REPO}/onnx-1.0"})
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
                sys.stderr.write(
                    f"  {dst.name}: {seen >> 20}/{total >> 20} MB ({pct:.0f}%)\n"
                )
                next_log = seen + (50 << 20)
    if expected_sha256 and not expected_sha256.startswith("__"):
        got = _sha256(tmp)
        if got != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {dst.name}: got {got}, expected {expected_sha256}"
            )
    tmp.rename(dst)


def materialize(verbose: bool = True) -> Path:
    """Ensure ``CACHE_ROOT`` has ``tokenizer.json`` + ONNX asset(s). Return path."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Tokenizer ships in the source tree — copy it next to the model so
    # consumers only need one cache dir.
    import shutil

    tok_src = SOURCE_DIR / TOKENIZER_FILENAME
    tok_dst = CACHE_ROOT / TOKENIZER_FILENAME
    if tok_src.exists() and (not tok_dst.exists()
                             or tok_dst.stat().st_size != tok_src.stat().st_size):
        shutil.copy2(tok_src, tok_dst)

    # Sidecar handling: HEAD the optional asset; if it 404s, it doesn't exist.
    for name, spec in ONNX_ASSETS.items():
        dst = CACHE_ROOT / name
        if dst.exists():
            continue
        if not spec["required"]:
            try:
                req = Request(spec["url"], method="HEAD",
                              headers={"User-Agent": f"{GH_REPO}/onnx-1.0"})
                with urlopen(req) as r:
                    if r.status >= 400:
                        continue
            except Exception:
                continue  # sidecar absent — single-file export
        if verbose:
            sys.stderr.write(f"fetching {name}\n")
        _download(spec["url"], dst, spec.get("sha256"))

    return CACHE_ROOT


_session_cache: dict = {}


def _load_session():
    if "session" in _session_cache:
        return _session_cache["session"], _session_cache["tokenizer"]
    cache_dir = materialize()

    import onnxruntime as ort
    from tokenizers import Tokenizer

    onnx_path = cache_dir / "model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"model.onnx not found at {onnx_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(cache_dir / TOKENIZER_FILENAME))
    # Pad with EOS (128001), per upstream tokenizer_config.json.
    tokenizer.enable_padding(pad_id=128001, pad_token="<|end_of_text|>")
    _session_cache["session"] = session
    _session_cache["tokenizer"] = tokenizer
    return session, tokenizer


def _tokenize(texts: list[str], tokenizer, max_length: int):
    """Tokenize and pad to the batch's longest sequence (truncate to max_length)."""
    tokenizer.enable_truncation(max_length=max_length)
    encoded = tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    return ids, mask


def embed(
    texts: Iterable[str],
    prompt_name: str = "document",
    max_length: int = 512,
    dim: int = 768,
) -> np.ndarray:
    """Encode ``texts`` with the retrieval adapter merged into the ONNX graph.

    Args:
        texts: input strings.
        prompt_name: ``"query"`` or ``"document"``. Prepends the asymmetric
            prefix from ``config_sentence_transformers.json``.
        max_length: token cap (default 512, matches retrieval defaults).
        dim: Matryoshka truncation. Must be one of
            ``(32, 64, 128, 256, 512, 768)``.

    Returns:
        ``(N, dim)`` float32 numpy array, last-token-pooled and L2-normalized.
    """
    if prompt_name not in PROMPT_PREFIXES:
        raise ValueError(
            f"unknown prompt_name {prompt_name!r}; pick 'query' or 'document'"
        )
    if dim not in ALLOWED_DIMS:
        raise ValueError(f"dim must be one of {ALLOWED_DIMS}, got {dim}")

    prefix = PROMPT_PREFIXES[prompt_name]
    inputs = [f"{prefix}{t}" for t in texts]
    if not inputs:
        return np.zeros((0, dim), dtype=np.float32)

    session, tokenizer = _load_session()
    ids, mask = _tokenize(inputs, tokenizer, max_length=max_length)

    hidden = session.run(
        ["last_hidden_state"],
        {"input_ids": ids, "attention_mask": mask},
    )[0]  # (N, S, 768) float32

    # Last-token pool: index per row is mask.sum(-1) - 1.
    lengths = mask.sum(axis=1) - 1  # (N,)
    rows = np.arange(hidden.shape[0])
    pooled = hidden[rows, lengths]  # (N, 768)

    if dim != DIM:
        pooled = pooled[:, :dim]

    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid /0 on degenerate inputs
    return (pooled / norms).astype(np.float32)


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Embed a single text via ONNX.")
    p.add_argument("text", help="text to embed")
    p.add_argument("--prompt-name", default="document", choices=("query", "document"))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dim", type=int, default=768, choices=ALLOWED_DIMS)
    args = p.parse_args()

    vec = embed(
        [args.text],
        prompt_name=args.prompt_name,
        max_length=args.max_length,
        dim=args.dim,
    )[0]
    print(
        f"shape={vec.shape} dtype={vec.dtype} "
        f"norm={float(np.linalg.norm(vec)):.4f} "
        f"first8={vec[:8].tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
