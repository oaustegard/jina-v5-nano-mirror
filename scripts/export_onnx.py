"""Export jina-v5-nano with the retrieval LoRA adapter merged into ONNX.

Build-time tool, not a runtime dependency. Produces ``model.onnx`` (+ optional
``model.onnx_data`` if external data is needed) suitable for upload as a
Release asset alongside the existing safetensors weights.

Pooling, Matryoshka truncation, and L2-normalize are intentionally **not**
baked into the graph — the loader does those in numpy. Reasoning: last-token
pooling depends on the attention mask in a way that's awkward to express as
an ONNX op, and Matryoshka has to truncate before normalize, which forecloses
"normalize once at graph output". See README §"ONNX path".

Usage::

    python scripts/export_onnx.py --out out/

Requires: torch, transformers, peft, onnx (build-time only — none of these
need to be installed for the runtime loader to work).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "model"


def _materialize_with_base_weights(weights_path: Path) -> Path:
    """Copy ``model/`` into a temp dir and drop in ``model.safetensors``.

    The mirror tree holds source files only; the actual base weights live
    in the Release. ``AutoModel.from_pretrained`` needs them co-located.
    """
    tmp = Path(tempfile.mkdtemp(prefix="jina-v5-nano-export-"))
    for src in SOURCE_DIR.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(SOURCE_DIR)
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(weights_path, tmp / "model.safetensors")
    # ``JinaEmbeddingsV5Model.from_pretrained`` auto-loads ALL task adapters,
    # not just the one we'll merge. Materialize all four so the load succeeds;
    # we'll merge_and_unload only the retrieval adapter afterwards.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from embed import materialize  # noqa: E402

    cache_dir = materialize(verbose=True)
    for task in ("retrieval", "text-matching", "clustering", "classification"):
        src_adapter = cache_dir / "adapters" / task / "adapter_model.safetensors"
        if not src_adapter.exists():
            raise SystemExit(f"missing {task} adapter weights at {src_adapter}")
        dst_adapter = tmp / "adapters" / task / "adapter_model.safetensors"
        dst_adapter.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_adapter, dst_adapter)
    return tmp


class _MergedEncoder(torch.nn.Module):
    """Wrap the merged EuroBERT base so it returns ``last_hidden_state`` directly.

    Exporting raw ``transformers``-style models with ``return_dict=True`` produces
    ONNX graphs whose outputs are model-specific dataclasses; coercing to a
    bare tensor at the boundary keeps the consumer side trivial.
    """

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def export(model_dir: Path, out_dir: Path, opset: int = 17) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export_onnx] loading base + retrieval adapter from {model_dir}")
    base = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32
    )
    # ``JinaEmbeddingsV5Model`` is itself a PeftMixedModel that auto-loads all
    # four task adapters in ``from_pretrained``. Activate retrieval and merge.
    base.set_adapter(["retrieval"])
    merged = base.merge_and_unload()
    merged.eval()
    print(f"[export_onnx] merge_and_unload complete; merged class: {type(merged).__name__}")

    wrapped = _MergedEncoder(merged).eval()

    # Dummy inputs — batch=2, seq=16 exercises both dynamic axes.
    dummy_ids = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    dummy_mask = torch.ones((2, 16), dtype=torch.long)

    onnx_path = out_dir / "model.onnx"
    print(f"[export_onnx] exporting → {onnx_path} (opset={opset})")
    # Force the legacy TorchScript exporter (``dynamo=False``). The new
    # torch.export-based path trips on data-dependent guards inside
    # ``scaled_dot_product_attention`` for transformers SDPA backends, and the
    # diagnostic suggests inserting torch._check calls into upstream code we
    # can't modify. Legacy export traces eagerly through SDPA without that
    # constraint and produces a working graph.
    # Switch SDPA backend to math/eager during trace for cleaner ops.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        torch.onnx.export(
            wrapped,
            (dummy_ids, dummy_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # Probe whether the exporter chose external data (large models trigger this
    # automatically; small ones don't).
    sidecar = out_dir / "model.onnx_data"
    if sidecar.exists():
        print(f"[export_onnx] external data file: {sidecar} ({sidecar.stat().st_size} bytes)")
    else:
        print(f"[export_onnx] single-file export: {onnx_path.stat().st_size} bytes")

    # Persist export metadata so the loader can sanity-check.
    (out_dir / "export_meta.json").write_text(
        json.dumps(
            {
                "upstream_sha": "8a7f00aac812071b69403df470f1038ec85f8925",
                "release_tag": "v5-nano-8a7f00aa",
                "adapter_merged": "retrieval",
                "opset": opset,
                "pooling": "none (encoder forward only)",
                "outputs": ["last_hidden_state"],
                "torch_version": torch.__version__,
            },
            indent=2,
        )
    )
    return onnx_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Path to model.safetensors (base weights). Defaults to the cached "
            "copy fetched by scripts/embed.py."
        ),
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "out", help="output dir")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        help="leave the temporary materialized model dir behind for inspection",
    )
    args = p.parse_args()

    if args.weights is None:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from embed import materialize  # noqa: E402

        cache_dir = materialize(verbose=True)
        args.weights = cache_dir / "model.safetensors"
        if not args.weights.exists():
            raise SystemExit(f"weights not found at {args.weights}")

    tmp_model_dir = _materialize_with_base_weights(args.weights)
    try:
        export(tmp_model_dir, args.out, opset=args.opset)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_model_dir, ignore_errors=True)
        else:
            print(f"[export_onnx] kept tmp dir: {tmp_model_dir}")
    print("[export_onnx] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
