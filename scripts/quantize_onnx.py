"""Build quantized variants of the retrieval ONNX export (model.onnx).

    python scripts/quantize_onnx.py model.onnx model.q4.onnx --bits 4
    python scripts/quantize_onnx.py model.onnx model.int8.onnx --int8

Derived assets — NOT upstream-identical. The fp32 model.onnx is produced by
export_onnx.py (retrieval adapter merged); this script quantizes its weights.

q4 (recommended): MatMulNBits 4-bit blockwise (block_size=32, symmetric) on the
linear weights, then int8 dynamic to mop up the leftover graph. The mop-up is
load-bearing: MatMulNBits leaves EuroBERT's large embedding `Gather` (~400 MB
fp32) untouched, so 4-bit-alone is ~465 MB > int8's 212 MB; with the mop-up it's
~170 MB. q4 is retrieval-identical to fp32 (per-doc cosine 0.975+) and domain-
robust — see PERFORMANCE.md.

int8: plain dynamic QUInt8. ~212 MB, ~2.2x faster CPU decode, but per-tensor int8
is DOMAIN-FRAGILE (0.445 cosine to fp32 on out-of-domain text). Use only for
high-throughput indexing of in-domain corpora.

Requires: onnx, onnxruntime, onnx_ir (for MatMulNBitsQuantizer). 3-bit is
unsupported (ORT asserts bits in {2,4,8}); 2-bit runs but halves recall.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_int8(src: str | Path, dst: str | Path) -> Path:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return Path(dst)


def build_nbits(src: str | Path, dst: str | Path, *, bits: int = 4,
                block_size: int = 32) -> Path:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    src, dst = Path(src), Path(dst)
    tmp = dst.with_suffix(".matmul.onnx")
    q = MatMulNBitsQuantizer(onnx.load(str(src)), bits=bits,
                             block_size=block_size, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QUInt8)  # embedding mop-up
    tmp.unlink(missing_ok=True)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source fp32 ONNX (model.onnx)")
    ap.add_argument("dst", help="destination quantized ONNX")
    ap.add_argument("--int8", action="store_true", help="plain int8 dynamic (else N-bit blockwise)")
    ap.add_argument("--bits", type=int, default=4, help="MatMulNBits width (2 or 4; default 4)")
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args(argv)
    out = build_int8(args.src, args.dst) if args.int8 \
        else build_nbits(args.src, args.dst, bits=args.bits, block_size=args.block_size)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
