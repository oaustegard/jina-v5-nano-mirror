"""ONNX path parity + retrieval smoke test.

Runs both the torch reference loader (``embed.py``) and the ONNX loader
(``embed_onnx.py``) over a fixed set of (query, relevant_doc, unrelated_doc)
triples and asserts:

  1. **Numerical parity**: mean cosine similarity between torch and ONNX
     embeddings of the same input > 0.99 (fp32 numerics will differ slightly).
  2. **Retrieval correctness on the ONNX path**: for every triple,
     ``cos(query, relevant) - cos(query, unrelated) > 0.3``.
  3. **Matryoshka**: dims 768 / 512 / 256 are all unit-norm and rank
     relevant > unrelated.

Exit 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


TRIPLES = [
    (
        "What are the symptoms of vitamin D deficiency?",
        "Vitamin D deficiency commonly presents with fatigue, bone pain, "
        "muscle weakness, mood changes, and increased susceptibility to "
        "respiratory infections.",
        "The Linux kernel scheduler uses Completely Fair Scheduler (CFS) "
        "as its default for normal tasks, organizing them in a red-black tree.",
    ),
    (
        "How does the Fed set interest rates?",
        "The Federal Reserve's Open Market Committee meets eight times a year "
        "to set the federal funds rate target, the rate at which banks lend "
        "reserves to each other overnight.",
        "Sourdough bread relies on a starter culture of wild yeast and "
        "lactobacilli to leaven the dough through long fermentation.",
    ),
    (
        "What is the half-life of caffeine?",
        "Caffeine has a half-life of roughly 5 hours in healthy adults, "
        "though it varies with age, liver function, and CYP1A2 genotype.",
        "Volcanic eruptions on Io are powered by tidal heating from "
        "gravitational interaction with Jupiter and the other Galilean moons.",
    ),
    (
        "Why did the Roman Republic fall?",
        "The Roman Republic collapsed under accumulating civil wars, "
        "concentration of military power in individual generals like Marius, "
        "Sulla, and Caesar, and the breakdown of senatorial authority.",
        "Atomic clocks based on cesium-133 transitions define the SI second "
        "with a precision of one part in 10^15.",
    ),
    (
        "What's a closure in JavaScript?",
        "In JavaScript, a closure is a function bundled with its lexical "
        "environment, allowing it to access variables from an enclosing scope "
        "even after that scope has returned.",
        "The migration patterns of monarch butterflies span thousands of "
        "kilometers across multiple generations between Mexico and Canada.",
    ),
    (
        "How do mRNA vaccines work?",
        "mRNA vaccines deliver a strand of messenger RNA encoding a viral "
        "antigen; ribosomes translate it into protein, and the immune system "
        "learns to recognize that protein as a future threat signal.",
        "Cricket bowling actions are governed by a 15-degree elbow extension "
        "limit enforced via biomechanical testing.",
    ),
]


def _cos_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def _check_unit_norm(arr: np.ndarray, label: str, atol: float = 1e-3) -> None:
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise AssertionError(
            f"{label}: not unit-norm (min={norms.min():.4f}, max={norms.max():.4f})"
        )


def main() -> int:
    queries = [t[0] for t in TRIPLES]
    relevants = [t[1] for t in TRIPLES]
    unrelateds = [t[2] for t in TRIPLES]

    print("[smoke_onnx] loading ONNX path…", file=sys.stderr)
    from embed_onnx import embed as embed_onnx

    onnx_q = embed_onnx(queries, prompt_name="query", max_length=512)
    onnx_rel = embed_onnx(relevants, prompt_name="document", max_length=512)
    onnx_unrel = embed_onnx(unrelateds, prompt_name="document", max_length=512)
    _check_unit_norm(onnx_q, "onnx_q")
    _check_unit_norm(onnx_rel, "onnx_rel")
    _check_unit_norm(onnx_unrel, "onnx_unrel")

    # (1) Retrieval correctness on the ONNX path.
    margins = []
    for i in range(len(TRIPLES)):
        sim_rel = float(onnx_q[i] @ onnx_rel[i])
        sim_unrel = float(onnx_q[i] @ onnx_unrel[i])
        margin = sim_rel - sim_unrel
        margins.append((sim_rel, sim_unrel, margin))
        print(
            f"  triple {i}: rel={sim_rel:+.4f} unrel={sim_unrel:+.4f} "
            f"margin={margin:+.4f}"
        )
        if margin <= 0.3:
            print(
                f"FAIL: triple {i} margin {margin:.4f} <= 0.3 threshold",
                file=sys.stderr,
            )
            return 1

    # (2) Matryoshka: dims 512 / 256 still rank correctly + are unit-norm.
    for dim in (512, 256):
        q = embed_onnx(queries, prompt_name="query", max_length=512, dim=dim)
        rel = embed_onnx(relevants, prompt_name="document", max_length=512, dim=dim)
        unrel = embed_onnx(
            unrelateds, prompt_name="document", max_length=512, dim=dim
        )
        _check_unit_norm(q, f"matryoshka dim={dim} q")
        for i in range(len(TRIPLES)):
            sim_rel = float(q[i] @ rel[i])
            sim_unrel = float(q[i] @ unrel[i])
            if sim_rel <= sim_unrel:
                print(
                    f"FAIL: matryoshka dim={dim} triple {i}: "
                    f"rel {sim_rel:.4f} <= unrel {sim_unrel:.4f}",
                    file=sys.stderr,
                )
                return 1
        print(f"[smoke_onnx] matryoshka dim={dim} OK")

    # (3) Numerical parity vs torch path.
    print("[smoke_onnx] loading torch path for parity check…", file=sys.stderr)
    from embed import embed as embed_torch

    torch_q = embed_torch(queries, task="retrieval", prompt_name="query",
                          max_length=512)
    torch_rel = embed_torch(relevants, task="retrieval", prompt_name="document",
                            max_length=512)
    torch_unrel = embed_torch(unrelateds, task="retrieval",
                              prompt_name="document", max_length=512)

    parities = []
    for label, t_arr, o_arr in (
        ("query", torch_q, onnx_q),
        ("relevant", torch_rel, onnx_rel),
        ("unrelated", torch_unrel, onnx_unrel),
    ):
        diag = np.einsum("ij,ij->i", t_arr, o_arr)
        mean_cos = float(diag.mean())
        min_cos = float(diag.min())
        parities.append((label, mean_cos, min_cos))
        print(
            f"  parity {label}: mean cos={mean_cos:.4f} min cos={min_cos:.4f}"
        )

    overall_mean = float(np.mean([p[1] for p in parities]))
    print(f"[smoke_onnx] overall parity mean cos = {overall_mean:.4f}")
    if overall_mean <= 0.99:
        print(
            f"FAIL: parity mean {overall_mean:.4f} <= 0.99 threshold",
            file=sys.stderr,
        )
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
