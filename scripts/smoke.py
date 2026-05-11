"""End-to-end smoke test.

Embeds a query and a relevant document via the asymmetric retrieval prompts
and verifies cosine similarity exceeds a sanity threshold. Exits nonzero if
the embedder is producing garbage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import DIM, embed  # noqa: E402


def main() -> int:
    query = "What are the symptoms of vitamin D deficiency?"
    relevant = (
        "Vitamin D deficiency commonly presents with fatigue, bone pain, "
        "muscle weakness, mood changes, and increased susceptibility to "
        "respiratory infections."
    )
    unrelated = (
        "The Linux kernel scheduler uses Completely Fair Scheduler (CFS) "
        "as its default for normal tasks, organizing them in a red-black tree."
    )

    q = embed([query], task="retrieval", prompt_name="query", max_length=512)
    d_rel, d_unrel = embed(
        [relevant, unrelated], task="retrieval", prompt_name="document",
        max_length=512,
    )

    assert q.shape == (1, DIM), f"unexpected query shape {q.shape}"
    assert d_rel.shape == (DIM,), f"unexpected doc shape {d_rel.shape}"
    np.testing.assert_allclose(np.linalg.norm(q[0]), 1.0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(d_rel), 1.0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(d_unrel), 1.0, atol=1e-3)

    sim_rel = float(q[0] @ d_rel)
    sim_unrel = float(q[0] @ d_unrel)

    print(f"query:     {query!r}")
    print(f"relevant:  cos={sim_rel:.4f}")
    print(f"unrelated: cos={sim_unrel:.4f}")
    print(f"margin:    {sim_rel - sim_unrel:+.4f}")

    if sim_rel <= 0.3:
        print(f"FAIL: relevant cosine {sim_rel:.4f} <= 0.3 threshold", file=sys.stderr)
        return 1
    if sim_rel <= sim_unrel:
        print("FAIL: relevant doc did not outscore unrelated doc", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
