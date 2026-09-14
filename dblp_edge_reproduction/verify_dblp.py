"""Load DBLP via PyTorch Geometric's CitationFull and verify its statistics.

Milestone 1 (Phases 2-9 of the plan, first checkpoint):

    Checkpoint A: DBLP loads
      x shape, edge_index shape verified against documented/paper stats.

Expected (documented in PyG CitationFull, README table):

    Nodes      ~17,716
    Edges      ~105,734  (undirected full edge list)
    Features   ~ 1,639
    Classes    ~ 4

These are reference values; we print the actual numbers loaded by the
installed dataset version.
"""

import os
import sys

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import is_undirected

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def main():
    print("=" * 70)
    print("Checkpoint A: load DBLP (CitationFull + NormalizeFeatures)")
    print("=" * 70)

    dataset = CitationFull(
        os.path.join(DATA_DIR, "DBLP"),
        "DBLP",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0]

    print("\n[Dataset]")
    print(f"  {dataset}")

    print("\n[Graph object]")
    print(f"  x            : {tuple(data.x.shape)}  dtype={data.x.dtype}")
    print(f"  edge_index   : {tuple(data.edge_index.shape)}  dtype={data.edge_index.dtype}")
    if hasattr(data, "y"):
        print(f"  y            : {tuple(data.y.shape)}  dtype={data.y.dtype}")
    print(f"  directed     : {not is_undirected(data.edge_index)}")

    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    num_features = data.x.shape[1]

    print("\n[Derived statistics]")
    print(f"  num_nodes    : {num_nodes:,}")
    print(f"  num_edges    : {num_edges:,}")
    print(f"  num_features : {num_features:,}")

    print("\n[Reference values]")
    ref_nodes = 17716
    ref_edges = 105734
    ref_features = 1639
    print(f"  reference nodes    : {ref_nodes:,}")
    print(f"  reference edges    : {ref_edges:,}")
    print(f"  reference features : {ref_features:,}")

    checks = [
        ("nodes", num_nodes, ref_nodes),
        ("edges", num_edges, ref_edges),
        ("features", num_features, ref_features),
    ]

    print("\n[Checkpoint A verification]")
    all_ok = True
    for name, actual, expected in checks:
        ok = actual == expected
        status = "PASS" if ok else "MISMATCH"
        print(f"  {name:<10} actual={actual:>9,}  expected={expected:>9,}  [{status}]")
        if not ok:
            all_ok = False

    print(f"\n[Result] Checkpoint A: {'PASSED' if all_ok else 'FAILED'}")

    # Sanity: feature means ~ small after NormalizeFeatures
    print("\n[Feature sanity]")
    print(f"  x.min={data.x.min().item():.4f}  x.max={data.x.max().item():.4f}")
    row_sums = data.x.sum(dim=-1)
    print(f"  row sum mean={row_sums.mean().item():.4f}  std={row_sums.std().item():.4f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())