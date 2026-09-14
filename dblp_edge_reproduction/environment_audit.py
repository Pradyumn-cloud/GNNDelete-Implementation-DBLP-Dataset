"""Environment audit for the GNNDelete DBLP reproduction.

Milestone 1 (Phase 1 of the plan): verify that Python, PyTorch, PyTorch
Geometric, and the required libraries are present and importable before
loading DBLP.
"""

import sys
import platform
import importlib

PACKAGES = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "numpy",
    "scipy",
    "networkx",
    "sklearn",
    "pandas",
    "tqdm",
    "ogb",
    "wandb",
]

REQUIRED_PYG_APIS = [
    ("torch_geometric.datasets", "CitationFull"),
    ("torch_geometric.transforms", "NormalizeFeatures"),
    ("torch_geometric.data", "Data"),
    ("torch_geometric.utils", "train_test_split_edges"),
    ("torch_geometric.utils", "negative_sampling"),
    ("torch_geometric.utils", "k_hop_subgraph"),
    ("torch_geometric.utils", "to_undirected"),
    ("torch_geometric.utils", "is_undirected"),
    ("torch_geometric.utils", "to_networkx"),
    ("torch_geometric.seed", "seed_everything"),
    ("torch_geometric.loader", "GraphSAINTRandomWalkSampler"),
    ("torch_geometric.nn", "GCNConv"),
]


def check_import(module_name):
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_pyg_api(module_name, attr_name):
    try:
        mod = importlib.import_module(module_name)
        attr = getattr(mod, attr_name, None)
        if attr is None:
            return False, "attribute not found"
        return True, "ok"
    except ImportError as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("GNNDelete DBLP Reproduction - Environment Audit (Milestone 1)")
    print("=" * 70)

    print("\n[System]")
    print(f"  Python          : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  Platform        : {platform.platform()}")
    print(f"  Machine         : {platform.machine()}")

    print("\n[Packages]")
    ok_all = True
    for name in PACKAGES:
        ok, version = check_import(name)
        status = "OK " if ok else "MISSING"
        print(f"  {name:<18} {status} {version}")
        if not ok:
            ok_all = False

    print("\n[PyTorch / CUDA]")
    try:
        import torch

        print(f"  torch version         : {torch.__version__}")
        print(f"  torch CUDA build      : {torch.version.cuda}")
        print(f"  CUDA available        : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device name      : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA device count     : {torch.cuda.device_count()}")
    except ImportError:
        print("  torch not installed!")

    print("\n[PyTorch Geometric API audit]")
    for module_name, attr in REQUIRED_PYG_APIS:
        ok, msg = check_pyg_api(module_name, attr)
        status = "OK " if ok else "BROKEN"
        print(f"  {module_name}.{attr:<40} {status} {msg}")
        if not ok:
            ok_all = False

    print("\n[Result]")
    if ok_all:
        print("  Environment is ready for DBLP loading.")
    else:
        print("  Environment is NOT ready. Fix missing/broken items above.")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())