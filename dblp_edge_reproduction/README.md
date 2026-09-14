# GNNDelete — DBLP Edge Deletion Reproduction

A DBLP-only reproduction of the official GNNDelete (ICLR 2023) edge-deletion
experiment with a GCN backbone and link-prediction downstream task.

The master plan lives at
[`../GNNDelete_DBLP_Edge_Deletion_PLAN.md`](../GNNDelete_DBLP_Edge_Deletion_PLAN.md).
All compatibility changes are recorded in [`MODERNIZATION.md`](MODERNIZATION.md).

## Scope

```text
Dataset:       DBLP          (CitationFull, NormalizeFeatures)
GNN:           2-layer GCN   (official framework/models/gcn.py)
Unlearning:    edge deletion
Downstream:    link prediction
Method:        GNNDelete
```

No node deletion, node-feature deletion, other datasets, or other backbones
in this phase.

## Folder layout

```text
dblp_edge_reproduction/
├── data/                  # PyG CitationFull DBLP cache (auto-downloaded)
├── checkpoints/
│   ├── original/          # trained original GCN
│   ├── gnndelete/         # GNNDelete unlearned models
│   └── retrain/           # retrained-from-scratch reference GCN
├── models/                # official GCN + deletion layers (reused copies)
├── trainer/               # official GNNDelete trainer (reused)
├── environment_audit.py   # Milestone 1: environment verification
├── verify_dblp.py         # Milestone 1: Checkpoint A (DBLP load + stats)
├── prepare_dblp.py        # Milestone 2 (TODO): prepare graph splits
├── train_original.py      # Milestone 2 (TODO): train original GCN
├── delete_edges.py        # Milestone 3 (TODO): run GNNDelete unlearning
├── evaluate.py            # Milestone 3 (TODO): Et/Ed AUROC + retrain compare
├── MODERNIZATION.md       # record of every compatibility change
└── README.md
```

## Current status

### Milestone 1 — DONE

1. Environment audited (`python environment_audit.py`).
2. Missing packages installed: `torch-geometric==2.8.0.post1`,
   `ogb==1.3.6`, `wandb==0.30.0`.
3. DBLP loaded and verified (`python verify_dblp.py`, Checkpoint A PASS):

   ```
   nodes      actual=   17,716   expected=   17,716  [PASS]
   edges      actual=  105,734   expected=  105,734  [PASS]
   features   actual=    1,639   expected=    1,639  [PASS]
   ```

### Milestone 2 — DONE

1. `python prepare_dblp.py` — official splits + deletion masks for all 5 seeds
   (train 47,581 / val 2,643 / test 2,643; `data/DBLP/d_<seed>.pkl`,
   `data/DBLP/df_<seed>.pt`).
2. `python train_original.py --lr 1e-3 --dataset DBLP --random_seed 42
   --unlearning_model original --gnn gcn` — Checkpoint B PASS:

   ```
   test loss   : 0.6566
   test dt_auc : 0.9701
   test dt_aup : 0.9731
   ```

   Checkpoints in `checkpoints/DBLP/gcn/original/42/`.

Next: Milestone 3 — select `E_d` (10-edge smoke test, easy/Out), compute
official 1/2-hop masks, build `GCNDelete`, run DEC+NI training, evaluate
Et/Ed AUROC.