# Guide for running the 3DSSG extension of GNNDelete

You are going to train and test the 3DSSG part of our GNNDelete project. This file explains
what is in this folder, how to run it, and what to send back.
**Status update:** the full pipeline has now been run successfully (Windows, RTX 4050,
PyTorch 2.6 + PyG 2.8). The smoke test, base models, and E1–E5 are complete; results are in
`results/` and `results_summary.md`. This guide is kept as the original runbook.

---------------------------------------------------------------------------------------------

## 1. What this project does (in short)

- GNNDelete (ICLR 2023) removes the influence of chosen edges from an already-trained graph
  neural network, without retraining from scratch.
- We already reproduced it on the DBLP citation graph (folder `dblp_edge_reproduction/`).
- This folder applies it to **3DSSG**: every 3D indoor scan is a small graph. Nodes are objects
  (chair, table, wall...) with labels and attributes (color, material, size...). Edges are
  relationships between objects ("standing on", "left", "attached to"...).
- Task: predict the relationship between two objects (link prediction). Unlearning: forget
  some relationships, whole scenes, or whole objects, and check that
  (a) the model forgot them, and (b) it still works on everything else.

## 2. What is in this folder

| File | Purpose |
|---|---|
| `prepare_3dssg.py` | Turns `raw/objects.json` + `raw/relationships.json` into a graph and train/val/test splits. Needs only numpy. |
| `verify_3dssg.py` | Sanity checks on the prepared data. Every line must say PASS. |
| `df_utils.py` | Chooses which edges/scenes/objects to delete (Df) and the neighbourhood around them. |
| `common.py` | Shared code: data loading, negative sampling, evaluation, training loop. |
| `models.py` | GCN and R-GCN models + the GNNDelete deletion layer. |
| `train_original.py` | Trains the base model once (the model we later "unlearn" from). |
| `delete_edges.py` | Runs the unlearning: `gnndelete`, `retrain`, `gradient_ascent`, or `original` (no unlearning, reference). |
| `run_experiments.py` | Runs the whole experiment grid with one command. Skips runs that already finished. |
| `collect_results.py` | Builds the results tables (`results_summary.md`). |
| `requirements.txt` | Python packages. |
| `README.md` | Short version of this guide. |

Folders created while you run things (do NOT commit `raw/`, `data/`, `checkpoints/`):
`raw/` (dataset you put there), `data/` (prepared graph), `checkpoints/` (saved models),
`results/` (one small JSON per run: **this is what we want back**).

## 3. Setup and smoke test (do this first)

    # from the repo root
    cd 3dssg
    pip install -r requirements.txt

- If `torch_geometric` fails to install, follow https://pytorch-geometric.readthedocs.io (plain
  `pip install torch_geometric` is enough, no extra CUDA packages are needed).
- A GPU is optional. The code uses CUDA automatically if available; otherwise CPU.

Put the dataset files here: `3dssg/raw/objects.json` and `3dssg/raw/relationships.json`
(from the 3DSSG.zip we have). Then:

    python prepare_3dssg.py
    python verify_3dssg.py --seeds 42 21 13 87 100

Expected: `ALL CHECKS PASSED`. Sanity numbers for the default edge set: about 33,796 nodes,
126,532 edges, 18 relation types; for seed 42: train 114,033 / val 6,252 / test 6,247 edges.
If your numbers are very different, stop and tell us.

Smoke test (tiny, a few minutes; results are meaningless, it only checks that nothing crashes):

    python run_experiments.py --quick

It trains a tiny R-GCN, then runs Retrain, GNNDelete and Gradient-Ascent once, and prints a table.
If this finishes, the pipeline works.

## 4. The real runs

Training length defaults: base model 1000 epochs, Retrain 1000, GNNDelete 500, Gradient-Ascent 10.
The speed on CPU is unknown. If one epoch is very slow, use a GPU (Colab is fine) or lower
`--epochs` and tell us what you used.

### Easiest: the experiment grid

    python run_experiments.py --stage base --gnn gcn  --seeds 42      # Stage A: type-agnostic GCN
    python run_experiments.py --stage all  --gnn rgcn --seeds 42      # Stage B: R-GCN, whole grid
    python collect_results.py

`--stage` options: `base` (train base only), `e1` (delete random edges far from test edges, 0.5/2.5/5 %),
`e2` (edges near test edges, hard), `e3` (forget 10 and 50 whole scenes), `e4` (forget 100 objects),
`e5` (lambda ablation), `all` (everything). Add more seeds later: `--seeds 42 21 13 87 100`.
The script is resumable: if it stops, run the same command again.

### Manual commands (if you want to control each step)

    # 1) base model (must come first; same --gnn/--seed/--edge_set as step 2)
    python train_original.py --gnn rgcn --seed 42

    # 2) unlearn. Methods: gnndelete | retrain | gradient_ascent | original
    python delete_edges.py --unlearning_model gnndelete --gnn rgcn --seed 42 --df out --df_size 2.5
    python delete_edges.py --unlearning_model retrain   --gnn rgcn --seed 42 --df out --df_size 2.5

`--df` = what to delete: `out` (random edges far from test edges), `in` (near test edges, harder),
`scene` (whole scenes), `node` (objects and all their edges).
`--df_size` = for out/in a percent of training edges (2.5 means 2.5 %; a value of 100 or more means
an exact number of edges); for scene/node a count (10 scenes, 100 objects).
Other useful flags: `--alpha 0.5` (balance of the two GNNDelete losses), `--epochs`, `--gnn gcn|rgcn`,
`--edge_set spatial_support|all|no_same`.

## 5. How to read the results

`python collect_results.py` writes `results_summary.md`. Per setting you get:

- **Et AUROC (higher is better)**: link prediction on test edges that were not deleted. Shows the
  model still works. Compare with the base model's number and with Retrain.
- **Ed AUROC**: deleted edges (label 0) against an equal number of remaining edges (label 1).
  Retrain is the reference for "the model never saw them". Values near or above Retrain mean the
  deleted edges are no longer recognised as real.
- **MI ratio**: a simplified membership-inference proxy (NOT the paper's exact attacker). Above 1
  means less information about the deleted edges remains.
- **time / trainable params**: GNNDelete should be much faster and train far fewer parameters than Retrain.

The `per_relation_et` field inside each JSON shows AUROC per relation type.

## 6. Things to know (so you are not surprised)

- Negatives are sampled inside the same scene. Pairs of objects from different scenes would be
  trivially negative and inflate the scores.
- Val/test edges come only from 30 % of the scenes. This is on purpose, so that the "far from test"
  (`out`) deletion set is large enough.
- Always use the same `--edge_set`, `--gnn` and `--seed` for `train_original.py` and `delete_edges.py`.
  Otherwise you get "Base model not found".
- Do not edit the code to make a number look better. If something looks wrong, report it.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `split_42.npz not found` | Run `prepare_3dssg.py` (it builds seeds 42 21 13 87 100 by default). |
| `Base model not found` | Run `train_original.py` with the same `--gnn/--seed/--edge_set` first. |
| `ModuleNotFoundError: torch_geometric` | `pip install torch_geometric` |
| CUDA out of memory | Run on CPU: `CUDA_VISIBLE_DEVICES="" python ...` (Windows: set the variable to empty first), or use `--hidden_dim 64`. |
| R-GCN is very slow | Use a GPU, or lower `--epochs`, or `--hidden_dim 64`. Report the time per epoch. |
| `MI skipped: ...` | Harmless; the other metrics are still saved. |

## 8. What to send back

1. The whole `3dssg/results/` folder (small JSON files) and `results_summary.md`.
2. The exact commands you ran, the number of epochs, GPU/CPU, and total time.
3. For any crash: the complete terminal output (copy-paste, not a screenshot if possible).

To share results through git, commit only `3dssg/results/`. Never commit `raw/`, `data/` or `checkpoints/`.
