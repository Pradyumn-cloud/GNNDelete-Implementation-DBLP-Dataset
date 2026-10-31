# Plan: Extending GNNDelete to 3DSSG (3D Semantic Scene Graphs)

## 1. Goal

We have reproduced GNNDelete (ICLR 2023) on DBLP with a GCN. The extension is to
apply the same graph-unlearning method to a new domain, **3DSSG**, a dataset of
semantic scene graphs of 3D indoor scans (computer graphics / 3D scene
understanding).

**Research question:** Can GNNDelete remove chosen relationships (or whole scenes)
from a GNN trained on 3D scene graphs while keeping link-prediction quality on the
remaining graph, and how does it compare with retraining from scratch and the
other baselines?

**Why this domain is a good extension**

- Scene graphs have many relationship types (support, spatial, attributes), so we
  need a relational GNN (R-GCN), which is the paper's heterogeneous setting
  (WordNet18 / BioKG) and goes beyond our single-type DBLP reproduction.
- Scans are separate connected components, so the "easy (OUT) vs hard (IN)"
  deletion settings from the paper have a natural meaning here.
- "Unlearn a whole scan" has a clear privacy motivation: a user asks for their
  scanned room to be removed from a trained model.

## 2. How GNNDelete works (the part that matters for this plan)

1. Train a base GNN **once** on the full graph.
2. When a deletion request arrives, **freeze** the base model and train only a
   small deletion operator `W_D` (an MLP applied after each GNN layer, active only
   inside the local subgraph of the deleted edges).
3. The loss is `λ * L_DEC + (1 - λ) * L_NI`:
   - **Deleted Edge Consistency (DEC):** deleted edges should look like random
     unconnected node pairs.
   - **Neighborhood Influence (NI):** representations in the local neighborhood
     should stay close to the original.
4. "Retrain from scratch" is only a **baseline** (trained on the graph without the
   deleted edges); it is not part of our method.

So the base model is trained once on 3DSSG, and each deletion experiment trains
only `W_D`.

## 3. The dataset

Provided files (`3DSSG.zip`): `objects.json`, `relationships.json`,
`classes.txt`, `attributes.txt`, `relationships.txt`, `affordances.txt`,
`wordnet_attributes.txt`. There are **no point clouds or images**, only
annotations, which is all we need for graph learning.

Measured on the provided files:

| Item | Value |
|---|---|
| Scans (scenes) | 1,335 |
| Object nodes | 43,450 (about 32.5 per scene) |
| Relationship edges (as stored) | 543,956 |
| Relationship types | 40+ (see `relationships.txt`) |

Most frequent types: `same object type` (107,084), `same shape` (85,218),
`same color` (66,940), `left` / `right` (45,357 each), `close by` (26,500),
`behind` / `front` (24,165 each), `same material` (20,828), `attached to`
(12,761).

Each relationship is stored as `[object_id_1, object_id_2, relation_id, name]`
inside its scan. Each object has a label, attributes (color, shape, state, ...),
and affordances.

### Dataset issues we must handle

1. **Inverse and symmetric duplicates cause leakage.** `left`/`right` and
   `front`/`behind` are stored as inverse pairs, and the `same ...` relations are
   symmetric. If A→B is in training and B→A is in test, the model gets the answer
   for free. We keep one edge per pair before splitting.
2. **Object ids repeat across scans.** We build a global node index from
   (scan, object id).
3. **Trivial edges inflate scores.** The `same ...` relations dominate the edge
   count and are easy to predict. The main experiment uses spatial and support
   relations only; "all relations" is a secondary experiment.

## 4. Graph construction

- **Nodes:** all objects across all scans, indexed globally.
- **Node features (`x`):** object label (one-hot or learned embedding) plus
  multi-hot attributes (color, shape, state, ...), producing a fixed-width float
  matrix.
- **Edges (`edge_index`):** relationships after deduplication of inverse and
  symmetric pairs.
- **Edge types (`edge_type`):** the relationship id, stored for the R-GCN stage.
- **Structure:** one large graph made of 1,335 disconnected components.
- **Task:** link prediction (does a relationship exist between two objects).

Edge sets for experiments:

| Set | Relations | Use |
|---|---|---|
| `spatial_support` (main) | supported by, standing on, attached to, left/right, front/behind, close by, inside, ... | Main results |
| `all` (secondary) | everything, including `same ...` | Check how much trivial edges change the picture |

## 5. Pipeline

### Stage A: reuse the existing pipeline (ignore edge types)

Goal: get an end-to-end result quickly with the current GCN code and confirm the
data and splits are sound.

1. Write `prepare_3dssg.py` and `verify_3dssg.py`.
2. Add a `--dataset` argument to `train_original.py` / `training_args.py`.
3. Train the base GCN on 3DSSG and check test AUROC.
4. Run one deletion experiment (2.5%, OUT): Retrain vs GNNDelete.

**Success check:** Retrain keeps high Et and has Ed near chance; GNNDelete keeps
Et close to Retrain and moves Ed towards the Retrain value or beyond. If the
base model's test AUROC is weak, fix that before unlearning.

### Stage B: relational model (R-GCN)

1. Add `models/rgcn.py` using `RGCNConv(in, out, num_relations)` with the same
   forward interface as `gcn.py`.
2. Pass `edge_type` through the trainers and, if needed, use a relation-aware
   decoder (for example DistMult).
3. Keep `edge_type` attached to edges in `delete_edges.py` when sampling and
   removing edges.
4. Confirm the deletion operator in `models/deletion.py` is applied to each
   R-GCN layer's output (it acts on node representations, so it should need
   little or no change).

### Stage C: full experiments

Run the experiment grid below, with 5 seeds for the final tables.

## 6. Repository changes (file by file)

| File | Change |
|---|---|
| `prepare_3dssg.py` (new) | Parse JSON, dedupe inverse/symmetric edges, build global ids, features, `edge_index`, `edge_type`; save in the same format as `prepare_dblp.py` |
| `verify_3dssg.py` (new) | Sanity checks: counts, no self-loops, no duplicates, no isolated nodes, no train/test leakage |
| `training_args.py` | Dataset choice, feature dimension, number of relations, model choice |
| `train_original.py` | Dataset switch, model switch (GCN / R-GCN) |
| `models/rgcn.py` (new) | R-GCN backbone |
| `models/deletion.py` | Check only; expected to work as is |
| `trainer/base.py`, `retrain.py`, `gnndelete.py` | Pass `edge_type` through forward and decoder |
| `delete_edges.py` | Sample Ed,OUT / Ed,IN with edge types; add whole-scene deletion |
| `evaluate.py` / `evaluation.py` | Same AUROC logic; add per-relation-type results (optional) and membership inference (MI) ratio |
| `README.md` | Add the 3DSSG section; fix the paper title (see below) |

> README fix: the paper is titled "GNNDelete: A General Strategy for Unlearning
> in **Graph Neural Networks**", not "...in Dynamic Graphs".

## 7. Experiments

| # | Experiment | Setting | Methods |
|---|---|---|---|
| E1 | Edge deletion, easy | Ed,OUT at 0.5 / 2.5 / 5.0% | Retrain, GNNDelete (+ baselines) |
| E2 | Edge deletion, hard | Ed,IN at 0.5 / 2.5 / 5.0% | same |
| E3 | Whole-scene deletion | Delete all edges of N scenes (for example 10, 50) | same |
| E4 | Node deletion | Delete about 100 objects and their edges | same, plus node-level check |
| E5 | λ ablation | λ in {0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} at 2.5% | GNNDelete |
| E6 | Efficiency | Wall-clock time and trainable parameters | all methods |
| E7 | Edge-set comparison | `spatial_support` vs `all` | Retrain, GNNDelete |

Baselines (as in the paper and our repo): Retrain-from-scratch, GradAscent,
Descent-to-Delete, and (if feasible) GraphEraser. GraphEditor and CertUnlearn are
linear-GNN methods and will likely be N/A for the relational setting.

### Definitions

- **Et:** held-out test edges that were not deleted. Higher is better; this is
  the utility we want to keep.
- **Ed:** the deleted edges. Following the paper, a score closer to the Retrain
  value, and not a very high score on deleted edges, indicates they were
  forgotten. Report it the same way as in our DBLP tables so the two
  reproductions are comparable.
- **IN vs OUT:** Ed,IN is sampled inside the 2-hop subgraph around the test
  edges (hard); Ed,OUT is sampled outside it (easy).

## 8. Evaluation metrics

- AUROC (and AUPRC) on Et and on Ed for link prediction.
- Membership Inference ratio for node and scene deletion.
- Wall-clock time and number of trainable parameters.
- Mean ± std over 5 seeds for final tables (single seed is fine while
  developing).

## 9. Deliverables

1. Working code in the repo (`prepare_3dssg.py`, `rgcn.py`, updated trainers).
2. Results tables for E1 to E7 in the same format as the DBLP tables in the
   README.
3. Figures: λ ablation curve, time vs deletion ratio, AUROC comparison bars.
4. A short comparison: DBLP (single relation, GCN) vs 3DSSG (many relations,
   R-GCN).
5. Updated README and a short report or slides.
6. (Optional) A small Streamlit demo: pick a scene, delete an edge, and show the
   graph and predicted edge probability before and after unlearning.

## 10. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Label leakage from inverse/symmetric edges | Dedupe before splitting; check in `verify_3dssg.py` |
| Trivial `same ...` edges inflate AUROC | Main experiment uses `spatial_support`; report `all` separately |
| Base model is weak on 3DSSG | Tune features, hidden size and decoder before unlearning |
| DEL operator hurts Et too much (we saw this in DBLP at higher ratios) | λ ablation; try last-layer-only DEL (paper Table 6) |
| R-GCN support missing in the trainers | Do Stage A first so there is always a working result |
| Single-seed noise | Use 5 seeds for final numbers |

## 11. Suggested order of work

1. Parse the data, build the graph, write `verify_3dssg.py` (Stage A).
2. Train the base GCN, then run E1 at 2.5% only (Stage A check).
3. Add the R-GCN and edge types (Stage B), then rerun the 2.5% check.
4. Run E1 and E2 fully.
5. Run E3 and E4 (scene and node deletion).
6. Run E5, E6 and E7, then make the figures.
7. Write the README, report, and optional demo.

## 12. Open questions to settle early

- Which relations count as `spatial_support` for the main run.
- Whether to use a plain dot-product decoder or DistMult for link prediction.
- Whether the split is random by edge (default, matches the paper) or by scene.
- How many seeds we have time for.
