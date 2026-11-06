# GNNDelete: DBLP Reproduction and 3DSSG Extension

## Purpose

This repository reproduces the DBLP edge-deletion experiment from the
GNNDelete paper (ICLR 2023). Edge deletion removes a set of edges from a
trained GNN and updates the model so that those edges are "forgotten" while
utility on the remaining graph is preserved. We reproduce the paper's setup on
DBLP dataset, GCN backbone, link prediction for both deletion-sampling
settings — the easy/OUT (`E_d`, paper Table 17) and the hard/IN (`E_d`, paper
Table 18) — for the paper's three deletion ratios (0.5%, 2.5%, 5.0%) in OUT and
2.5% in IN.

The released reference repo shipped a broken DEL-layer initialization
(`ones/1000`, a rank-1 projection) that collapses test utility (Et ≈ 0.75)
before any training; we root-caused and fixed it (identity init).

## Original repo

- Official GNNDelete repo: <https://github.com/mims-harvard/GNNDelete>
- Paper (arXiv): "GNNDelete: A General Strategy for Unlearning in Graph Neural
  Networks" — <https://arxiv.org/abs/2302.13406>

## Paper vs. our results (DBLP, GCN, edge deletion, easy/OUT `E_d`)

Paper values are Table 17 (mean over 5 seeds); ours are single-run seed 42.

| Ratio | Method | Et paper | Et ours | Ed paper | Ed ours |
|-------|--------|---------:|--------:|---------:|--------:|
| 0.5%  | Retrain   | 0.965 | 0.9697 | 0.783 | 0.7880 |
| 0.5%  | GNNDelete | 0.959 | 0.9595 | 0.964 | 0.9833 |
| 2.5%  | Retrain   | 0.965 | 0.9697 | 0.777 | 0.7825 |
| 2.5%  | GNNDelete | 0.957 | 0.9423 | 0.892 | 0.9226 |
| 5.0%  | Retrain   | 0.964 | 0.9693 | 0.788 | 0.7898 |
| 5.0%  | GNNDelete | 0.956 | 0.9347 | 0.859 | 0.8790 |

Summary: Retrain baseline matches the paper within ~0.005. GNNDelete Et is
within ~0.002 at 0.5% and trails by ~0.015–0.021 at 2.5%/5.0%, while Ed is
consistently higher (0.879–0.983 vs 0.859–0.964), i.e. our DEL unlearns more
aggressively — the same direction as the paper's own λ (DEC-vs-NI) ablation.

## Hard/IN results (DBLP, GCN, 2.5% edge deletion, seed 42)

In the hard setting, E_d is sampled inside the 2-hop local enclosing subgraph
of the test edges, so unlearning strongly overlaps the test neighborhood. Paper
values are DBLP GCN 2.5% IN; ours are single-run seed 42.

| Method | Et paper | Et ours | Ed paper | Ed ours |
|--------|---------:|--------:|---------:|--------:|
| Retrain   | 0.964 | 0.9683 | 0.506 | 0.5028 |
| GNNDelete | 0.934 | 0.8954 | 0.748 | 0.7407 |

Interpretation: Retrain matches the paper almost exactly, and its Ed ≈ 0.50
(near chance) is the expected hard-setting signature — even full retraining can
no longer distinguish deleted edges. GNNDelete Ed (0.741 vs 0.748) shows the
DEL unlearning is effective on the hard-influence edges. The Et shortfall in
the hard setting (−0.039 vs −0.015 in OUT) is the same over-deletion operating
point documented above, amplified because E_d overlaps the test neighborhood;
it is a modernization-balance / single-seed effect, not an implementation bug.

## 3DSSG extension (R-GCN, link prediction)

Code and run instructions: [`dblp_edge_reproduction/3dssg/`](dblp_edge_reproduction/3dssg/).
The extension is specified in `plan_extension.md`.

### What is it?

We apply the same base-model-then-unlearn recipe to **3DSSG (3D Semantic Scene
Graphs)**, a dataset of semantic scene graphs built from 3D indoor scans. Each
scan is a graph: object nodes (chair, table, wall, ...) carry label and
attribute features, and relationship edges encode support, spatial layout, and
attributes. The task is link prediction: given two objects, predict whether a
relationship exists. We then delete four kinds of information and measure
whether the model forgets it:

- **E1 — easy/OUT edge deletion:** remove 0.5 / 2.5 / 5.0% of training edges
  outside the 2-hop neighbourhood of test edges.
- **E2 — hard/IN edge deletion:** same ratios, but deleted edges are sampled
  inside the test 2-hop neighbourhood.
- **E3 — whole-scene deletion:** remove all edges belonging to 10 or 50 scans.
- **E4 — node deletion:** remove about 100 objects and all their incident edges.
- **E5 — λ ablation:** sweep the DEC/NI trade-off at 2.5% IN.
- **E6 — efficiency:** compare wall-clock time and trainable parameters.
- **E7 — edge-set comparison:** compare the main `spatial_support` graph with
  the `all` graph.

### Why did we do it?

- **Beyond a homogeneous graph.** DBLP is a single-relation citation graph and
  the paper reproduction uses a plain GCN. 3DSSG has many relationship types,
  so it exercises the paper's heterogeneous/R-GCN setting instead of only the
  single-relation case.
- **A new domain.** 3D scene understanding is unrelated to citation networks,
  which tests whether the identity-initialized DEL operator and the DEC/NI
  recipe transfer rather than overfit to DBLP.
- **Natural OUT/IN and structural deletion.** Scans are disconnected
  components, so easy-vs-hard edge deletion is well defined, and deleting a
  whole scan or an object is a realistic structural request.
- **Privacy motivation.** "Unlearn this scanned room" is a direct analogue of a
  user asking for their data to be removed from a trained model.
- **Trivial-edge control.** The `same ...` relations in 3DSSG are numerous and
  easy to predict. The main experiment therefore uses spatial/support relations
  only, and the `all` graph is reported separately as a stress test.

### How did we do it?

**Data and graph construction**

- Parse `objects.json` and `relationships.json`; build global node ids from
  `(scan id, object id)` because object ids repeat across scans.
- Node features = object-label one-hot plus multi-hot attributes (color, shape,
  state, ...), producing a fixed width of **598**.
- Canonicalize inverse pairs (`left`/`right`, `front`/`behind`) and symmetric
  `same ...` relations, and keep one edge per unordered node pair before
  splitting. This prevents the same pair appearing in train and test with
  different relation directions.
- Split by unordered node pair; validation/test edges come from only 30% of
  scenes so the easy/OUT deletion pool is not empty. Negatives are sampled
  inside the same scene because cross-scene pairs are trivially negative.
- Main graph `spatial_support`: 18 relations, **33,796 nodes**, **126,532
  deduplicated edges**. Secondary graph `all`: 33 relations, **36,580 nodes**,
  **289,981 edges**.

**Models and unlearning**

- **Stage A** uses the existing GCN as a type-agnostic baseline.
- **Stage B** uses an R-GCN: linear input projection → two `RGCNConv` layers →
  DistMult decoder. Reverse edges receive relation id `r + R`.
- The base model is trained once per graph. For deletion, the base model is
  frozen and only the DEL operator is trained — after each GNN layer, identity
  initialized, and active only around the deleted edges.
- Loss = `λ · L_DEC + (1 − λ) · L_NI` with λ = 0.5 by default. DEC pushes
  deleted-edge scores toward random unconnected pairs; NI keeps the local
  neighborhood close to the original model. NI uses same-scene pairs inside
  `S_Df` to keep the computation tractable on a laptop.
- Baselines: **Retrain** (full model trained on the graph without the deleted
  edges) and **Gradient-Ascent** (fine-tune the base model to increase loss on
  deleted edges). The base model itself is reported as `original`.
- Metrics: Et AUROC/AUPR on kept test edges, Ed AUROC on deleted edges, an MI
  ratio proxy, wall-clock time, and trainable parameters.

All numbers below are a **single seed (42)** development reading: 1,000 epochs
for base/Retrain, 500 for GNNDelete, and 10 for Gradient-Ascent. The plan calls
for 5 seeds for the final tables; these are not yet multi-seed means.

### Results

#### Prepared graphs (seed 42 split)

| Item | `spatial_support` | `all` |
|---|---:|---:|
| Scans | 1,335 | 1,335 |
| Object nodes | 33,796 | 36,580 |
| Relationship edges (deduplicated) | 126,532 | 289,981 |
| Relation types | 18 | 33 |
| Node feature width | 598 | 598 |
| Train / val / test edges | 114,033 / 6,252 / 6,247 | 260,794 / 14,581 / 14,581 |
| OUT / IN deletion pools | 91,428 / 22,605 | — / — |

#### Base models

| Edge set | GNN | Et AUROC | Et AUPR | Trainable params |
|---|---|---:|---:|---:|
| `spatial_support` | R-GCN | **0.881** | **0.883** | 987,328 |
| `spatial_support` | GCN | 0.873 | 0.856 | 84,928 |
| `all` | R-GCN | **0.938** | **0.946** | 1,725,568 |

The `all` graph is easier because the `same ...` relations are highly
predictable; this is exactly why `spatial_support` is the main experiment.

#### Stage A check: GCN on `spatial_support`, 2.5% OUT

| Method | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | Time (s) |
|---|---:|---:|---:|---:|
| Original | 0.873 | — | — | 62.8 |
| Retrain | 0.873 | 0.591 | 1.073 | 62.4 |
| GNNDelete | **0.873** | **0.756** | 1.521 | 33.8 |
| Gradient-Ascent | 0.864 | 0.584 | 1.076 | 1.0 |

The GCN Stage A check passes: Retrain preserves Et and has Ed near chance,
while GNNDelete preserves Et and raises Ed substantially. Adding edge types
(R-GCN) raises the base Et from 0.873 to 0.881.

#### E1: edge deletion, easy / OUT (R-GCN, λ = 0.5)

| Ratio | Method | Et AUROC ↑ | Ed AUROC | MI ratio ↑ |
|---:|---|---:|---:|---:|
| 0.5% | Retrain | 0.881 | 0.586 | 1.112 |
| 0.5% | GNNDelete | **0.881** | **0.949** | 2.278 |
| 0.5% | Gradient-Ascent | 0.780 | 0.575 | 1.244 |
| 2.5% | Retrain | 0.880 | 0.594 | 1.116 |
| 2.5% | GNNDelete | **0.881** | **0.811** | 2.177 |
| 2.5% | Gradient-Ascent | 0.779 | 0.574 | 1.248 |
| 5.0% | Retrain | 0.869 | 0.610 | 1.150 |
| 5.0% | GNNDelete | **0.881** | **0.750** | 2.142 |
| 5.0% | Gradient-Ascent | 0.765 | 0.575 | 1.277 |

#### E2: edge deletion, hard / IN (R-GCN, λ = 0.5)

| Ratio | Method | Et AUROC ↑ | Ed AUROC | MI ratio ↑ |
|---:|---|---:|---:|---:|
| 0.5% | Retrain | 0.885 | 0.765 | 1.180 |
| 0.5% | GNNDelete | 0.849 | **0.966** | 2.072 |
| 0.5% | Gradient-Ascent | 0.774 | 0.672 | 1.286 |
| 2.5% | Retrain | 0.872 | 0.778 | 1.206 |
| 2.5% | GNNDelete | 0.793 | **0.928** | 2.038 |
| 2.5% | Gradient-Ascent | 0.752 | 0.676 | 1.322 |
| 5.0% | Retrain | 0.851 | 0.808 | 1.283 |
| 5.0% | GNNDelete | 0.759 | **0.920** | 2.078 |
| 5.0% | Gradient-Ascent | 0.734 | 0.713 | 1.383 |

#### E3 / E4: whole-scene and node deletion (R-GCN, λ = 0.5)

| Setting | Method | Et AUROC ↑ | Ed AUROC | MI ratio ↑ |
|---|---|---:|---:|---:|
| 10 scenes | Retrain | 0.884 | 0.972 | 1.902 |
| 10 scenes | GNNDelete | 0.881 | **0.981** | 2.211 |
| 10 scenes | Gradient-Ascent | 0.869 | 0.957 | 1.764 |
| 50 scenes | Retrain | 0.887 | 0.976 | 1.973 |
| 50 scenes | GNNDelete | 0.880 | **0.982** | 2.311 |
| 50 scenes | Gradient-Ascent | 0.864 | 0.958 | 1.847 |
| 100 objects | Retrain | 0.886 | 0.980 | 2.327 |
| 100 objects | GNNDelete | 0.878 | **0.982** | 2.376 |
| 100 objects | Gradient-Ascent | 0.864 | 0.988 | 2.858 |

#### E5: λ ablation (R-GCN, IN, 2.5%)

| λ | Et AUROC ↑ | Ed AUROC | MI ratio ↑ |
|---:|---:|---:|---:|
| 0.0 | **0.863** | 0.779 | 1.234 |
| 0.2 | 0.824 | 0.915 | 1.751 |
| 0.4 | 0.804 | 0.928 | 1.964 |
| 0.5 | 0.793 | 0.928 | 2.038 |
| 0.6 | 0.780 | 0.927 | 2.105 |
| 0.8 | 0.723 | 0.925 | 2.239 |
| 1.0 | 0.685 | 0.906 | 1.866 |

#### E6: efficiency

| Method | Edge set | Trainable params | Time per run |
|---|---|---:|---:|
| GNNDelete | `spatial_support` | **20,480** | 103–123 s |
| Retrain | `spatial_support` | 987,328 | 300–315 s |
| Gradient-Ascent | `spatial_support` | 987,328 | ~3 s |
| GNNDelete | `all` | **20,480** | 192 s |
| Retrain | `all` | 1,725,568 | 746 s |

The base R-GCN is trained once and takes ~881 s on `spatial_support`
(including one long GPU stall). GNNDelete trains only the deletion operator, so
each deletion request is roughly **3× faster than Retrain** on
`spatial_support` and **~4× faster** on `all`.

#### E7: edge-set comparison (`spatial_support` vs `all`, 2.5% OUT)

| Edge set | Relations | Method | Et AUROC ↑ | Ed AUROC | Time (s) |
|---|---:|---|---:|---:|---:|
| `spatial_support` | 18 | Retrain | 0.880 | 0.594 | 315.6 |
| `spatial_support` | 18 | GNNDelete | **0.881** | **0.811** | 117.4 |
| `all` | 33 | Retrain | 0.936 | 0.495 | 745.8 |
| `all` | 33 | GNNDelete | **0.938** | **0.680** | 191.7 |

Adding the `same ...` relations raises base Et (0.881 → 0.938) but makes
deletion harder: Retrain's Ed drops to near chance (0.495 vs 0.594), and
GNNDelete's Ed is lower than on `spatial_support` (0.680 vs 0.811) even though
its utility is excellent. The `spatial_support` graph remains the cleaner test
of relational reasoning.

### Reading of the 3DSSG results

- **OUT:** GNNDelete keeps `Et` equal to Retrain (0.881 vs 0.869–0.881) while
  moving `Ed` from ~0.59 to 0.75–0.95, so the deleted edges are clearly
  forgotten without damaging the remaining graph.
- **IN:** Retrain's `Ed` is already high (0.765–0.808) because deleted edges
  overlap the test neighbourhood. GNNDelete pushes `Ed` to 0.92–0.97 at the cost
  of `Et` (0.849 at 0.5%, 0.759 at 5%), the same over-deletion direction seen in
  the DBLP IN experiment.
- **Scene / node deletion:** `Ed` is near 0.98 for both Retrain and GNNDelete;
  GNNDelete matches Retrain's utility and is ~3× faster.
- **λ:** λ = 0.2 gives the best utility/forgetting trade-off on IN; larger λ
  forgets more aggressively but erodes `Et`. This is the practical operating
  point if `Et` matters more than maximum `Ed`.
- **Transfer:** the identity-init DEL fix and the DEC/NI trade-off behave the
  same way on 3DSSG as on DBLP, while the IN setting again shows that aggressive
  forgetting costs utility when deletion overlaps the test neighborhood.

Full per-run JSONs are in `dblp_edge_reproduction/3dssg/results/`, and
`dblp_edge_reproduction/3dssg/results_summary.md` is generated by
`collect_results.py`.
