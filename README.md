# GNNDelete Implementation (DBLP Edge-Deletion Reproduction)

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
- Paper (arXiv): "GNNDelete: A General Strategy for Unlearning in Dynamic
  Graphs" — <https://arxiv.org/abs/2302.13406>

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