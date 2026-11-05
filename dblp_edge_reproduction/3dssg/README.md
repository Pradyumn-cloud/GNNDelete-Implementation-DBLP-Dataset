# GNNDelete on 3DSSG (3D semantic scene graphs)

Extension of the DBLP reproduction. Link prediction on scene graphs, then unlearning
relationships / whole scenes / objects with GNNDelete (compared with Retrain and Gradient-Ascent).

## Setup
    pip install -r requirements.txt        # torch + torch_geometric (see pyg.org if the install fails)
Put `objects.json` and `relationships.json` (from 3DSSG.zip) into `3dssg/raw/`.

## Run
    cd 3dssg
    python prepare_3dssg.py                 # builds data/spatial_support/ (numpy only, ~1 min)
    python verify_3dssg.py --seeds 42       # every line must say PASS
    python run_experiments.py --quick       # tiny end-to-end smoke test (rgcn, 1 seed)
    python run_experiments.py --stage all --gnn rgcn --seeds 42      # full grid, one seed
    python collect_results.py               # -> results_summary.md

Manual commands: `train_original.py --gnn rgcn`, then
`delete_edges.py --unlearning_model gnndelete|retrain|gradient_ascent|original --df out|in|scene|node --df_size 2.5`.
Use `--gnn gcn` for the type-agnostic Stage A baseline. Results land in `results/` (commit these).

## Deliberate differences from the DBLP code (and why)
- Negatives are sampled inside the same scene (cross-scene pairs are trivially negative).
- Split by unordered node pair, so one pair never appears in train and test with different relations.
- Val/test edges come from only 30% of scenes (`--heldout_scene_frac`); otherwise every scene
  has test edges and the easy OUT deletion pool is nearly empty.
- The NI loss uses same-scene node pairs inside S_Df instead of a dense N x N matrix (too big for a laptop).
- R-GCN = Linear projection -> 2 RGCNConv layers, DistMult decoder, reverse edges get relation id r+R.
- Deletion operator starts as identity (the fix from the DBLP work).
- MI ratio is a simplified logistic-regression proxy, not the paper's MLP attacker.
- Training code has not been run by the author yet: first run `--quick` and report any error.
