"""Train the base GNN ONCE on the full 3DSSG training graph (link prediction).

    python train_original.py --edge_set spatial_support --gnn gcn  --seed 42
    python train_original.py --edge_set spatial_support --gnn rgcn --seed 42

Writes checkpoints/<edge_set>/<gnn>/original/<seed>/model_best.pt and
       results/<edge_set>/<gnn>/original/<seed>.json   (test AUROC / AUPR, also per relation)
"""
import argparse
import os

import torch

from common import (ROOT, count_trainable, eval_lp, get_device, load_data, make_mp, results_path,
                    save_json, set_seed, to_device, train_link_prediction)
from models import build_model


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--root', default=ROOT, help='where checkpoints/ and results/ are written')
    ap.add_argument('--edge_set', default='spatial_support')
    ap.add_argument('--gnn', default='gcn', choices=['gcn', 'rgcn'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--out_dim', type=int, default=64)
    ap.add_argument('--num_bases', type=int, default=0, help='R-GCN basis decomposition (0 = off)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--valid_freq', type=int, default=50)
    return ap.parse_args(argv)


def main():
    args = get_args()
    device = get_device()
    set_seed(args.seed)
    d = to_device(load_data(args.data_dir, args.edge_set, args.seed), device)
    print(f'device={device} nodes={d.n:,} train edges={d.train.src.numel():,} relations={d.R} feat={d.x.shape[1]}')

    mp = make_mp(d.train, d.R)
    model = build_model(args, d).to(device)
    print(f'{args.gnn}: {count_trainable(model):,} trainable parameters')
    info = train_link_prediction(model, d, d.train, mp, args.epochs, args.lr, args.valid_freq, desc='original')

    test, _ = eval_lp(model, d, 'test', mp, per_relation=True)
    print(f'\n=== BASE MODEL ({args.gnn}, {args.edge_set}, seed {args.seed}) ===')
    print(f'test AUROC {test["auroc"]:.4f} | test AUPR {test["aupr"]:.4f} | best epoch {info["best_epoch"]}')
    for r, v in sorted(test['per_relation'].items(), key=lambda kv: -kv[1]['n']):
        print(f'   {r:18s} n={v["n"]:5d} AUROC {v["auroc"]:.3f}')

    ck = os.path.join(args.root, 'checkpoints', args.edge_set, args.gnn, 'original', str(args.seed))
    os.makedirs(ck, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'args': vars(args)}, os.path.join(ck, 'model_best.pt'))
    save_json(results_path(args.root, args.edge_set, args.gnn, 'original', str(args.seed)),
              dict(method='original', edge_set=args.edge_set, gnn=args.gnn, seed=args.seed,
                   et_auroc=test['auroc'], et_aupr=test['aupr'], per_relation=test['per_relation'],
                   trainable_params=count_trainable(model), args=vars(args), **info))


if __name__ == '__main__':
    main()
