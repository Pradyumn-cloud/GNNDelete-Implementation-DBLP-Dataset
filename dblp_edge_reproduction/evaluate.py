'''Et / Ed evaluation readout for a completed GNNDelete/retrain run.

Loads `trainer_log.json` (and `training_args.json`) written by
`delete_edges.py` / `train_original.py` and prints the paper-style
Et / Ed AUROC table, plus the paper reference values for DBLP.

Usage:
    python evaluate.py <checkpoint_dir>
    e.g.  python evaluate.py checkpoints/DBLP/gcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected/out-10-42
'''

import os
import sys
import json

PAPER_REF = {  # GNNDelete, DBLP 2.5% edge deletion, GCN, link prediction (Table 1)
    'RETRAIN':  {'Et': 0.964, 'Ed': 0.506},
    'GNNDelete': {'Et': 0.934, 'Ed': 0.748},
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ckpt_dir = sys.argv[1]

    log_path = os.path.join(ckpt_dir, 'trainer_log.json')
    args_path = os.path.join(ckpt_dir, 'training_args.json')

    with open(log_path, 'r') as f:
        log = json.load(f)

    with open(args_path, 'r') as f:
        args = json.load(f)

    et = log.get('dt_auc')
    ed = log.get('df_auc')
    ve = log.get('ve')

    print('=' * 64)
    print(f'Run  : {args.get("unlearning_model")} | {args.get("dataset")} | {args.get("gnn")} | '
          f"df={args.get('df')} df_size={args.get('df_size')} seed={args.get('random_seed')}")
    print(f"Model: {args.get('gnn')} hidden={args.get('hidden_dim')} out={args.get('out_dim')} "
          f"(lr={args.get('lr')}, epochs={args.get('epochs')})")
    print('=' * 64)
    print(f'  Et AUROC (retained/test edges): {et:.4f}' if et is not None else '  Et AUROC: n/a')
    print(f'  Et AUPR  (retained/test edges): {log.get("dt_aup"):.4f}' if log.get('dt_aup') is not None else '  Et AUPR: n/a')
    print(f'  Ed AUROC (deleted edges)      : {ed:.4f}' if ed is not None else '  Ed AUROC: n/a')
    print(f'  Ed AUPR  (deleted edges)      : {log.get("df_aup"):.4f}' if log.get('df_aup') is not None else '  Ed AUPR: n/a')
    print(f'  auc_sum (dt+df)               : {log.get("auc_sum"):.4f}' if log.get('auc_sum') is not None else '')
    print(f'  verification error (vs retrain): {ve:.4f}' if ve is not None else '  verification error: n/a')
    print(f'  best epoch                     : {log.get("best_epoch")}' if log.get('best_epoch') is not None else '')
    print(f'  best metric                    : {log.get("best_metric")}' if log.get('best_metric') is not None else '')

    method = args.get('unlearning_model', '')
    if 'gnndelete' in method:
        label = 'GNNDelete'
    elif method == 'retrain':
        label = 'RETRAIN'
    else:
        label = None

    if label is not None:
        ref = PAPER_REF[label]
        print('-' * 64)
        print(f'Paper reference (DBLP 2.5%): {label}: Et={ref["Et"]}, Ed={ref["Ed"]}')
        print('  (environment/preprocessing/seed differences expected; not an exact target)')

    return 0


if __name__ == '__main__':
    sys.exit(main())