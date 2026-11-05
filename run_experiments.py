"""Run the experiment grid of plan_extension.md (resumable: finished runs are skipped).

    python run_experiments.py --quick                 # tiny smoke test (1 seed, a few epochs)
    python run_experiments.py --stage base --gnn gcn  # Stage A base model only
    python run_experiments.py --stage e1 --gnn rgcn --seeds 42
    python run_experiments.py --stage all --gnn rgcn --seeds 42 21 13 87 100

Stages: prepare | base | e1 (OUT 0.5/2.5/5 %) | e2 (IN 0.5/2.5/5 %) | e3 (scenes 10, 50)
        e4 (100 nodes) | e5 (lambda ablation, IN 2.5 %) | all
(E6 efficiency = time / params columns of every result; E7 = rerun with --edge_set all.)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd):
    print('\n$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print('!! command failed, continuing with the next one', flush=True)
    return r.returncode


def done(edge_set, gnn, method, tag):
    return os.path.exists(os.path.join(ROOT, 'results', edge_set, gnn, method, f'{tag}.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    ap.add_argument('--gnn', default='rgcn', choices=['gcn', 'rgcn'])
    ap.add_argument('--edge_set', default='spatial_support')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--methods', nargs='+', default=['retrain', 'gnndelete', 'gradient_ascent'])
    ap.add_argument('--quick', action='store_true', help='smoke test: seed 42, very few epochs, stage e1 @2.5%%')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    common = ['--edge_set', a.edge_set, '--gnn', a.gnn]
    q = a.quick
    seeds = [42] if q else a.seeds
    ep_base = ['--epochs', '30', '--valid_freq', '10'] if q else []
    ep = {'retrain': ['--epochs', '30', '--valid_freq', '10'], 'gnndelete': ['--epochs', '20', '--valid_freq', '10'],
          'gradient_ascent': ['--epochs', '3', '--valid_freq', '1']} if q else {}
    stage = 'e1' if q else a.stage

    if stage in ('prepare', 'all'):
        run([PY, 'prepare_3dssg.py', '--edge_set', a.edge_set, '--seeds', *map(str, seeds if q else [42, 21, 13, 87, 100])])
        run([PY, 'verify_3dssg.py', '--edge_set', a.edge_set, '--seeds', *map(str, seeds)])
    if stage in ('base', 'e1', 'e2', 'e3', 'e4', 'e5', 'all'):
        for s in seeds:
            if a.force or not done(a.edge_set, a.gnn, 'original', str(s)):
                run([PY, 'train_original.py', *common, '--seed', str(s), *ep_base])

    jobs = []     # (df, size, method, alpha)
    if stage in ('e1', 'all'):
        jobs += [('out', z, m, 0.5) for z in ([2.5] if q else [0.5, 2.5, 5.0]) for m in a.methods]
    if stage in ('e2', 'all') and not q:
        jobs += [('in', z, m, 0.5) for z in [0.5, 2.5, 5.0] for m in a.methods]
    if stage in ('e3', 'all') and not q:
        jobs += [('scene', z, m, 0.5) for z in [10, 50] for m in a.methods]
    if stage in ('e4', 'all') and not q:
        jobs += [('node', 100, m, 0.5) for m in a.methods]
    if stage in ('e5', 'all') and not q:
        jobs += [('in', 2.5, 'gnndelete', al) for al in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]

    from common import unlearn_tag
    for s in seeds:
        for df, size, m, al in jobs:
            if not a.force and done(a.edge_set, a.gnn, m, unlearn_tag(m, df, size, al, s)):
                continue
            extra = ep.get(m, []) if q else []
            run([PY, 'delete_edges.py', *common, '--seed', str(s), '--unlearning_model', m, '--df', df,
                 '--df_size', str(size), '--alpha', str(al), *extra])
    run([PY, 'collect_results.py'])


if __name__ == '__main__':
    main()
