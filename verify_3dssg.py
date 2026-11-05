"""Sanity checks on the prepared 3DSSG data (numpy only).  Exit code 1 if a check fails.

    python verify_3dssg.py --edge_set spatial_support
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def pair_key(u, v, n):
    return np.minimum(u, v) * n + np.maximum(u, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--edge_set', default='spatial_support')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    a = ap.parse_args()
    d = os.path.join(a.data_dir, a.edge_set)
    g = np.load(os.path.join(d, 'graph.npz'), allow_pickle=True)
    n = len(g['x'])
    ok = True

    def check(name, cond, info=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    src, dst, et = g['src'], g['dst'], g['et']
    ns, ptr = g['node_scan'], g['scan_ptr']
    R = len(g['rel_names'])
    print(f'nodes={n:,} edges={len(src):,} relations={R} scans={len(g["scan_ids"]):,} feat={g["x"].shape[1]}')
    check('no self-loops', (src != dst).all())
    check('edges stay inside one scan', (ns[src] == ns[dst]).all())
    check('scan nodes are contiguous', (np.diff(ns) >= 0).all() and ptr[-1] == n)
    trip = src * n * R + dst * R + et
    check('no duplicate (u,v,relation)', len(np.unique(trip)) == len(trip))
    rev = dst * n * R + src * R + et
    n_rec = int(np.isin(trip, rev).sum())
    check('(near-)no reciprocal duplicates (u,v,r)/(v,u,r)', n_rec <= 0.001 * len(src),
          f'({n_rec} found; mutual "supported by" annotations are expected)')
    check('relation ids valid', et.min() >= 0 and et.max() < R)
    print('edges per relation:', {str(r): int((et == i).sum()) for i, r in enumerate(g['rel_names'])})
    deg = np.bincount(np.concatenate([src, dst]), minlength=n)
    check('no isolated nodes', (deg > 0).all(), f'(isolated: {(deg == 0).sum()})')

    for s in a.seeds:
        sp = np.load(os.path.join(d, f'split_{s}.npz'))
        print(f'--- seed {s}')
        kt = pair_key(sp['train_src'], sp['train_dst'], n)
        kv = pair_key(sp['val_src'], sp['val_dst'], n)
        ke = pair_key(sp['test_src'], sp['test_dst'], n)
        check('train/val/test node pairs are disjoint',
              not (np.isin(kv, kt).any() or np.isin(ke, kt).any() or np.isin(ke, kv).any()))
        allk = np.sort(pair_key(src, dst, n))
        for nm in ('val', 'test'):
            ns_, nd_ = sp[f'{nm}_src'], sp[f'{nm}_neg_dst']
            kn = pair_key(ns_, nd_, n)
            check(f'{nm} negatives are not real pairs', not np.isin(kn, allk).any())
            check(f'{nm} negatives are in the same scan as the positive', (ns[ns_] == ns[nd_]).all())
            check(f'{nm} pos/neg aligned', len(ns_) == len(nd_) == len(sp[f'{nm}_et']))
        check('train+val+test cover all edges',
              len(sp['train_src']) + len(sp['val_src']) + len(sp['test_src']) == len(src))

        mo, mi = sp['mask_out'], sp['mask_in']
        check('OUT and IN partition the train edges', (mo ^ mi).all() and len(mo) == len(sp['train_src']))
        print(f'    train={len(sp["train_src"]):,} val={len(sp["val_src"]):,} test={len(sp["test_src"]):,} '
              f'OUT={mo.sum():,} IN={mi.sum():,}  (5% of train = {int(0.05 * len(mo)):,})')
        check('OUT large enough for a 5% deletion', mo.sum() >= 0.05 * len(mo))
        check('IN large enough for a 5% deletion', mi.sum() >= 0.05 * len(mo))
    print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
