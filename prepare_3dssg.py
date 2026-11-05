"""Build the 3DSSG link-prediction graph for GNNDelete (numpy only, no torch needed).

Reads  raw/objects.json, raw/relationships.json   (from 3DSSG.zip)
Writes data/<edge_set>/graph.npz            (nodes, features, canonical edges)
       data/<edge_set>/split_<seed>.npz     (train/val/test split + OUT/IN masks)

Key design decisions (see README for the reasoning):
  * Global node ids = (scan, object id). Nodes of one scan are CONTIGUOUS.
  * Inverse relations (right/behind/lower than/smaller than/darker than/cleaner than)
    are rewritten as their canonical partner with swapped endpoints, symmetric
    relations (same *, close by, ...) are stored once as (min, max), then duplicates
    are removed per (scan, a, b, relation).  Relation types are KEPT (a node pair
    may carry several relations).
  * Splits are done by UNORDERED NODE PAIR, so a pair seen in training with one
    relation never appears in val/test with another relation.
  * Val/test edges are only drawn from a fraction of scans (--heldout_scene_frac).
    Otherwise every scene contains test edges, the 2-hop "IN" region covers almost
    every scene and the easy "OUT" deletion set would be (nearly) empty.
  * Negatives are sampled INSIDE the same scan, never across scans (cross-scan pairs
    are trivially negative and would inflate AUROC).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.abspath(__file__))

# inverse relation -> canonical relation (endpoints are swapped)
INVERSE = {
    'right': 'left', 'behind': 'front', 'lower than': 'higher than',
    'smaller than': 'bigger than', 'darker than': 'brighter than',
    'cleaner than': 'messier than',
}
# kept in the main experiment: support / spatial / part-whole relations
SPATIAL_SUPPORT = [
    'supported by', 'standing on', 'attached to', 'left', 'front', 'close by',
    'lying on', 'hanging on', 'connected to', 'leaning against', 'part of',
    'belonging to', 'build in', 'standing in', 'cover', 'lying in', 'hanging in',
    'higher than',
]
ATTR_KEYS = ['color', 'shape', 'style', 'state', 'size', 'material', 'texture', 'other', 'symmetry']


def load_raw(raw_dir):
    with open(os.path.join(raw_dir, 'objects.json')) as f:
        objects = json.load(f)['scans']
    with open(os.path.join(raw_dir, 'relationships.json')) as f:
        rels = json.load(f)['scans']
    return objects, rels


def detect_symmetric(rels, thr=0.9):
    """A relation is symmetric if (a,b,r) is almost always accompanied by (b,a,r)."""
    tot, rec = Counter(), Counter()
    for s in rels:
        S = {(a, b, nm) for a, b, _, nm in s['relationships']}
        for a, b, nm in S:
            tot[nm] += 1
            rec[nm] += (b, a, nm) in S
    return {nm for nm in tot if rec[nm] / tot[nm] >= thr}


def build_graph(objects, rels, edge_set='spatial_support', use_attr=True):
    obj_by_scan = {s['scan']: s['objects'] for s in objects}
    rel_scans = [s for s in rels if len(s['relationships']) > 0]
    scan_ids = sorted(s['scan'] for s in rel_scans)
    rel_by_scan = {s['scan']: s['relationships'] for s in rel_scans}
    symmetric = detect_symmetric(rels)

    # ---- nodes: contiguous per scan --------------------------------------------------
    node_of = {}                       # (scan, obj_id) -> global idx
    labels, attrs, node_scan = [], [], []
    scan_ptr = [0]
    for si, scan in enumerate(scan_ids):
        for ob in obj_by_scan[scan]:
            node_of[(scan, int(ob['id']))] = len(labels)
            labels.append(ob['label'])
            a = []
            if use_attr:
                for k, vals in ob.get('attributes', {}).items():
                    if k in ATTR_KEYS:
                        a += [f'{k}:{v}' for v in vals]
            attrs.append(a)
            node_scan.append(si)
        scan_ptr.append(len(labels))
    n = len(labels)

    label_vocab = sorted(set(labels))
    attr_vocab = sorted({a for al in attrs for a in al})
    l_idx = {l: i for i, l in enumerate(label_vocab)}
    a_idx = {a: i for i, a in enumerate(attr_vocab)}
    x = np.zeros((n, len(label_vocab) + len(attr_vocab)), dtype=np.uint8)
    for i, (l, al) in enumerate(zip(labels, attrs)):
        x[i, l_idx[l]] = 1
        for a in al:
            x[i, len(label_vocab) + a_idx[a]] = 1

    # ---- edges: canonicalise + dedupe, keep relation types ----------------------------
    keep = set()
    n_raw = n_self = 0
    for scan in scan_ids:
        for a, b, _, nm in rel_by_scan[scan]:
            n_raw += 1
            if a == b:
                n_self += 1
                continue
            if nm in INVERSE:
                nm, a, b = INVERSE[nm], b, a
            if nm in symmetric and a > b:
                a, b = b, a
            if edge_set == 'spatial_support' and nm not in SPATIAL_SUPPORT:
                continue
            if edge_set == 'no_same' and nm.startswith('same'):
                continue
            keep.add((node_of[(scan, a)], node_of[(scan, b)], nm))
    rel_names = sorted({nm for _, _, nm in keep})
    r_idx = {r: i for i, r in enumerate(rel_names)}
    arr = np.array(sorted((u, v, r_idx[nm]) for u, v, nm in keep), dtype=np.int64)
    src, dst, et = arr[:, 0], arr[:, 1], arr[:, 2]

    # drop nodes that have no edge in this edge set (keeps scan contiguity)
    deg = np.bincount(np.concatenate([src, dst]), minlength=n)
    alive = deg > 0
    n_isolated = int((~alive).sum())
    remap = np.cumsum(alive) - 1
    src, dst = remap[src], remap[dst]
    x = x[alive]
    node_scan = np.array(node_scan)[alive]
    scan_ptr = np.concatenate([[0], np.cumsum(np.bincount(node_scan, minlength=len(scan_ids)))])
    n = int(alive.sum())
    stats = dict(raw_relationships=n_raw, self_loops_dropped=n_self, edges=len(src),
                 nodes=n, isolated_nodes_dropped=n_isolated, scans=len(scan_ids), relations=len(rel_names),
                 symmetric_relations=sorted(symmetric))
    return dict(x=x, node_scan=node_scan, scan_ptr=scan_ptr,
                src=src, dst=dst, et=et, rel_names=np.array(rel_names),
                label_vocab=np.array(label_vocab), attr_vocab=np.array(attr_vocab),
                scan_ids=np.array(scan_ids), stats=stats)


def pair_key(u, v, n):
    return np.minimum(u, v) * n + np.maximum(u, v)


def khop_nodes(n, src, dst, seeds, hops):
    """Boolean mask of nodes within `hops` of `seeds` on the undirected graph."""
    A = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    A = (A + A.T).tocsr()
    vis = np.zeros(n, dtype=np.float32)
    vis[seeds] = 1
    for _ in range(hops):
        vis = np.maximum(vis, (A @ vis > 0).astype(np.float32))
    return vis > 0


def sample_neg_dst(src, dst, node_scan, scan_ptr, forbidden_sorted, n, rng, tries=60):
    """For each positive (s, o) pick a corrupted tail t in the SAME scan with (s,t) not an
    existing pair and t != s.  Returns (neg_dst, valid)."""
    scan = node_scan[src]
    start, size = scan_ptr[scan], scan_ptr[scan + 1] - scan_ptr[scan]
    neg = np.full(len(src), -1, dtype=np.int64)
    todo = np.arange(len(src))
    for _ in range(tries):
        if len(todo) == 0:
            break
        cand = start[todo] + (rng.random(len(todo)) * size[todo]).astype(np.int64)
        k = pair_key(src[todo], cand, n)
        pos = np.searchsorted(forbidden_sorted, k)
        pos = np.minimum(pos, len(forbidden_sorted) - 1)
        bad = (forbidden_sorted[pos] == k) | (cand == src[todo])
        neg[todo[~bad]] = cand[~bad]
        todo = todo[bad]
    return neg, neg >= 0


def make_split(g, seed, val_ratio, test_ratio, heldout_scene_frac):
    rng = np.random.default_rng(seed)
    n = len(g['x'])
    src, dst, et = g['src'], g['dst'], g['et']
    n_scans = len(g['scan_ids'])

    keys = pair_key(src, dst, n)
    upairs, inv = np.unique(keys, return_inverse=True)
    pair_scan = g['node_scan'][upairs // n]

    held = np.zeros(n_scans, dtype=bool)
    held[rng.permutation(n_scans)[:max(1, int(round(heldout_scene_frac * n_scans)))]] = True
    cand = np.nonzero(held[pair_scan])[0]
    n_val = int(val_ratio * len(upairs))
    n_test = int(test_ratio * len(upairs))
    n_val, n_test = min(n_val, len(cand) // 3), min(n_test, len(cand) // 3)
    perm = rng.permutation(cand)
    test_p, val_p = perm[:n_test], perm[n_test:n_test + n_val]
    role = np.zeros(len(upairs), dtype=np.int8)       # 0 train, 1 val, 2 test
    role[val_p], role[test_p] = 1, 2
    e_role = role[inv]

    forbidden = np.sort(upairs)       # every existing pair (any relation, any split)
    out = {'seed': seed, 'heldout_scenes': held}
    # sample negatives; a pair whose edge has no valid same-scan negative (tiny scene) goes
    # back to train TOGETHER with all its other edges, so pairs never straddle two splits
    negs, oks = {}, {}
    bad_pairs = []
    for name, code in (('val', 1), ('test', 2)):
        idx = np.nonzero(e_role == code)[0]
        neg, ok = sample_neg_dst(src[idx], dst[idx], g['node_scan'], g['scan_ptr'], forbidden, n, rng)
        negs[name], oks[name] = (idx, neg), ok
        bad_pairs.append(inv[idx[~ok]])
    bad_pairs = np.unique(np.concatenate(bad_pairs))
    to_train = np.isin(inv, bad_pairs)
    for name, code in (('val', 1), ('test', 2)):
        idx, neg = negs[name]
        keep_ = ~to_train[idx]
        idx = idx[keep_]
        out[f'{name}_src'], out[f'{name}_dst'], out[f'{name}_et'] = src[idx], dst[idx], et[idx]
        out[f'{name}_neg_dst'] = neg[keep_]
    m = (e_role == 0) | to_train
    out['train_src'], out['train_dst'], out['train_et'] = src[m], dst[m], et[m]

    # OUT / IN deletion masks over the canonical TRAIN edges (paper: 2-hop of test edges)
    seeds = np.unique(np.concatenate([out['test_src'], out['test_dst']]))
    vis = khop_nodes(n, out['train_src'], out['train_dst'], seeds, 2)
    in_mask = vis[out['train_src']] & vis[out['train_dst']]
    out['mask_in'], out['mask_out'] = in_mask, ~in_mask
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=os.path.join(ROOT, 'raw'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--edge_set', default='spatial_support', choices=['spatial_support', 'all', 'no_same'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 21, 13, 87, 100])
    ap.add_argument('--val_ratio', type=float, default=0.05, help='fraction of ALL node pairs')
    ap.add_argument('--test_ratio', type=float, default=0.05, help='fraction of ALL node pairs')
    ap.add_argument('--heldout_scene_frac', type=float, default=0.3,
                    help='val/test edges are drawn only from this fraction of scans')
    ap.add_argument('--no_attr_features', action='store_true', help='label one-hot only')
    a = ap.parse_args()

    objects, rels = load_raw(a.raw_dir)
    g = build_graph(objects, rels, a.edge_set, use_attr=not a.no_attr_features)
    d = os.path.join(a.out_dir, a.edge_set)
    os.makedirs(d, exist_ok=True)
    stats = g.pop('stats')
    print(json.dumps(stats, indent=1))
    print('relations:', [str(r) for r in g['rel_names']])
    print('feature dim:', g['x'].shape[1])
    np.savez_compressed(os.path.join(d, 'graph.npz'), **g)
    with open(os.path.join(d, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)

    for s in a.seeds:
        sp_ = make_split(g, s, a.val_ratio, a.test_ratio, a.heldout_scene_frac)
        np.savez_compressed(os.path.join(d, f'split_{s}.npz'), **sp_)
        print(f"seed {s}: train={len(sp_['train_src']):,} val={len(sp_['val_src']):,} "
              f"test={len(sp_['test_src']):,} | OUT={sp_['mask_out'].sum():,} IN={sp_['mask_in'].sum():,}")
    print('done ->', d)


if __name__ == '__main__':
    sys.exit(main())
