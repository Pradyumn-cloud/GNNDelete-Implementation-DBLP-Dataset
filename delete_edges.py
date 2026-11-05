"""Unlearning on 3DSSG:  GNNDelete vs. Retrain / Gradient-Ascent, edge / scene / node deletion.

    # E1/E2: delete 2.5 % of the training edges, easy (out) or hard (in) setting
    python delete_edges.py --unlearning_model gnndelete --df out --df_size 2.5 --gnn rgcn --seed 42
    python delete_edges.py --unlearning_model retrain   --df out --df_size 2.5 --gnn rgcn --seed 42
    # E3: forget 10 whole scenes        E4: forget 100 objects (and all their edges)
    python delete_edges.py --unlearning_model gnndelete --df scene --df_size 10 --gnn rgcn
    python delete_edges.py --unlearning_model gnndelete --df node  --df_size 100 --gnn rgcn

Needs the base model from train_original.py (same --edge_set/--gnn/--seed).
Metrics (paper definitions)
  Et AUROC : link prediction on held-out test edges that were NOT deleted (utility, higher = better)
  Ed AUROC : deleted edges (label 0) vs. an equal number of remaining edges (label 1).
             Retrain's value is the reference; closer to it (or higher, as in the paper) = forgotten.
  MI ratio : simplified membership-inference proxy (see README), > 1 = less information left.
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from common import (ROOT, count_trainable, eval_deleted, eval_lp, fit_mi_attacker, get_device, load_data,
                    make_mp, pair_key, presence, results_path, sample_neg_dst, save_json, set_seed, subset,
                    to_device, train_link_prediction, unlearn_tag)
from df_utils import build_ni_pairs, keep_mask, sdf_sets, select_df
from models import build_model

DEFAULT_EPOCHS = {'gnndelete': 500, 'retrain': 1000, 'gradient_ascent': 10, 'original': 0}
DEFAULT_VALID = {'gnndelete': 25, 'retrain': 50, 'gradient_ascent': 1, 'original': 1}


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--edge_set', default='spatial_support')
    ap.add_argument('--gnn', default='gcn', choices=['gcn', 'rgcn'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--unlearning_model', default='gnndelete',
                    choices=['gnndelete', 'retrain', 'gradient_ascent', 'original'])
    ap.add_argument('--df', default='out', choices=['out', 'in', 'scene', 'node'])
    ap.add_argument('--df_size', type=float, default=2.5,
                    help='out/in: percent of train edges (<100) or a count (>=100); scene/node: a count')
    ap.add_argument('--alpha', type=float, default=0.5, help='lambda: weight of Deleted-Edge-Consistency loss')
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--out_dim', type=int, default=64)
    ap.add_argument('--num_bases', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--valid_freq', type=int, default=None)
    ap.add_argument('--ni_pairs', type=int, default=200000, help='max node pairs per step for the NI loss')
    ap.add_argument('--df_auc_repeats', type=int, default=50)
    ap.add_argument('--no_mi', action='store_true')
    ap.add_argument('--save_ckpt', action='store_true')
    a = ap.parse_args(argv)
    a.epochs = DEFAULT_EPOCHS[a.unlearning_model] if a.epochs is None else a.epochs
    a.valid_freq = DEFAULT_VALID[a.unlearning_model] if a.valid_freq is None else a.valid_freq
    return a


def load_original(args, d, device, with_del):
    path = os.path.join(args.root, 'checkpoints', args.edge_set, args.gnn, 'original', str(args.seed), 'model_best.pt')
    if not os.path.exists(path):
        raise SystemExit(f'Base model not found: {path}\nRun train_original.py with the same --edge_set/--gnn/--seed first.')
    model = build_model(args, d, with_del=with_del).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state'], strict=False)   # deletion layers are not in the base ckpt
    return model


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def select_by_val(model, d, mp_dr, keep_val, df, dr):
    m, z = eval_lp(model, d, 'val', mp_dr, keep_val)
    e = eval_deleted(model, z, df, dr, repeats=10)
    return m['auroc'], e['auroc']


def run_gnndelete(args, d, device, df, dr, mp_dr, mp_full, keep_val, nodes1, nodes2, sdf_np, ni, forbidden):
    model = load_original(args, d, device, with_del=True)
    model.mask1, model.mask2 = nodes1, nodes2
    model.freeze_base()
    base = load_original(args, d, device, with_del=False).eval()
    with torch.no_grad():
        z_ori = base(d.x, mp_full.ei, mp_full.et)
    mp_sdf = make_mp(subset(d.train, sdf_np), d.R)
    params = [p for n, p in model.named_parameters() if 'deletion' in n]
    print(f'trainable (deletion operator) parameters: {sum(p.numel() for p in params):,}')
    opt = torch.optim.Adam(params, lr=args.lr)
    ni_i, ni_j = ni
    best, best_state, best_epoch, t_train = -1.0, None, -1, 0.0
    for epoch in range(args.epochs):
        _sync(device); t0 = time.time()
        model.train()
        z = model(d.x, mp_sdf.ei, mp_sdf.et)
        # Deleted Edge Consistency: deleted edges should score like random unconnected pairs
        neg = sample_neg_dst(df.src, d, forbidden)
        ok = neg >= 0
        p_del = model.decode(z, df.src[ok], df.dst[ok], df.et[ok]).sigmoid()
        p_rnd = model.decode(z, df.src[ok], neg[ok], df.et[ok]).sigmoid()
        loss_r = F.mse_loss(p_del, p_rnd)
        # Neighborhood Influence: pair probabilities inside S_Df stay close to the original model
        if ni_i.numel() > 0:
            idx = torch.randperm(ni_i.numel(), device=device)[:args.ni_pairs]
            i, j = ni_i[idx], ni_j[idx]
            loss_l = F.mse_loss((z[i] * z[j]).sum(-1).sigmoid(), (z_ori[i] * z_ori[j]).sum(-1).sigmoid())
        else:
            loss_l = torch.zeros((), device=device)
        loss = args.alpha * loss_r + (1 - args.alpha) * loss_l
        opt.zero_grad(); loss.backward(); opt.step()
        _sync(device); t_train += time.time() - t0

        if (epoch + 1) % args.valid_freq == 0 or epoch == args.epochs - 1:
            dt, de = select_by_val(model, d, mp_dr, keep_val, df, dr)
            print(f'  epoch {epoch + 1:4d} | loss {loss.item():.4f} (DEC {loss_r.item():.4f}, NI {loss_l.item():.4f}) '
                  f'| val Et {dt:.4f} Ed {de:.4f}')
            if dt + de > best:
                best, best_epoch = dt + de, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_epoch, train_time_sec=t_train), count_trainable(model)


def run_gradient_ascent(args, d, device, df, dr, mp_dr, keep_val, forbidden):
    model = load_original(args, d, device, with_del=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, best_epoch, t_train = -1.0, None, -1, 0.0
    for epoch in range(args.epochs):
        _sync(device); t0 = time.time()
        model.train()
        z = model(d.x, mp_dr.ei, mp_dr.et)
        neg = sample_neg_dst(df.src, d, forbidden)
        ok = neg >= 0
        pos_l = model.decode(z, df.src, df.dst, df.et)
        neg_l = model.decode(z, df.src[ok], neg[ok], df.et[ok])
        logits = torch.cat([pos_l, neg_l])
        label = torch.cat([torch.ones_like(pos_l), torch.zeros_like(neg_l)])
        loss = -F.binary_cross_entropy_with_logits(logits, label)      # ascent on the deleted edges
        opt.zero_grad(); loss.backward(); opt.step()
        _sync(device); t_train += time.time() - t0
        if (epoch + 1) % args.valid_freq == 0 or epoch == args.epochs - 1:
            dt, de = select_by_val(model, d, mp_dr, keep_val, df, dr)
            print(f'  epoch {epoch + 1:3d} | val Et {dt:.4f} Ed {de:.4f}')
            if dt + de > best:
                best, best_epoch = dt + de, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_epoch, train_time_sec=t_train), count_trainable(model)


def main():
    args = get_args()
    device = get_device()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    d = load_data(args.data_dir, args.edge_set, args.seed)               # CPU first (numpy helpers)
    src_np, dst_np = d.train.src.numpy(), d.train.dst.numpy()
    df_np, extra = select_df(args.df, args.df_size, rng, src_np, dst_np, d.mask_in, d.mask_out, d.node_scan_np)
    keep_val = keep_mask(d.val.src.numpy(), d.val.dst.numpy(), d.val.neg_dst.numpy(), d.node_scan_np, extra)
    keep_test = keep_mask(d.test.src.numpy(), d.test.dst.numpy(), d.test.neg_dst.numpy(), d.node_scan_np, extra)
    nodes1_np, nodes2_np, sdf_np = sdf_sets(d.n, src_np, dst_np, df_np)
    ni_np = build_ni_pairs(nodes2_np, d.node_scan_np, src_np[df_np], dst_np[df_np], d.n)
    print(f'|Df|={df_np.sum():,} of {len(df_np):,} train edges | S_Df nodes: 1-hop {nodes1_np.sum():,}, '
          f'2-hop {nodes2_np.sum():,} | Et edges kept: {keep_test.sum():,}/{len(keep_test):,}')

    to_device(d, device)
    df, dr = subset(d.train, df_np), subset(d.train, ~df_np)
    mp_full, mp_dr = make_mp(d.train, d.R), make_mp(dr, d.R)
    keep_val_t, keep_test_t = torch.as_tensor(keep_val, device=device), torch.as_tensor(keep_test, device=device)
    forbidden = pair_key(d.train.src, d.train.dst, d.n).unique()

    method = args.unlearning_model
    if method == 'gnndelete':
        ni = tuple(torch.from_numpy(a).long().to(device) for a in ni_np)
        model, info, n_params = run_gnndelete(
            args, d, device, df, dr, mp_dr, mp_full, keep_val_t,
            torch.from_numpy(nodes1_np).to(device), torch.from_numpy(nodes2_np).to(device), sdf_np, ni, forbidden)
    elif method == 'retrain':
        model = build_model(args, d).to(device)
        info = train_link_prediction(model, d, dr, mp_dr, args.epochs, args.lr, args.valid_freq, keep_val_t, desc='retrain')
        n_params = count_trainable(model)
    elif method == 'gradient_ascent':
        model, info, n_params = run_gradient_ascent(args, d, device, df, dr, mp_dr, keep_val_t, forbidden)
    else:                                                              # 'original': no unlearning
        model, info, n_params = load_original(args, d, device, with_del=False), dict(train_time_sec=0.0), 0

    # ---- final evaluation on the test edges that were not deleted + the deleted edges --------------
    test, z = eval_lp(model, d, 'test', mp_dr, keep_test_t, per_relation=True)
    ed = eval_deleted(model, z, df, dr, repeats=args.df_auc_repeats)
    res = dict(method=method, edge_set=args.edge_set, gnn=args.gnn, seed=args.seed, df=args.df, df_size=args.df_size,
               alpha=args.alpha if method == 'gnndelete' else None, n_deleted_edges=int(df_np.sum()),
               n_train_edges=int(len(df_np)), n_sdf_nodes_1hop=int(nodes1_np.sum()), n_sdf_nodes_2hop=int(nodes2_np.sum()),
               et_auroc=test['auroc'], et_aupr=test['aupr'], ed_auroc=ed['auroc'], ed_aupr=ed['aupr'],
               mean_prob_deleted_edges=ed['mean_prob_deleted'], per_relation_et=test['per_relation'],
               trainable_params=n_params, args=vars(args), **info)

    # ---- membership-inference proxy (needs the original model) -------------------------------------
    if not args.no_mi:
        try:
            base = load_original(args, d, device, with_del=False).eval()
            with torch.no_grad():
                z0 = base(d.x, mp_full.ei, mp_full.et)
                g = torch.Generator(device='cpu').manual_seed(args.seed)
                sel = torch.randperm(dr.src.numel(), generator=g)[:20000].to(device)
                p_mem = base.decode(z0, dr.src[sel], dr.dst[sel], dr.et[sel]).sigmoid().cpu().numpy()
                tk = keep_test_t
                p_non = base.decode(z0, d.test.src[tk], d.test.dst[tk], d.test.et[tk]).sigmoid().cpu().numpy()
                p_before = base.decode(z0, df.src, df.dst, df.et).sigmoid().cpu().numpy()
                p_after = model.decode(z, df.src, df.dst, df.et).sigmoid().cpu().numpy()
            att = fit_mi_attacker(p_mem, p_non)
            res['mi_presence_before'], res['mi_presence_after'] = presence(att, p_before), presence(att, p_after)
            res['mi_ratio'] = res['mi_presence_before'] / max(res['mi_presence_after'], 1e-8)
        except (SystemExit, ValueError) as e:
            print('MI skipped:', e)

    tag = unlearn_tag(method, args.df, args.df_size, args.alpha, args.seed)
    path = results_path(args.root, args.edge_set, args.gnn, method, tag)
    save_json(path, res)
    if args.save_ckpt:
        ck = os.path.join(args.root, 'checkpoints', args.edge_set, args.gnn, method, tag)
        os.makedirs(ck, exist_ok=True)
        torch.save({'model_state': model.state_dict()}, os.path.join(ck, 'model_best.pt'))

    print(f'\n=== {method} | {args.gnn} | {args.edge_set} | df={args.df} {args.df_size:g} | seed {args.seed} ===')
    print(f'Et AUROC (test edges kept) : {res["et_auroc"]:.4f}')
    print(f'Ed AUROC (deleted edges)   : {res["ed_auroc"]:.4f}   (mean prob of deleted edges {ed["mean_prob_deleted"]:.3f})')
    if 'mi_ratio' in res:
        print(f'MI ratio (proxy)           : {res["mi_ratio"]:.3f}')
    print(f'unlearning time            : {info["train_time_sec"]:.1f}s | trainable params {n_params:,}')
    print('saved ->', path)


if __name__ == '__main__':
    main()
