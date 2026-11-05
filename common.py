"""Shared helpers: data loading, same-scene negative sampling, evaluation, training loop."""
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import trange

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pair_key(u, v, n):
    return torch.minimum(u, v) * n + torch.maximum(u, v)


def _long(a):
    return torch.from_numpy(np.asarray(a).astype(np.int64))


def _edges(s, name):
    return SimpleNamespace(src=_long(s[f'{name}_src']), dst=_long(s[f'{name}_dst']), et=_long(s[f'{name}_et']))


def load_data(data_dir, edge_set, seed):
    """Everything stays on CPU here; call to_device() afterwards."""
    folder = os.path.join(data_dir, edge_set)
    if not os.path.exists(os.path.join(folder, f'split_{seed}.npz')):
        raise SystemExit(f'{folder}/split_{seed}.npz not found. Run prepare_3dssg.py first.')
    g = np.load(os.path.join(folder, 'graph.npz'), allow_pickle=True)
    s = np.load(os.path.join(folder, f'split_{seed}.npz'))
    d = SimpleNamespace()
    d.x = torch.from_numpy(g['x']).float()
    d.n = d.x.shape[0]
    d.R = len(g['rel_names'])
    d.rel_names = [str(r) for r in g['rel_names']]
    d.node_scan_np = np.asarray(g['node_scan']).astype(np.int64)
    d.node_scan = _long(g['node_scan'])
    ptr = _long(g['scan_ptr'])
    d.node_start = ptr[d.node_scan]
    d.node_size = (ptr[1:] - ptr[:-1])[d.node_scan]
    d.train, d.val, d.test = _edges(s, 'train'), _edges(s, 'val'), _edges(s, 'test')
    d.val.neg_dst, d.test.neg_dst = _long(s['val_neg_dst']), _long(s['test_neg_dst'])
    d.mask_in, d.mask_out = np.asarray(s['mask_in']), np.asarray(s['mask_out'])
    return d


def to_device(obj, device):
    """Recursively move every tensor in a SimpleNamespace to `device`."""
    for k, v in vars(obj).items():
        if torch.is_tensor(v):
            setattr(obj, k, v.to(device))
        elif isinstance(v, SimpleNamespace):
            to_device(v, device)
    return obj


def subset(edges, mask):
    mask = torch.as_tensor(mask, device=edges.src.device)
    return SimpleNamespace(src=edges.src[mask], dst=edges.dst[mask], et=edges.et[mask])


def make_mp(edges, R):
    """Message-passing graph: both directions; reverse edges get relation id r + R."""
    ei = torch.stack([torch.cat([edges.src, edges.dst]), torch.cat([edges.dst, edges.src])])
    et = torch.cat([edges.et, edges.et + R])
    return SimpleNamespace(ei=ei, et=et)


@torch.no_grad()
def sample_neg_dst(src, d, forbidden, tries=6):
    """Corrupt the tail of every positive INSIDE its scene. Returns neg dst (-1 = no valid one)."""
    start, size = d.node_start[src], d.node_size[src]
    neg = torch.full_like(src, -1)
    todo = torch.arange(src.numel(), device=src.device)
    for _ in range(tries):
        if todo.numel() == 0:
            break
        r = torch.rand(todo.numel(), device=src.device)
        cand = start[todo] + torch.minimum((r * size[todo]).long(), size[todo] - 1)
        bad = torch.isin(pair_key(src[todo], cand, d.n), forbidden) | (cand == src[todo])
        neg[todo[~bad]] = cand[~bad]
        todo = todo[bad]
    return neg


@torch.no_grad()
def eval_lp(model, d, split, mp, keep=None, per_relation=False):
    """Link-prediction AUROC/AUPR on the val/test positives vs. their same-scene negatives."""
    model.eval()
    z = model(d.x, mp.ei, mp.et)
    sp = getattr(d, split)
    s, o, r, neg = sp.src, sp.dst, sp.et, sp.neg_dst
    if keep is not None:
        keep = torch.as_tensor(keep, device=s.device)
        s, o, r, neg = s[keep], o[keep], r[keep], neg[keep]
    if s.numel() == 0:
        return dict(loss=float('nan'), auroc=float('nan'), aupr=float('nan')), z
    pos_l, neg_l = model.decode(z, s, o, r), model.decode(z, s, neg, r)
    logits = torch.cat([pos_l, neg_l])
    label = torch.cat([torch.ones_like(pos_l), torch.zeros_like(neg_l)])
    prob = logits.sigmoid().cpu().numpy()
    y = label.cpu().numpy()
    out = dict(loss=F.binary_cross_entropy_with_logits(logits, label).item(),
               auroc=float(roc_auc_score(y, prob)), aupr=float(average_precision_score(y, prob)))
    if per_relation:
        rr, k = r.cpu().numpy(), len(r)
        pr = {}
        for t in np.unique(rr):
            m = rr == t
            if m.sum() >= 20:
                yy = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
                pr[d.rel_names[t]] = dict(n=int(m.sum()), auroc=float(roc_auc_score(yy, np.r_[prob[:k][m], prob[k:][m]])))
        out['per_relation'] = pr
    return out, z


@torch.no_grad()
def eval_deleted(model, z, df, dr, repeats=50, seed=0):
    """Paper metric Ed: deleted edges are label 0, an equal number of REMAINING edges label 1."""
    p_df = model.decode(z, df.src, df.dst, df.et).sigmoid().cpu().numpy()
    p_dr = model.decode(z, dr.src, dr.dst, dr.et).sigmoid().cpu().numpy()
    rng = np.random.default_rng(seed)
    k = min(len(p_df), len(p_dr))
    aucs, aups = [], []
    for _ in range(repeats):
        a = rng.choice(len(p_df), k, replace=False)
        b = rng.choice(len(p_dr), k, replace=False)
        y = np.r_[np.zeros(k), np.ones(k)]
        p = np.r_[p_df[a], p_dr[b]]
        aucs.append(roc_auc_score(y, p))
        aups.append(average_precision_score(y, p))
    return dict(auroc=float(np.mean(aucs)), aupr=float(np.mean(aups)),
                mean_prob_deleted=float(p_df.mean()))


def count_trainable(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_link_prediction(model, d, edges, mp, epochs, lr, valid_freq, keep_val=None, desc='train'):
    """Full-batch BCE link prediction with same-scene negatives (used for the original model
    and for Retrain).  `edges` = positives = the graph the model is allowed to see.
    Keeps the weights with the lowest validation loss."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    forbidden = pair_key(edges.src, edges.dst, d.n).unique()
    best, best_state, best_epoch = float('inf'), None, -1
    t0 = time.time()
    for epoch in trange(epochs, desc=desc):
        model.train()
        z = model(d.x, mp.ei, mp.et)
        neg = sample_neg_dst(edges.src, d, forbidden)
        ok = neg >= 0
        pos_l = model.decode(z, edges.src, edges.dst, edges.et)
        neg_l = model.decode(z, edges.src[ok], neg[ok], edges.et[ok])
        logits = torch.cat([pos_l, neg_l])
        label = torch.cat([torch.ones_like(pos_l), torch.zeros_like(neg_l)])
        loss = F.binary_cross_entropy_with_logits(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % valid_freq == 0 or epoch == epochs - 1:
            m, _ = eval_lp(model, d, 'val', mp, keep_val)
            print(f'  epoch {epoch + 1:5d} | train loss {loss.item():.4f} | val loss {m["loss"]:.4f} | val AUROC {m["auroc"]:.4f}')
            if m['loss'] < best:
                best, best_epoch = m['loss'], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch + 1, best_val_loss=best, train_time_sec=time.time() - t0)


# ------------------------------------------------------------------------------------------
# Membership-inference proxy (NOT the official MLP attacker of the paper, see README)
# ------------------------------------------------------------------------------------------
def _mi_feat(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.stack([p, np.log(p / (1 - p))], axis=1)


def fit_mi_attacker(p_members, p_nonmembers):
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([_mi_feat(p_members), _mi_feat(p_nonmembers)])
    y = np.r_[np.ones(len(p_members)), np.zeros(len(p_nonmembers))]
    return LogisticRegression(max_iter=1000).fit(X, y)


def presence(attacker, p):
    """Mean probability that the attacker believes the edges were in the training set."""
    return float(attacker.predict_proba(_mi_feat(p))[:, 1].mean())


def results_path(root, edge_set, gnn, method, tag):
    p = os.path.join(root, 'results', edge_set, gnn, method)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f'{tag}.json')


def unlearn_tag(method, df, df_size, alpha, seed):
    if method == 'gnndelete':
        return f'{df}-{df_size:g}-a{alpha:g}-{seed}'
    return f'{df}-{df_size:g}-{seed}'


def _json_default(o):
    return o.item() if hasattr(o, 'item') else str(o)


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=_json_default)
