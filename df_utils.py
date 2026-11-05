"""numpy-only helpers for choosing the deletion set Df and the GNNDelete subgraph S_Df."""
import numpy as np

from prepare_3dssg import khop_nodes


def select_df(mode, size, rng, src, dst, mask_in, mask_out, node_scan):
    """Choose Df among the canonical train edges.

    mode 'out' / 'in' : random edges from the OUT / IN pool. size < 100 -> percent of train
                        edges, size >= 100 -> absolute number (same rule as the DBLP repo).
    mode 'scene'      : ALL train edges of `size` random scenes.
    mode 'node'       : ALL train edges touching `size` random nodes.
    Returns (df_mask, extra); `extra` lists deleted scenes / nodes so that val/test edges that
    belong to them can be excluded from the utility (Et) evaluation.
    """
    E = len(src)
    extra = {}
    if mode in ('out', 'in'):
        pool = np.nonzero(mask_out if mode == 'out' else mask_in)[0]
        k = int(size) if size >= 100 else int(round(size / 100 * E))
        if k < 1 or k > len(pool):
            raise SystemExit(f'df={mode}: asked for {k} edges but the pool has {len(pool)}')
        df = np.zeros(E, dtype=bool)
        df[rng.choice(pool, k, replace=False)] = True
    elif mode == 'scene':
        scans = np.unique(node_scan[src])
        chosen = rng.choice(scans, int(size), replace=False)
        df = np.isin(node_scan[src], chosen)
        extra['deleted_scans'] = chosen
    elif mode == 'node':
        nodes = np.unique(np.concatenate([src, dst]))
        chosen = rng.choice(nodes, int(size), replace=False)
        df = np.isin(src, chosen) | np.isin(dst, chosen)
        extra['deleted_nodes'] = chosen
    else:
        raise ValueError(mode)
    return df, extra


def keep_mask(s, o, neg, node_scan, extra):
    """Val/test positives (and their aligned negatives) that are NOT part of what was deleted."""
    keep = np.ones(len(s), dtype=bool)
    if 'deleted_scans' in extra:
        keep &= ~np.isin(node_scan[s], extra['deleted_scans'])
    if 'deleted_nodes' in extra:
        keep &= ~(np.isin(s, extra['deleted_nodes']) | np.isin(o, extra['deleted_nodes'])
                  | np.isin(neg, extra['deleted_nodes']))
    return keep


def sdf_sets(n, src, dst, df_mask):
    """1-hop / 2-hop node masks around the deleted edges + the train edges inside the 2-hop
    subgraph (these are the edges GNNDelete uses for message passing while training DEL).
    Hops are taken on the UNDIRECTED train graph."""
    seeds = np.unique(np.concatenate([src[df_mask], dst[df_mask]]))
    nodes1 = khop_nodes(n, src, dst, seeds, 1)
    nodes2 = khop_nodes(n, src, dst, seeds, 2)
    sdf_edges = nodes2[src] & nodes2[dst]
    return nodes1, nodes2, sdf_edges


def build_ni_pairs(nodes2, node_scan, df_src, df_dst, n):
    """All same-scene node pairs inside S_Df (minus the deleted pairs themselves).

    The paper/DBLP code uses ALL pairs of S_Df nodes through a dense N x N matrix.  Scenes are
    disconnected, so cross-scene pairs carry no information and, with ~34k nodes, a dense
    matrix would not fit on a laptop.  Restricting to same-scene pairs keeps the objective
    (Neighborhood Influence on pair probabilities) and makes it cheap."""
    nodes = np.nonzero(nodes2)[0]
    sc = node_scan[nodes]
    bounds = np.concatenate([[0], np.nonzero(np.diff(sc))[0] + 1, [len(nodes)]])
    I, J = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        iu = np.triu_indices(b - a, 1)
        I.append(nodes[a:b][iu[0]])
        J.append(nodes[a:b][iu[1]])
    if not I:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    I, J = np.concatenate(I), np.concatenate(J)
    keys = np.minimum(I, J) * n + np.maximum(I, J)
    dfk = np.minimum(df_src, df_dst) * n + np.maximum(df_src, df_dst)
    ok = ~np.isin(keys, dfk)
    return I[ok], J[ok]
