"""GCN / R-GCN backbones with an optional GNNDelete deletion operator (DEL).

Both models share one interface:
    z = model(x, edge_index, edge_type)            # node embeddings
    logit = model.decode(z, src, dst, rel)         # link score
GCN ignores the relation (dot product); R-GCN uses relation-specific weights and a DistMult
decoder.  With_del=True adds one DeletionLayer after each GNN layer, applied only to the nodes
in `mask1` (layer 1, 1-hop S_Df) / `mask2` (layer 2, 2-hop S_Df).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv


class DeletionLayer(nn.Module):
    """Learnable d x d matrix, shared by all nodes, applied only to the masked nodes.
    Starts as the IDENTITY (the official repo's ones/1000 init collapses utility, see the
    DBLP reproduction README)."""

    def __init__(self, dim):
        super().__init__()
        self.deletion_weight = nn.Parameter(torch.eye(dim))

    def forward(self, x, mask):
        if mask is None:
            return x
        out = x.clone()
        out[mask] = out[mask] @ self.deletion_weight
        return out


class _Base(nn.Module):
    with_del = False
    mask1 = None
    mask2 = None

    def _del(self, h, layer):
        if not self.with_del:
            return h
        return self.deletion1(h, self.mask1) if layer == 1 else self.deletion2(h, self.mask2)

    def freeze_base(self):
        """Freeze everything except the deletion operator."""
        for n, p in self.named_parameters():
            p.requires_grad_('deletion' in n)


class GCN(_Base):
    def __init__(self, in_dim, hidden, out, R, with_del=False, **kw):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hidden), GCNConv(hidden, out)
        self.with_del = with_del
        if with_del:
            self.deletion1, self.deletion2 = DeletionLayer(hidden), DeletionLayer(out)

    def forward(self, x, edge_index, edge_type=None):
        h = self._del(self.conv1(x, edge_index), 1)
        return self._del(self.conv2(F.relu(h), edge_index), 2)

    def decode(self, z, s, o, r=None):
        return (z[s] * z[o]).sum(-1)


class RGCN(_Base):
    """x -> Linear(in, hidden) -> RGCNConv(hidden, hidden) -> RGCNConv(hidden, out).
    The input projection keeps the per-relation aggregation cheap (hidden-dim instead of the
    ~600-dim raw features).  2R relation ids = R forward + R reverse."""

    def __init__(self, in_dim, hidden, out, R, with_del=False, num_bases=0, **kw):
        super().__init__()
        nb = num_bases if num_bases and num_bases > 0 else None
        self.proj = nn.Linear(in_dim, hidden)
        self.conv1 = RGCNConv(hidden, hidden, 2 * R, num_bases=nb)
        self.conv2 = RGCNConv(hidden, out, 2 * R, num_bases=nb)
        self.rel = nn.Parameter(torch.ones(R, out))          # DistMult relation vectors
        self.with_del = with_del
        if with_del:
            self.deletion1, self.deletion2 = DeletionLayer(hidden), DeletionLayer(out)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.proj(x))
        h = self._del(self.conv1(h, edge_index, edge_type), 1)
        return self._del(self.conv2(F.relu(h), edge_index, edge_type), 2)

    def decode(self, z, s, o, r):
        return (z[s] * self.rel[r] * z[o]).sum(-1)


def build_model(args, d, with_del=False):
    cls = {'gcn': GCN, 'rgcn': RGCN}[args.gnn]
    return cls(d.x.shape[1], args.hidden_dim, args.out_dim, d.R, with_del=with_del,
               num_bases=getattr(args, 'num_bases', 0))
