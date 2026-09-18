'''
GNNDelete deletion operators.

Copy of the official `framework/models/deletion.py` restricted to the
DBLP + GCN reproduction scope:
  - kept: DeletionLayer, DeletionLayerKG, GCNDelete
  - dropped: GATDelete, GINDelete, RGCNDelete, RGATDelete (not in scope)
  - import change only: `from . import GCN, GAT, ...` -> `from .gcn import GCN`
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .gcn import GCN


class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        # paper-vs-repo fix (C-class): start DEL as the IDENTITY map.
        # The released repo initialized deletion_weight to ones/1000, a rank-1
        # projection that destroys the S_Df node embeddings at epoch 0 (test
        # dt_auc drops to 0.70 BEFORE any training). The paper's DEL is a
        # perturbation around the identity. See MODERNIZATION.md entry #12.
        self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        # init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class GCNDelete(GCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        # Freeze the GCN. (Compatibility: the official `self.convX.requires_grad
        # = False` is a NO-OP in torch >= 2.8, so freeze at the parameter level.
        # Same effect: the optimizer also only ever updates `deletion*`, so the
        # GCN weights never change during unlearning.)
        for p in self.conv1.parameters():
            p.requires_grad_(False)
        for p in self.conv2.parameters():
            p.requires_grad_(False)

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.conv1(x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)