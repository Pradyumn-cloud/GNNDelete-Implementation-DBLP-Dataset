"""Prepare the DBLP graph for the GNNDelete reproduction.

Faithful reuse of the official `prepare_dataset.py` for the DBLP graph path:

    1. Load CitationFull('DBLP', transform=NormalizeFeatures).
    2. For each official seed, run `train_test_split_edges_no_neg_adj_mask`
       (val_ratio=0.05, test_ratio=0.05) and save `data/DBLP/d_<seed>.pkl`
       as a (dataset, data) pair, exactly like the official repo.
    3. Compute the official easy/Out vs hard/In deletion masks from the
       2-hop local enclosing subgraph of the test edges, and save
       `data/DBLP/df_<seed>.pt` = {'out': out_mask, 'in': in_mask}.

Notes (scope/compat):
  - The official `process_graph` computes a `two_hop_degree` value for every
    edge, but it is ONLY used by the `ogbl` branch of the split. For DBLP the
    split is called without `two_hop_degree`, so skipping that computation
    does NOT change any saved artifact for DBLP. Skipped (scope simplification).
  - Masks are defined over the DIRECTED upper-triangle `train_pos_edge_index`
    (the same ordering later consumed by the unlearning scripts).
"""

import os
import math
import sys
import pickle

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import negative_sampling, k_hop_subgraph, is_undirected
from torch_geometric.seed import seed_everything

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

SEEDS = [42, 21, 13, 87, 100]


def train_test_split_edges_no_neg_adj_mask(data, val_ratio=0.05, test_ratio=0.1):
    '''Avoid adding neg_adj_mask. Verbatim from official prepare_dataset.py
    (graph / non-KG branch).'''

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = data.edge_weight = data.edge_year = data.edge_type = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]

    # Train
    r, c = row[n_v + n_t:], col[n_v + n_t:]

    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    assert not is_undirected(data.train_pos_edge_index)

    # Test
    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data


def process_dblp():
    d = 'DBLP'
    dataset = CitationFull(os.path.join(DATA_DIR, d), d, transform=T.NormalizeFeatures())

    print('Processing:', d)
    print(dataset)
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = None

    for s in SEEDS:
        seed_everything(s)

        data = dataset[0]
        data.train_mask = data.val_mask = data.test_mask = None
        data = train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.05)
        print(s, data)

        with open(os.path.join(DATA_DIR, d, f'd_{s}.pkl'), 'wb') as f:
            pickle.dump((dataset, data), f)

        # Two ways to sample Df from the training set:
        #   1. 'in'  : Df is within 2 hop local enclosing subgraph of Dtest (hard)
        #   2. 'out' : Df is outside of 2 hop local enclosing subgraph (easy)
        _, local_edges, _, mask = k_hop_subgraph(
            data.test_pos_edge_index.flatten().unique(),
            2,
            data.train_pos_edge_index,
            num_nodes=dataset[0].num_nodes)
        distant_edges = data.train_pos_edge_index[:, ~mask]
        print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

        in_mask = mask
        out_mask = ~mask

        torch.save(
            {'out': out_mask, 'in': in_mask},
            os.path.join(DATA_DIR, d, f'df_{s}.pt')
        )


def main():
    process_dblp()
    print('\nDBLP preparation finished.')


if __name__ == "__main__":
    sys.exit(main())