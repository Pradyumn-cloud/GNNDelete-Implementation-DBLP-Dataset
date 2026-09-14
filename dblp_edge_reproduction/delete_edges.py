'''Edge deletion (unlearning) for DBLP + GCN with GNNDelete.

Faithful adaptation of the official `delete_gnn.py` restricted to the
DBLP + GCN + edge-deletion scope:

  - `from framework import get_model, get_trainer`        -> local models/trainer
  - `from framework.training_args import parse_args`      -> local training_args
  - `from framework.utils import *`                       -> dropped (unused here)
  - `from train_mi import MLPAttacker`                    -> dropped (MI not in scope)
  - `torch.autograd.set_detect_anomaly(True)`             -> dropped (debug-only,
    slows training; no anomaly is expected in the official code path)
  - the official retrain-reference lookup uses a hardcoded `'checkpoint'` dir
    string; here it uses the base `--checkpoint_dir` (same relative layout)

Everything that defines the GNNDelete algorithm (E_d selection, E_r masks,
1/2-hop S_Df masks, GCNDelete construction, frozen GCN / trainable DEL,
DEC + NI training, evaluation) is verbatim.

Usage (first smoke test, official easy/Out sampling):
    python delete_edges.py --lr 1e-3 --epochs 1500 --dataset DBLP \
        --random_seed 42 --unlearning_model gnndelete --gnn gcn \
        --df out --df_size 10
'''

import os
import copy
import json
import wandb
import pickle
import argparse
import torch
from torch_geometric.utils import to_undirected, k_hop_subgraph, is_undirected
from torch_geometric.seed import seed_everything

from models import get_model
from trainer import get_trainer
from training_args import parse_args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    checkpoint_base = args.checkpoint_dir          # keep base for retrain lookup
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    seed_everything(args.random_seed)

    if 'gnndelete' in args.unlearning_model:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
            '-'.join([str(i) for i in [args.loss_fct, args.loss_type, args.alpha, args.neg_sample_random]]),
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    else:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print('Checkpoint dir:', args.checkpoint_dir)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features

    print('Training args', args)
    wandb.init(config=args)

    # Df and Dr
    assert args.df != 'none'

    if args.df_size >= 100:     # df_size is number of nodes/edges to be deleted
        df_size = int(args.df_size)
    else:                       # df_size is the ratio
        df_size = int(args.df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')

    df_mask_all = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))[args.df]
    df_nonzero = df_mask_all.nonzero().squeeze()

    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    print('Deleting the following edges:', df_global_idx)

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # --- Checkpoint C: E_d and E_r are disjoint by construction ---
    assert (df_mask & dr_mask).sum() == 0
    print(f'[Checkpoint C] |E_d| = {df_mask.sum().item():,}   '
          f'|E_r| = {dr_mask.sum().item():,}   E_d & E_r overlap = {(df_mask & dr_mask).sum().item()}')

    # For testing
    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]

    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        1,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    # --- Checkpoint D: affected 1-hop / 2-hop node counts ---
    print(f'[Checkpoint D] S_Df nodes 1-hop = {sdf_node_1hop.sum().item():,}   2-hop = {sdf_node_2hop.sum().item():,}   '
          f'(deleted-node seeds = {data.train_pos_edge_index[:, df_mask].flatten().unique().numel():,})')

    # To undirected for message passing
    assert not is_undirected(data.train_pos_edge_index)

    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
    two_hop_mask = two_hop_mask.bool()
    df_mask = df_mask.bool()
    dr_mask = ~df_mask

    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    print('Undirected dataset:', data)

    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask

    # Model
    model = get_model(args, sdf_node_1hop, sdf_node_2hop, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)

    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))
            if logits_ori is not None:
                logits_ori = logits_ori.to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)

    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)

    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])

    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])

    optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)

    # --- Checkpoint E: original GCN frozen, DEL trainable ---
    print('[Checkpoint E] parameter requires_grad flag:')
    for n, p in model.named_parameters():
        print(f'    {n:<16s} requires_grad = {p.requires_grad}')

    wandb.watch(model, log_freq=100)

    attack_model_all = None
    attack_model_sub = None

    # Train
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            checkpoint_base, args.dataset, args.gnn, 'retrain',
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]),
            'model_best.pt')
        if os.path.exists(retrain_path):
            retrain_ckpt = torch.load(retrain_path, map_location=device)
            retrain_args = copy.deepcopy(args)
            retrain_args.unlearning_model = 'retrain'
            retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            retrain.load_state_dict(retrain_ckpt['model_state'])
            retrain = retrain.to(device)
            retrain.eval()
        else:
            retrain = None
    else:
        retrain = None

    test_results = trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    print(test_results[-1])
    trainer.save_log()


if __name__ == "__main__":
    main()