"""Train the original 2-layer GCN on DBLP for link prediction.

Faithful adaptation of the official `train_gnn.py` for DBLP + GCN:

    1. Load the prepared `data/DBLP/d_<seed>.pkl`.
    2. Convert `train_pos_edge_index` to undirected (official step).
    3. Build the official GCN (2 layers, in->128->64).
    4. Train with the official full-batch link-prediction loss
       (BCE over positive + negative sampled edges).
    5. Save `checkpoints/.../model_best.pt`, `node_embeddings.pt`,
       `training_args.json`, `trainer_log.json`.
    6. Report test-edge AUROC (this is Checkpoint B / the unlearning baseline).

Only the checkpoint/data directories differ from the official `train_gnn.py`;
the model, loss, sampling and evaluation logic are unchanged.
"""

import os
import sys
import pickle
import wandb

import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected

from trainer.base import Trainer
from models.gcn import GCN
from training_args import parse_args

ROOT = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    # Keep all defaults; force reproduction settings sane for the DBLP run.
    if args.checkpoint_dir == './checkpoints':
        args.checkpoint_dir = os.path.join(ROOT, 'checkpoints')
    if args.data_dir == './data':
        args.data_dir = os.path.join(ROOT, 'data')

    args.unlearning_model = 'original'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    args.in_dim = dataset.num_features

    wandb.init(config=args, mode=os.environ.get('WANDB_MODE', 'offline'))

    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)

    # To undirected
    train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index = train_pos_edge_index
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    assert is_undirected(data.train_pos_edge_index)

    print('Undirected dataset:', data)

    # Model
    model = GCN(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    trainer = Trainer(args)
    trainer.train(model, data, optimizer, args)

    # Test
    loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = trainer.test(model, data)
    print(f'\n=== CHECKPOINT B (original GCN baseline) ===')
    print(f'  test loss       : {loss:.4f}')
    print(f'  test dt_auc     : {dt_auc:.4f}  (retained/test-edge AUROC)')
    print(f'  test dt_aup     : {dt_aup:.4f}')
    print(f'  test df_auc     : {df_auc}')
    print(f'  test df_aup     : {df_aup}')
    trainer.save_log()


if __name__ == "__main__":
    sys.exit(main())