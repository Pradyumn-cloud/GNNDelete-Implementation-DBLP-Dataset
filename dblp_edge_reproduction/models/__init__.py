from .gcn import GCN
from .deletion import GCNDelete


def get_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):
    '''Local mirror of official `framework.get_model`, restricted to GCN.'''

    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete}
    else:
        model_mapping = {'gcn': GCN}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)