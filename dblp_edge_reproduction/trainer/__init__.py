from .base import Trainer, KGTrainer
from .gnndelete import GNNDeleteTrainer
from .retrain import RetrainTrainer


trainer_mapping = {
    'original': Trainer,
    'retrain': RetrainTrainer,
    'gnndelete': GNNDeleteTrainer,
    'gnndelete_mse': GNNDeleteTrainer,
    'gnndelete_kld': GNNDeleteTrainer,
    'gnndelete_cosine': GNNDeleteTrainer,
}


def get_trainer(args):
    return trainer_mapping[args.unlearning_model](args)