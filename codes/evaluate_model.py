import random
import os
import pickle

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import tqdm

import pprint

from opts import parser
from dataset.com_video_dataset import CompositionVideoDataset
from models.compositional_models import get_model
import test as test

from loss import *
from utils.my_lr_scheduler import WarmupCosineAnnealingLR

import yaml
import shutil
from utils.get_optimizer import get_optimizer
from utils import CosineAnnealingLR
from loss import KLLoss
import torch.multiprocessing
from train_models import regular_train_model, dere_train_model

torch.multiprocessing.set_sharing_strategy('file_system')


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def save_checkpoint(state, save_path, epoch, best=False):
#     filename = os.path.join(save_path, f"epoch_{epoch}_resume.pt")
#     torch.save(state, filename)
#
#
#
# def evaluate(model, dataset):
#     model.eval()
#     evaluator = test.Evaluator(dataset, model=None)
#     all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
#         model, dataset, config)
#     test_stats = test.test(
#         dataset,
#         evaluator,
#         all_logits,
#         all_attr_gt,
#         all_obj_gt,
#         all_pair_gt,
#         config
#     )
#     result = ""
#     key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
#     for key in test_stats:
#         if key in key_set:
#             result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
#     print(result)
#     model.train()
#     return loss_avg, test_stats


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(config.config, config)
    if config.framework == 'vlm':
        config.save_path = config.save_path
    else:
        config.save_path = config.save_path
    print(config)
    # set the seed value
    set_seed(config.seed)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    if os.path.exists(config.save_path):
        print('file already exists')
        print('exiting!')
        # exit(0)

    dataset_path = config.dataset_path

    train_dataset = CompositionVideoDataset(dataset_path,
                                            phase='train',
                                            split='compositional-split-natural',
                                            tdn_input='tdn' in config.arch)

    val_dataset = CompositionVideoDataset(dataset_path,
                                          phase='val',
                                          split='compositional-split-natural',
                                          tdn_input='tdn' in config.arch)

    test_dataset = CompositionVideoDataset(dataset_path,
                                           phase='test',
                                           split='compositional-split-natural',
                                           tdn_input='tdn' in config.arch)

    model = get_model(train_dataset, config)
    model = torch.nn.DataParallel(model).cuda()
    sd=torch.load(os.path.join(config.load_from, "epoch_49_resume.pt"))['state_dict']
    model.load_state_dict(sd)
    from train_models import evaluate
    evaluate(model, test_dataset, config)
    print("done!")
