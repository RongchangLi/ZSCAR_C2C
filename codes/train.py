import random
import os
import pprint
from opts import parser
from models.compositional_models import get_model

from loss import *
from utils.my_lr_scheduler import WarmupCosineAnnealingLR

import yaml
import shutil
from utils.get_optimizer import get_optimizer
from utils import CosineAnnealingLR
from loss import KLLoss
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    seed = int(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(config.config, config)
    config.save_path = config.save_path + '/'
    print(config)
    # set the seed value
    set_seed(config.seed)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    i = 0
    temp_save_path = config.save_path + str(i)
    while os.path.exists(temp_save_path):
        i = i + 1
        print(f'file {temp_save_path} already exists')
        print('exiting!')
        temp_save_path = config.save_path + str(i)
    config.save_path = temp_save_path
    print(f'file {temp_save_path} ')

    dataset_path = config.dataset_path

    use_composed_pair_loss = True if config.method == 'oadis' else False
    if config.dataset == 'sth-com':
        from dataset.com_video_dataset import CompositionVideoDataset
    else:
        raise NotImplemented

    train_dataset = CompositionVideoDataset(dataset_path,
                                            phase='train',
                                            split='compositional-split-natural',
                                            tdn_input='tdn' in config.arch,
                                            aux_input=config.aux_input,
                                            ade_input=config.ade_input,
                                            frames_duration=config.num_frames,
                                            use_composed_pair_loss=use_composed_pair_loss)

    val_dataset = CompositionVideoDataset(dataset_path,
                                          phase='val',
                                          split='compositional-split-natural',
                                          tdn_input='tdn' in config.arch,
                                          frames_duration=config.num_frames,
                                          use_composed_pair_loss=use_composed_pair_loss
                                          )

    test_dataset = CompositionVideoDataset(dataset_path,
                                           phase='test',
                                           split='compositional-split-natural',
                                           tdn_input='tdn' in config.arch,
                                           frames_duration=config.num_frames,
                                           use_composed_pair_loss=use_composed_pair_loss)

    model = get_model(train_dataset, config)
    optimizer = get_optimizer(config, model)

    lr_scheduler = CosineAnnealingLR.WarmupCosineLR(optimizer=optimizer, milestones=[config.warmup, config.epochs],
                                                    warmup_iters=config.warmup, min_ratio=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if config.method == 'c2c_vanilla':
        from train_models import c2c_vanilla
        train_model = c2c_vanilla
    elif config.method == 'c2c_enhance':
        from train_models import c2c_enhance
        train_model = c2c_enhance
    else:
        raise NotImplementedError

    os.makedirs(config.save_path, exist_ok=True)
    model = torch.nn.DataParallel(model).cuda()
    if config.pretrain:
        model.load_state_dict(torch.load(config.load_model), strict=False)

    config_path = os.path.join(config.save_path, "config.yml")
    shutil.copyfile(config.config, config_path)

    shutil.copytree('./models', os.path.join(config.save_path, "models"))
    shutil.copy('./train_models.py', config.save_path)
    train_model(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler)

    print("done!")
