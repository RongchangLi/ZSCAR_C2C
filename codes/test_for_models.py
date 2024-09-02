import random
import os
import pprint

from opts import parser
from models.compositional_models import get_model
from loss import *

import yaml
import torch.multiprocessing


torch.multiprocessing.set_sharing_strategy('file_system')

from train_models import evaluate


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
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
    config_path = os.path.join(config.logpath, 'config.yml')
    load_args(config_path, config)
    print(config)

    # set the seed value
    set_seed(config.seed)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Test details")
    pprint.pprint(config)

    dataset_path = config.dataset_path

    use_composed_pair_loss = True if config.method == 'oadis' else False

    if config.dataset=='sth-com':
        from dataset.com_video_dataset import CompositionVideoDataset
    else:
        raise NotImplemented
    train_dataset = CompositionVideoDataset(dataset_path,
                                            phase='train',
                                            split='compositional-split-natural',
                                            tdn_input='tdn' in config.arch,
                                            # aux_input=config.aux_input,
                                            use_composed_pair_loss=use_composed_pair_loss,
                                            frames_duration=config.num_frames)

    test_dataset = CompositionVideoDataset(dataset_path,
                                           phase='test',
                                           split='compositional-split-natural',
                                           tdn_input='tdn' in config.arch,
                                           frames_duration=config.num_frames)

    model = get_model(train_dataset, config)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(config.logpath, "best.pt")),strict=False)

    model.eval()
    log_test = open(os.path.join(config.logpath, 'test_log.txt'), 'w')

    print("Evaluating test dataset:")
    loss_avg, val_result = evaluate(model, test_dataset, config)
    result = ""
    key_set = [ "attr_acc", "obj_acc","ub_seen","ub_unseen","ub_all","best_seen", "best_unseen", "best_hm","AUC"]
    for key in key_set:
        # if key in key_set:
        result = result + key + "  " + str(round(val_result[key], 4)) + "| "
    log_test.write('\n')
    log_test.write(result)
    print("Loss average on test dataset: {}".format(loss_avg))
    log_test.write('\n')
    log_test.write("Loss average on test dataset: {}\n".format(loss_avg))
    print("done!")
