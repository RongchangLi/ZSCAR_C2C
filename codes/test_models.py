import os
from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
import torch.multiprocessing
import numpy as np


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in test_stats:
        if key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)



def discrete_train_model(model, config,  val_dataset, test_dataset):

    model.eval()
    log_test = open(os.path.join(config.save_path, 'test_log.txt'), 'w')

    print("Evaluating test dataset:")
    loss_avg, val_result = evaluate(model, test_dataset, config)
    torch.save(model.state_dict(), os.path.join(
        config.save_path, f"best.pt"
    ))
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in val_result:
        if key in key_set:
            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
    log_test.write('\n')
    log_test.write(result)
    print("Loss average on test dataset: {}".format(loss_avg))
    log_test.write('\n')
    log_test.write("Loss average on test dataset: {}\n".format(loss_avg))