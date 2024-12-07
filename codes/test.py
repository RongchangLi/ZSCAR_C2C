import argparse
import copy
import json
import os
from itertools import product

import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2

from utils import *
from loss import loss_calu
from torch.nn.modules.loss import CrossEntropyLoss
# from parameters import parser, YML_PATH
# from dataset import CompositionDataset
from dataset.com_video_dataset import CompositionVideoDataset
from models.compositional_models import get_model
# from model.dfsp import DFSP
from opts import parser
import yaml

cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"


class Evaluator:
    """
    Evaluator class, adapted from:
    https://github.com/Tushar-N/attributes-as-operators

    With modifications from:
    https://github.com/ExplainableML/czsl
    """

    def __init__(self, dset, model):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe',
        # 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)


        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs+dset.ex_test_pairs)
            test_pair_gt = set(dset.test_pairs+dset.ex_test_pairs)

        self.test_pair_dict = [
            (dset.attr2idx[attr],
             dset.obj2idx[obj]) for attr,obj in test_pair_gt]

        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle
        # setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(
                topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension #seen_mask.sum()=1717
        scores[~mask] += bias  # Add bias to test pairs(only account unseen)

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})# only account unseen
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        ) #only account unseen
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1) #mask= train_pair+unseen_pair(in test)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 # only consider the train and test pairs; when val and test have overlapped categories, it is also okay.
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10  # only consider the train and test pairs
        results.update({"closed": get_pred_from_scores(closed_scores, topk)}) #
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)} #unbiased_closed, used to calculate the
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        # Return only attributes that are in our pairs
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias  # Add bias to test pairs#add 3451 positions

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()  # totally 4287 positions
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k
        # largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                              self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (
                    attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                    obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            # Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])# only unseen
        unbiased_closed = _process(predictions["unbiased_closed"]) # unseen and seen (only train and test)
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        unbiased_open = _process(predictions["unbiased_open"]) # unseen and seen (only train and test)
        _add_to_dict(unbiased_open, "open_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
                              0][:, topk - 1] #

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        # base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats


def  predict_logits(model, dataset, config):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    # print(text_rep.shape)
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    loss = 0
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        for idx, data in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            batch_img = data[0].cuda()
            batch_target = data[3].cuda()
            # if config.framework == 'vlm':

            predict = model(batch_img, pairs.repeat(torch.cuda.device_count(), 1))  # TODO Using nagetive sample
            # else:
            #     predict = model(batch_img, pairs.repeat(torch.cuda.device_count(),1))

            logits = predict
            # print(logits.shape)
            # print(batch_target.shape)
            loss += loss_fn(predict, batch_target)

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss / len(dataloader)


def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasiblity=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasiblity (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score


def test(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=1e3, topk=1
    )

    attr_acc = float(torch.mean(
        (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean(
        (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    attr_acc_open = float(torch.mean(
        (results['unbiased_open'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc_open = float(torch.mean(
        (results['unbiased_open'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=1,
    )

    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc
    stats['attr_acc_open'] = attr_acc_open
    stats['obj_acc_open'] = obj_acc_open

    stats['ub_seen']=stats['closed_ub_seen_match']
    stats['ub_unseen']=stats['closed_ub_unseen_match']
    stats['ub_all']=stats['closed_ub_match']

    stats['ub_open_seen']=stats['open_ub_seen_match']
    stats['ub_open_unseen']=stats['open_ub_unseen_match']
    stats['ub_open_all']=stats['open_ub_match']
    return stats


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(config.config, config)
    print(config)

    # set the seed value

    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")

    dataset_path = config.dataset_path

    print('loading validation dataset')
    val_dataset = CompositionVideoDataset(dataset_path,
                                          phase='val',
                                          split='compositional-split-natural',
                                          open_world=config.open_world)

    print('loading test dataset')
    test_dataset = CompositionVideoDataset(dataset_path,
                                           phase='test',
                                           split='compositional-split-natural',
                                           open_world=config.open_world)

    allattrs = val_dataset.attrs
    allobj = val_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(val_dataset, config)
    model.load_state_dict(torch.load(config.load_model))

    print('evaluating on the validation set')
    if config.open_world and config.threshold is None:
        evaluator = Evaluator(val_dataset, model=None)
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        seen_mask = val_dataset.seen_mask.to('cpu')
        min_feasibility = (unseen_scores + seen_mask * 10.).min()
        max_feasibility = (unseen_scores - seen_mask * 10.).max()
        thresholds = np.linspace(
            min_feasibility,
            max_feasibility,
            num=config.threshold_trials)
        best_auc = 0.
        best_th = -10
        val_stats = None
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits(
                model, val_dataset, device, config)
            for th in thresholds:
                temp_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
                results = test(
                    val_dataset,
                    evaluator,
                    temp_logits,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config
                )
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_th = th
                    print('New best AUC', best_auc)
                    print('Threshold', best_th)
                    val_stats = copy.deepcopy(results)
    else:
        best_th = config.threshold
        evaluator = Evaluator(val_dataset, model=None)
        # feasibility_path = os.path.join(
        #     DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        # unseen_scores = torch.load(
        #     feasibility_path,
        #     map_location='cpu')['feasibility']
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits(
                model, val_dataset, config)
            if config.open_world:
                # print('using threshold: ', best_th)
                # all_logits = threshold_with_feasibility(
                #     all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
                pass
            results = test(
                val_dataset,
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )
        val_stats = copy.deepcopy(results)
        result = ""
        for key in val_stats:
            result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
        print(result)

    print('evaluating on the test set')
    with torch.no_grad():
        evaluator = Evaluator(test_dataset, model=None)
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits(
            model, test_dataset, config)
        # if config.open_world and best_th is not None:
        #     print('using threshold: ', best_th)
        #     all_logits = threshold_with_feasibility(
        #         all_logits,
        #         test_dataset.seen_mask,
        #         threshold=best_th,
        #         feasiblity=unseen_scores)
        test_stats = test(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )

        result = ""
        for key in test_stats:
            result = result + key + "  " + \
                     str(round(test_stats[key], 4)) + "| "
        print(result)

    results = {
        'val': val_stats,
        'test': test_stats,
    }

    if best_th is not None:
        results['best_threshold'] = best_th

    if config.open_world:
        result_path = config.load_model[:-2] + "open.calibrated.json"
    else:
        result_path = config.load_model[:-2] + "closed.json"

    with open(result_path, 'w+') as fp:
        json.dump(results, fp)

    print("done!")
