import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter

from utils.hsic import hsic_normalized_cca


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


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
    # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    # key_set = [ "attr_acc", "obj_acc",'attr_acc_open','obj_acc_open',"ub_seen","ub_unseen","ub_all","ub_open_seen","ub_open_unseen","ub_open_all","best_seen", "best_unseen", "best_hm","AUC"]
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        # if key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


# ========conditional train=
def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        for bid, batch in enumerate(train_dataloader):
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_img = batch[0].cuda()
            with torch.cuda.amp.autocast(enabled=True):
                p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = model(batch_img)
                # component loss
                loss_verb = Loss_fn(p_v * config.cosine_scale, batch_verb)
                loss_obj = Loss_fn(p_o * config.cosine_scale, batch_obj)
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                loss_com = Loss_fn(pred_com_train * config.cosine_scale, batch_target)
                loss = loss_com + 0.2 * (loss_verb + loss_obj)

                loss = loss / config.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)  # TODO:May be the reason for low acc on verb
                # scaler.step(prompt_optimizer)
                scaler.step(optimizer)
                scaler.update()

                # prompt_optimizer.zero_grad()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

            # break
        lr_scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        # if (i + 1) > config.val_epochs_ts:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))


def c2c_enhance(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    from utils.hsic import hsic_normalized_cca
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    v_o_on_v, v_o_on_o = cal_conditional(attr2idx, obj2idx, 'train', train_dataset)
    v_o_on_v, v_o_on_o = v_o_on_v.cuda(), v_o_on_o.cuda()

    # loss = loss_com
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []
        epoch_hsic_v_losses = []
        epoch_hsic_o_losses = []
        epoch_hsic_vo_losses = []
        epoch_con_train_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        for bid, batch in enumerate(train_dataloader):
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_img = batch[0].cuda()

            gama = 0.1

            with torch.cuda.amp.autocast(enabled=True):
                r = np.random.rand(1)
                if r < config.cutmix_prob:
                    lam = np.random.beta(config.beta, config.beta)
                    rand_index = torch.randperm(batch_verb.size()[0]).cuda()
                    target_o_a = batch_obj
                    target_o_b = batch_obj[rand_index]
                    target_v_a = batch_verb
                    target_v_b = batch_verb[rand_index]
                    target_a_a = batch_target
                    target_a_b = batch_target[rand_index]
                    # label adjustment-new combinations
                    target_all_a_c = target_v_a * len(obj2idx) + target_o_b
                    target_all_a_d = target_v_b * len(obj2idx) + target_o_a

                    bbx1, bby1, bbx2, bby2 = rand_bbox(batch_img.size(), lam)
                    batch_img[:, :, :, bbx1:bbx2, bby1:bby2] = batch_img[rand_index, :, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_img.size()[-1] * batch_img.size()[-2]))
                    p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = model(batch_img)

                    # component loss
                    loss_verb = Loss_fn(p_v * config.cosine_scale, target_v_a) * lam + Loss_fn(
                        p_v * config.cosine_scale, target_v_b) * (1.0 - lam)
                    loss_obj = Loss_fn(p_o * config.cosine_scale, target_o_a) * lam + Loss_fn(p_o * config.cosine_scale,
                                                                                              target_o_b) * (1.0 - lam)

                    # train only
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = p_pair_v[:, train_v_inds, train_o_inds] + p_pair_o[:, train_v_inds, train_o_inds]

                    loss_com_train = Loss_fn(pred_com_train * config.cosine_scale, target_a_a) * lam + Loss_fn(
                        pred_com_train * config.cosine_scale, target_a_b) * (1.0 - lam)
                    # extend to unseen world
                    pred_com_all = (p_pair_v + p_pair_o).reshape(batch_verb.size()[0], -1)
                    loss_com_all = Loss_fn(pred_com_all * config.cosine_scale, target_all_a_c) + Loss_fn(
                        pred_com_all * config.cosine_scale, target_all_a_d)

                    loss_com = loss_com_train + loss_com_all * gama

                    # hsic loss
                    obj_y = lam * F.one_hot(target_o_a.view(-1, 1), len(obj2idx))[:, 0] + (1.0 - lam) * F.one_hot(
                        target_o_b.view(-1, 1), len(obj2idx))[:, 0]
                    verb_y = lam * F.one_hot(target_v_a.view(-1, 1), len(attr2idx))[:, 0] + (1.0 - lam) * F.one_hot(
                        target_v_b.view(-1, 1), len(attr2idx))[:, 0]
                    vid_feat = vid_feat.mean(-1)
                    loss_hsic_v = hsic_normalized_cca(vid_feat, v_feat, 20) \
                                  - hsic_normalized_cca(v_feat, verb_y.float(), 20)
                    loss_hsic_o = hsic_normalized_cca(vid_feat, o_feat, 20) \
                                  - hsic_normalized_cca(o_feat, obj_y.float(), 20)
                    n_c = v_feat.shape[-1]
                    loss_hsic_vo = hsic_normalized_cca(v_feat[:, :int(n_c * 0.5)], o_feat[:, :int(n_c * 0.5)], 20)
                    loss_hsic = loss_hsic_v + loss_hsic_o + loss_hsic_vo

                    # condition loss
                    loss_con_train = torch.tensor([0.0]).cuda()

                else:
                    p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = model(batch_img)
                    # component loss
                    loss_verb = Loss_fn(p_v * config.cosine_scale, batch_verb)
                    loss_obj = Loss_fn(p_o * config.cosine_scale, batch_obj)
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                    loss_com_both = Loss_fn(pred_com_train * config.cosine_scale, batch_target)

                    loss_com = loss_com_both

                    # hsic loss
                    obj_y = F.one_hot(batch_obj.view(-1, 1), len(obj2idx))[:, 0]
                    verb_y = F.one_hot(batch_verb.view(-1, 1), len(attr2idx))[:, 0]
                    vid_feat = vid_feat.mean(-1)
                    loss_hsic_v = hsic_normalized_cca(vid_feat, v_feat, 20) \
                                  - hsic_normalized_cca(v_feat, verb_y.float(), 20)
                    loss_hsic_o = hsic_normalized_cca(vid_feat, o_feat, 20) \
                                  - hsic_normalized_cca(o_feat, obj_y.float(), 20)
                    n_c = v_feat.shape[-1]
                    loss_hsic_vo = hsic_normalized_cca(v_feat[:, :int(n_c * 0.5)], o_feat[:, :int(n_c * 0.5)], 20)
                    loss_hsic = loss_hsic_v + loss_hsic_o + loss_hsic_vo


                    # condition loss
                    p_o_con_v_mean = p_o_con_v.mean(0)
                    p_v_con_o_mean = p_v_con_o.mean(0)
                    #
                    loss_on_v = Loss_fn(p_o_con_v_mean, v_o_on_v)
                    loss_on_o = Loss_fn(p_v_con_o_mean.permute(1, 0), v_o_on_o.permute(1, 0))
                    loss_con_train = loss_on_o + loss_on_v

                loss = loss_com + 0.2 * (loss_verb + loss_obj) + 0.1 * loss_hsic + gama * loss_con_train

                loss = loss / config.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)  # TODO:May be the reason for low acc on verb
                # scaler.step(prompt_optimizer)
                scaler.step(optimizer)
                scaler.update()

                # prompt_optimizer.zero_grad()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())
            epoch_hsic_v_losses.append(loss_hsic_v.item())
            epoch_hsic_o_losses.append(loss_hsic_o.item())
            epoch_hsic_vo_losses.append(loss_hsic_vo.item())
            epoch_con_train_losses.append(loss_con_train.item())

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

            # break
        lr_scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")
        log_training.write(f"epoch {i + 1} hsic_v loss {np.mean(epoch_hsic_v_losses)}\n")
        log_training.write(f"epoch {i + 1} hsic_o loss {np.mean(epoch_hsic_o_losses)}\n")
        log_training.write(f"epoch {i + 1} hsic_vo loss {np.mean(epoch_hsic_vo_losses)}\n")
        log_training.write(f"epoch {i + 1} con_train loss {np.mean(epoch_con_train_losses)}\n")
        # log_training.write(f"epoch {i + 1} con_x loss {np.mean(epoch_con_x_losses)}\n")
        # log_training.write(f"epoch {i + 1} con_e loss {np.mean(epoch_con_e_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        # if (i + 1) > config.val_epochs_ts:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
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
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()

        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))
