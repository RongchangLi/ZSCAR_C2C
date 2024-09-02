import torch
def get_optimizer_vm(cfg,model):
    comp_param=[]
    video_en_param=[]
    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            video_en_param.append(param)
        else:
            comp_param.append(param)
    optimizer = torch.optim.Adam([
        {'params': comp_param, 'lr': cfg.com_lr,'weight_decay': cfg.com_wd},
        {'params': video_en_param, 'lr': cfg.ve_lr,'weight_decay': cfg.ve_wd}],
        lr=cfg.ve_lr, eps=1e-8,weight_decay=cfg.ve_wd)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    return optimizer


def get_optimizer(cfg,model):
    if cfg.framework=='vm':
        return get_optimizer_vm(cfg,model)