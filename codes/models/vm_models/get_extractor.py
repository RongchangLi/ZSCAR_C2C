import torch.hub

from .video_models.tsmresnet import tsmresnet
import torch.nn as nn


def get_video_extractor(cfg):
    if cfg.arch == 'tsmresnet18':
        if cfg.method in ['oadis','ade']:
            spatial_pool=False
        else:
            spatial_pool=True
        model = tsmresnet('tsmresnet18', shift_start=cfg.shift_start, num_segments=cfg.num_frames,
                          temporal_pool=cfg.temporal_pool,spatial_pool=spatial_pool)
        model.fc = nn.Sequential()
    elif cfg.arch == 'swintiny':
        from .video_models.swin_transformer_mmaction import get_swinvideo
        model = get_swinvideo(cfg)
    else:
        raise NotImplementedError

    return model
