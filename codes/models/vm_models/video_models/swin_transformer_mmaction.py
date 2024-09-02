from .my_mmaction2.models.swinvideo import SwinTransformer3D


def get_swinvideo(cfg):
    if 'tiny' in cfg.arch:
        model = SwinTransformer3D(arch='tiny',
                                  pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
                                  pretrained2d=True,
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  mlp_ratio=4.,
                                  qkv_bias=True,
                                  qk_scale=None,
                                  drop_rate=0.,
                                  attn_drop_rate=0.,
                                  drop_path_rate=0.1,
                                  patch_norm=True)

        return model
    else:
        raise NotImplementedError
