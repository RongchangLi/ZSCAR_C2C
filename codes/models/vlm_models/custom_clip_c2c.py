import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from models.vlm_models.text_learner import get_text_learner

import torch.nn.functional as F

from einops import rearrange

_tokenizer = _Tokenizer()


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class MLP_ST(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            # mod.append(nn.Linear(incoming, outgoing, bias=bias))
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        # mod.append(nn.Linear(incoming, out_dim, bias=bias))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        for block in self.transformer.resblocks:
            block.attn_mask = block.attn_mask[:cfg.ctx_length, :cfg.ctx_length]
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):  # have been added positional emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj

        self.num_frames=cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)#be annotated


        return out


class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        """
        Using component to deduce the composition, without composition
        """
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        # self.dtype = clip_model.dtype

        # ======== C2C part =====
        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = []
        for a in fc_emb:
            a = int(a)
            layers.append(a)

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False,
                           norm=True, layers=layers)

        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False,
                           norm=True, layers=layers)

        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False,
                              norm=True, layers=layers)

        self.c2c_VE2 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False,
                              norm=True, layers=layers)


        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)  # TODO
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)  # TODO

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)  # TODO
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)  # TODO

    def forward(self, video, pairs=None):
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features=self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features=self.c2c_text_o(obj_text_features)

        video_features = self.video_encoder(video)# b d t

        # independent learning
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))  # b,c
        o_feat_normed = F.normalize(o_feat, dim=1)
        v_feat_t = self.c2c_VE1(video_features)  # b,c,t
        v_feat = v_feat_t.mean(dim=-1)  # b,c
        v_feat_normed = F.normalize(v_feat, dim=1)

        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        verb_text_features_norm = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)
        obj_text_features_norm = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)

        verb_logits = v_feat_normed @ verb_text_features_norm.t()
        obj_logits = o_feat_normed @ obj_text_features_norm.t()


        verb_logits = verb_logits * 0.5 + 0.5
        obj_logits = obj_logits * 0.5 + 0.5


        # ===condition learning===
        b = video_features.shape[0]
        c = verb_text_features.shape[-1]
        n_v = verb_logits.shape[-1]
        n_o = obj_logits.shape[-1]

        # visual features
        o_feat_c = self.c2c_OE2(video_features.mean(dim=-1))
        v_feat_c = self.c2c_VE2(video_features)
        v_feat_c = v_feat_c.mean(dim=-1)

        p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, obj_text_features, n_o, b,
                                                     c, n_v)
        p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)  # b,nv,no
        p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)  # b,nv,no


        if self.training:
            return verb_logits, obj_logits, p_pair_v, p_pair_o, video_features, o_feat, v_feat, p_v_con_o, p_o_con_v
        else:
            verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
            com_logits = p_pair_o[:, verb_idx, obj_idx] + p_pair_v[:, verb_idx, obj_idx]
            return com_logits

    def condition_module(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb_normed = F.normalize(o_emb, dim=1)

        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1,
                                                                                                                  c * 2))  # b,1,c+1,n,c -->b,n,c*2--->b,no,c
        f_v_e_o_norm = F.normalize(f_v_e_o, dim=-1)
        f_v_e_o_norm = f_v_e_o_norm.view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1,
                                                                                                                  c * 2))  # b,1,c+1,n,c -->b,n,c*2--->b,nv,c
        f_o_e_v_norm = F.normalize(f_o_e_v, dim=-1)
        f_o_e_v_norm = f_o_e_v_norm.view(b, n_v, c)

        p_v_con_o = torch.einsum('bnc,mc->bnm', f_v_e_o_norm, v_emb_normed) * 0.5 + 0.5  # b,no,nv
        p_v_con_o = p_v_con_o.permute(0, 2, 1)  # b,nv,no
        p_o_con_v = torch.einsum('bnc,mc->bnm', f_o_e_v_norm, o_emb_normed) * 0.5 + 0.5  # b,nv,no
        return p_v_con_o, p_o_con_v


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def build_model(train_dataset,cfg):
    cfg = cfg
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg, train_dataset, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    # model.logit_scale
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                # if 'positional_embedding' in name:
                #     param.requires_grad_(True)
                #     print(f'{name}: {param.requires_grad}')
                if cfg.learn_input_method == 'coop':
                    if 'prompt_vectors' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'csp':
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'spm':
                    if 'prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                else:
                    raise NotImplementedError
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
                print(f'{name}: {param.requires_grad}')
        elif 'c2c' in name:
            param.requires_grad = True
            print(f'{name}: {param.requires_grad}')
    return model
