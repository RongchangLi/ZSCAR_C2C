import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class compositionPromptLearner(nn.Module):
    # For verb or object
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()

        dtype = clip_model.dtype

        input_template = cfg.input_template
        self.learn_input_method = cfg.learn_input_method
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = nn.Parameter(clip_model.positional_embedding[:cfg.ctx_length, :].unsqueeze(0),
                                                 requires_grad=True)
        # prepare the prompt part
        token_ids = clip.tokenize(input_template, context_length=cfg.ctx_length)
        self.register_buffer('token_ids', token_ids)
        input_template_list = input_template.split('x')
        n_head = len(input_template_list[0].split(' ')) - 1
        n_mid = len(input_template_list[1].split(' ')) - 2
        prompt_head = clip.tokenize(input_template_list[0])
        prompt_mid = clip.tokenize(input_template_list[1])
        with torch.no_grad():
            embedding_head = clip_model.token_embedding(prompt_head).type(dtype)
            embedding_mid = clip_model.token_embedding(prompt_mid).type(dtype)
        prompt_vectors_head = embedding_head[0, 1: 1 + n_head, :]  # TODO
        prompt_vectors_mid = embedding_mid[0, 1: 1 + n_mid, :]  # TODO

        # prepare the object and verb part
        objectnames = [o.replace("[", "").replace("]", "").replace(",", "").lower() for o in train_dataset.objs]
        verbnames = [v.replace("[", "").replace("]", "").replace(",", "").lower() for v in train_dataset.attrs]
        for i, verbname in enumerate(verbnames):
            if 'from left to right' in verbname or 'from right to left' in verbname:
                all_words = verbname.split(' ')
                new_verb = ' '.join(all_words[:-1])
                verbnames[i] = new_verb
        obj_tokenized = torch.cat([clip.tokenize(tok) for tok in objectnames])
        verb_tokenized = torch.cat([clip.tokenize(tok) for tok in verbnames])

        # TODO, Check the indexes
        with torch.no_grad():
            obj_token_embedding = clip_model.token_embedding(obj_tokenized).type(dtype)
            verb_token_embedding = clip_model.token_embedding(verb_tokenized).type(dtype)
            obj_embedding = torch.zeros(
                (len(objectnames), 1, clip_model.token_embedding.weight.size(-1)))
            verb_embedding = torch.zeros(
                (len(verbnames), 1, clip_model.token_embedding.weight.size(-1)))
            for idx, rep in enumerate(obj_token_embedding):
                eos_idx = obj_tokenized[idx].argmax()
                # verb_obj_n.append(eos_ind+1-2)
                obj_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
            # verb_n = []
            for idx, rep in enumerate(verb_token_embedding):
                eos_ind  = verb_tokenized[idx].argmax()
                # verb_n.append(eos_ind + 1 - 2)
                # verb_n.append(1)
                verb_embedding[idx, :] = torch.mean(rep[1:eos_ind, :], axis=0)  # with soc

        if cfg.learn_input_method == 'coop':  # suitable for composition learning framework.
            self.prompt_vectors_head = nn.Parameter(prompt_vectors_head, requires_grad=True)  # to be optimized
            self.prompt_vectors_mid = nn.Parameter(prompt_vectors_mid, requires_grad=True)  # to be optimized
            self.register_buffer('obj_embedding', obj_embedding)
            self.register_buffer('verb_embedding', verb_embedding)
        elif cfg.learn_input_method == 'csp':
            self.register_buffer('prompt_vectors_head', prompt_vectors_head)
            self.register_buffer('prompt_vectors_mid', prompt_vectors_mid)
            self.obj_embedding = nn.Parameter(obj_embedding, requires_grad=True)
            self.verb_embedding = nn.Parameter(verb_embedding, requires_grad=True)
        elif cfg.learn_input_method == 'spm':
            self.prompt_vectors_head = nn.Parameter(prompt_vectors_head, requires_grad=True)  # to be optimized
            self.prompt_vectors_mid = nn.Parameter(prompt_vectors_mid, requires_grad=True)  # to be optimized
            self.obj_embedding = nn.Parameter(obj_embedding, requires_grad=True)
            self.verb_embedding = nn.Parameter(verb_embedding, requires_grad=True)
        elif cfg.learn_input_method == 'zero':
            self.register_buffer('prompt_vectors_head', prompt_vectors_head)
            self.register_buffer('prompt_vectors_mid', prompt_vectors_mid)
            self.register_buffer('obj_embedding', obj_embedding)
            self.register_buffer('verb_embedding', verb_embedding)
        else:
            raise NotImplementedError
        #
        # verb_n = torch.tensor(verb_n) + token_ids.argmax() - 1
        # self.register_buffer('verb_n', verb_n)
        self.n_prompt_head = n_head
        self.n_prompt_mid = n_mid

    def forward(self, pair_idx):
        verb_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        verb_idx, obj_idx = verb_idx.view(-1, ), obj_idx.view(-1, )
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.token_embedding(class_token_ids)
        token_tensor[:, 1:self.n_prompt_head + 1] = self.prompt_vectors_head
        token_tensor[:, self.n_prompt_head + 1:
                        self.n_prompt_head + 2] = self.verb_embedding[verb_idx]
        token_tensor[:, self.n_prompt_head + 2:
                        self.n_prompt_head + 2 + self.n_prompt_mid] = self.prompt_vectors_mid
        token_tensor[:, self.n_prompt_head + 2 + self.n_prompt_mid:
                        self.n_prompt_head + 3 + self.n_prompt_mid] = self.obj_embedding[obj_idx]
        return token_tensor + self.positional_embedding


class componentPromptLearner(nn.Module):
    # For verb or object
    def __init__(self, cfg, train_dataset, clip_model, comp):
        super().__init__()


        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [train_dataset.attr2idx[attr] for attr in attrs]
            objs = [train_dataset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.Tensor(attrs)
            objs = torch.Tensor(objs)
            pairs = torch.Tensor(pairs)
            return attrs, objs, pairs

        # Validation
        val_attrs, val_objs, val_pairs = get_all_ids(train_dataset.pairs)
        self.register_buffer('val_attrs', val_attrs)
        self.register_buffer('val_objs', val_objs)
        self.register_buffer('val_pairs', val_pairs)
        # for indivual projections
        uniq_attrs, uniq_objs = torch.arange(len(train_dataset.attrs)), \
                                torch.arange(len(train_dataset.objs))
        self.register_buffer('uniq_attrs', uniq_attrs)
        self.register_buffer('uniq_objs', uniq_objs)
        # Precompute training compositions
        train_attrs, train_objs, train_pairs = get_all_ids(train_dataset.train_pairs)

        self.register_buffer('train_attrs', train_attrs)
        self.register_buffer('train_objs', train_objs)
        self.register_buffer('train_pairs', train_pairs)


        dtype = clip_model.dtype
        self.comp = comp

        if comp == 'verb':
            input_template = cfg.input_template_verb
        elif comp == 'object':
            input_template = cfg.input_template_obj
        else:
            raise NotImplementedError

        # input_template = cfg.input_template
        self.learn_input_method = cfg.learn_input_method
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = nn.Parameter(clip_model.positional_embedding[:cfg.ctx_length, :].unsqueeze(0),
                                                 requires_grad=True)
        # prepare the prompt part
        token_ids = clip.tokenize(input_template, context_length=cfg.ctx_length)
        self.register_buffer('token_ids', token_ids)
        input_template_list = input_template.split('x')
        n_head = len(input_template_list[0].split(' ')) - 1
        prompt_head = clip.tokenize(input_template_list[0])
        with torch.no_grad():
            embedding_head = clip_model.token_embedding(prompt_head).type(dtype)
        prompt_vectors_head = embedding_head[0, 1: 1 + n_head, :]  # TODO

        # prepare the object and verb part
        if comp == 'verb':
            compnames = [v.replace("[", "").replace("]", "").replace(",", "").lower() for v in train_dataset.attrs]
        elif comp == 'object':
            compnames = [o.replace("[", "").replace("]", "").replace(",", "").lower() for o in train_dataset.objs]

        if comp == 'verb':
            for i, compname in enumerate(compnames):
                if 'from left to right' in compname or 'from right to left' in compname:
                    all_words = compname.split(' ')
                    new_verb = ' '.join(all_words[:-1])
                    compnames[i] = new_verb

        comp_tokenized = torch.cat([clip.tokenize(tok) for tok in compnames])

        # TODO, Check the indexes
        with torch.no_grad():
            comp_token_embedding = clip_model.token_embedding(comp_tokenized).type(dtype)
            comp_embedding = torch.zeros(
                (len(compnames), 1, clip_model.token_embedding.weight.size(-1)))
            for idx, rep in enumerate(comp_token_embedding):
                eos_idx = comp_tokenized[idx].argmax()
                comp_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
            # for idx, rep in enumerate(comp_token_embedding):
            #     eos_ind = comp_tokenized[idx].argmax()
            #     comp_embedding[idx, :] = torch.mean(rep[1:eos_ind, :], axis=0)  # with soc

        if cfg.learn_input_method == 'coop':  # suitable for composition learning framework.
            self.prompt_vectors_head = nn.Parameter(prompt_vectors_head, requires_grad=True)  # to be optimized
            self.register_buffer('comp_embedding', comp_embedding)
        elif cfg.learn_input_method == 'csp':
            self.register_buffer('prompt_vectors_head', prompt_vectors_head)
            self.comp_embedding = nn.Parameter(comp_embedding, requires_grad=True)  # to be optimized
        elif cfg.learn_input_method == 'spm':
            self.prompt_vectors_head = nn.Parameter(prompt_vectors_head, requires_grad=True)  # to be optimized
            self.comp_embedding = nn.Parameter(comp_embedding, requires_grad=True)  # to be optimized
        elif cfg.learn_input_method == 'zero':
            self.register_buffer('prompt_vectors_head', prompt_vectors_head)
            self.register_buffer('comp_embedding', comp_embedding)
        else:
            raise NotImplementedError

        self.n_prompt_head = n_head

    def forward(self):

        verb_idx, obj_idx = self.uniq_attrs, self.uniq_objs
        # verb_idx, obj_idx = pair_idx[:,0], pair_idx[:,1]

        if self.comp == 'verb':
            this_comp= self.uniq_attrs
            this_idx = verb_idx
        elif self.comp == 'object':
            this_comp=self.uniq_objs
            this_idx = obj_idx
        class_token_ids = self.token_ids.repeat(len(this_comp), 1)
        token_tensor = self.token_embedding(class_token_ids)
        token_tensor[:, 1:self.n_prompt_head + 1] = self.prompt_vectors_head
        token_tensor[:, self.n_prompt_head + 1:
                        self.n_prompt_head + 2] = self.comp_embedding[this_idx]
        return token_tensor + self.positional_embedding


def get_text_learner(cfg, train_dataset, clip_model, comp='verb'):
    if cfg.text_encoding_manner == 'composition':
        text_learner = compositionPromptLearner(cfg, train_dataset, clip_model)
    elif cfg.text_encoding_manner == 'component':
        text_learner = componentPromptLearner(cfg, train_dataset, clip_model, comp)
    else:
        raise NotImplementedError
    return text_learner
