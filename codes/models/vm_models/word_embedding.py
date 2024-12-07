import torch
import numpy as np


def load_word_embeddings(emb_type, vocab):
    embeds = load_fasttext_embeddings(vocab)
    return embeds


def load_fasttext_embeddings(vocab):
    custom_map = {
        'Doesn\'t': 'does_not',
        'doesn\'t': 'does_not',
    }

    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)

    import fasttext.util
    ft=fasttext.load_model('/data/Disk_D/rongchang/ex_ideas/mappings/2023/com_act_rec/init_w//cc.en.300.bin')
    embeds = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        elif ' ' in k:
            emb_list = []
            # TODO to check
            if 'from left to right' in k or 'from right to left' in k:
                all_words = k.split(' ')
                new_verb = ' '.join(all_words[:-1])
                k = new_verb
            ks = k.split(' ')

            for it in ks:
                it = it.replace('[', '').replace(']', '').replace(',', '')

                if it in custom_map:
                    it = custom_map[it]
                if '_' in it:
                    it = it.split('_')
                    now_emb = np.stack([ft.get_word_vector(i) for i in it]).mean(axis=0)
                else:
                    try:
                        now_emb = ft.get_word_vector(it)
                    except:
                        now_emb = ft.get_word_vector(it.capitalize())
                emb_list.append(now_emb)
            emb = np.stack(emb_list).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    print('Fasttext Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds
