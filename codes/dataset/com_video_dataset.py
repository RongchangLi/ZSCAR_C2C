from itertools import product
from os.path import join as ospj
import os
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

import dataset.gtransform as gtransform
from numpy.random import randint
from random import choice

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def dataset_transform(phase):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use

        Returns
            transform: A list of pytorch transforms
    '''
    # mean, std = get_norm_values(norm_family=norm_family)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    if phase == 'train':
        transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupMultiScaleCrop(224),
            # transforms.RandomHorizontalFlip(),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupCenterCrop(224),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupCenterCrop(224),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionVideoDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            num_negs=-1,
            frames_duration=8,
            tdn_input=False,
            aux_input=False,
            use_composed_pair_loss=False,
            ade_input=False,
            return_n_matrix=True,
            test_json='test_pairs.json',
            ex_test_json='test_pairs.json'
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        split_root='./data_split/sth_com/'
        self.splitroot =split_root
        self.test_json=test_json
        self.val_json='val_pairs.json'
        self.ex_test_json=ex_test_json

        # todo about video sampling #230612:seems done
        self.tdn_input = tdn_input
        self.in_duration = frames_duration
        self.seg_length = 1 if not tdn_input else 5
        self.index_bias = 1
        self.total_length = self.in_duration * self.seg_length

        # about_train_sampling
        self.num_negs = num_negs

        self.feat_dim = None
        self.transform = dataset_transform(self.phase)

        self.attrs, self.objs, self.pairs, \
        self.train_pairs, self.val_pairs, \
        self.test_pairs,self.ex_test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data
        self.prepare_data()

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        self.aux_input = aux_input
        self.use_composed_pair_loss = use_composed_pair_loss
        if aux_input:
            # Images that contain an object.
            self.image_with_obj = {}
            for i, instance in enumerate(self.train_data):
                obj = instance[2]
                if obj not in self.image_with_obj:
                    self.image_with_obj[obj] = []
                self.image_with_obj[obj].append(i)

            # Images that contain an attribute.
            self.image_with_attr = {}
            for i, instance in enumerate(self.train_data):
                attr = instance[1]
                if attr not in self.image_with_attr:
                    self.image_with_attr[attr] = []
                self.image_with_attr[attr].append(i)
        if use_composed_pair_loss:
            unseen_pairs = set()
            for pair in self.val_pairs + self.test_pairs:
                if pair not in self.train_pair_to_idx:
                    unseen_pairs.add(pair)
            self.unseen_pairs = list(unseen_pairs)
            self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}

        self.return_n_matrix=return_n_matrix

        self.ade_input = ade_input
        if ade_input:
            self.obj_affordance = {}
            self.train_obj_affordance = {}
            for _obj in self.objs:
                candidates = [attr for (_, attr, obj) in self.train_data + self.test_data if obj == _obj]
                self.obj_affordance[_obj] = list(set(candidates))

                candidates = [attr for (_, attr, obj) in self.train_data if obj == _obj]
                self.train_obj_affordance[_obj] = list(set(candidates))

            self.train_attr_set = {}
            self.train_attr_set_obj_num = {}
            for _attr in self.attrs:
                candidates = [i for i, (_, attr, obj) in enumerate(self.train_data) if attr == _attr]
                self.train_attr_set[_attr] = list(set(candidates))
                self.train_attr_set_obj_num[_attr] = len(
                    set([self.train_data[idx][2] for idx in self.train_attr_set[_attr]]))

            self.train_obj_set = {}
            self.train_obj_set_attr_num = {}
            for _obj in self.objs:
                candidates = [i for i, (_, attr, obj) in enumerate(self.train_data) if obj == _obj]
                self.train_obj_set[_obj] = list(set(candidates))
                self.train_obj_set_attr_num[_obj] = len(
                    set([self.train_data[idx][1] for idx in self.train_obj_set[_obj]]))

    def prepare_data(self):
        frame_cnts = {}
        for item in self.data:
            item_id = item[0]
            try:
                frames_path = ospj(self.root, item_id)
                frames = os.listdir(frames_path)
                n_frame = int(len(frames))
            except Exception as e:
                print(str(e))
            frame_cnts[item_id] = n_frame

        self.frame_cnts = frame_cnts

    def get_split_info(self):
        with open(ospj(self.splitroot, 'train_pairs.json'), 'r') as f:
            items = json.load(f)
            train_data = [[item['id'], item['verb'], item['object']] for item in items]

        with open(ospj(self.splitroot, self.val_json), 'r') as f:
            items = json.load(f)
            val_data = [[item['id'], item['verb'], item['object']] for item in items]

        with open(ospj(self.splitroot, self.test_json), 'r') as f:
            items = json.load(f)
            test_data = [[item['id'], item['verb'], item['object']] for item in items]

        return train_data, val_data, test_data

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                items = json.load(f)
                pairs = [[item['verb'], item['object']] for item in items]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return list(set(attrs)), list(set(objs)), list(set(pairs))

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.splitroot, 'train_pairs.json')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.splitroot, self.val_json)
        )

        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.splitroot, self.test_json)
        )

        ex_ts_attrs, ex_ts_objs, ex_ts_pairs = parse_pairs(
            ospj(self.splitroot, self.ex_test_json)
        )

        # now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs+ex_ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs+ex_ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs+ex_ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs,ex_ts_pairs

    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        return Image.open(ospj(self.root, vid_name, '%06d.jpg' % (frame_idx))).convert('RGB')

    def _sample_indices(self, id):
        if not self.tdn_input:
            if self.frame_cnts[id] <= self.total_length:
                offsets = np.concatenate((
                    np.arange(self.frame_cnts[id]),
                    randint(self.frame_cnts[id],
                            size=self.total_length - self.frame_cnts[id])))
                offsets.sort()
                return offsets
            offsets = list()
            ticks = [i * self.frame_cnts[id] // self.in_duration
                     for i in range(self.in_duration + 1)]

            for i in range(self.in_duration):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return offsets
        else:
            if ((self.frame_cnts[id] - self.seg_length + 1) < self.in_duration):
                average_duration = (self.frame_cnts[id] - 5 + 1) // (self.in_duration)
            else:
                average_duration = (self.frame_cnts[id] - self.seg_length + 1) // (self.in_duration)
            offsets = []
            if average_duration > 0:
                offsets += list(
                    np.multiply(list(range(self.in_duration)), average_duration) + randint(average_duration,
                                                                                           size=self.in_duration))
            elif self.frame_cnts[id] > self.in_duration:
                if ((self.frame_cnts[id] - self.seg_length + 1) >= self.in_duration):
                    offsets += list(np.sort(randint(self.frame_cnts[id] - self.seg_length + 1, size=self.in_duration)))
                else:
                    offsets += list(np.sort(randint(self.frame_cnts[id] - 5 + 1, size=self.in_duration)))
            else:
                offsets += list(np.zeros((self.in_duration,)))
            final_offset = []
            for i in offsets:
                for bias in range(5):
                    final_offset.append(i + bias)
            return final_offset

    def _get_val_indices(self, id):
        if not self.tdn_input:
            if self.in_duration == 1:
                return np.array([self.frame_cnts[id] // 2], dtype=np.int) + self.index_bias

            if self.frame_cnts[id] <= self.in_duration:
                return np.array([i * self.frame_cnts[id] // self.in_duration
                                 for i in range(self.in_duration)], dtype=np.int) + self.index_bias
            offset = (self.frame_cnts[id] / self.in_duration - self.seg_length) / 2.0
            return [i * self.frame_cnts[id] / self.in_duration + offset + j
                    for i in range(self.in_duration)
                    for j in range(self.seg_length)]
        else:
            if self.frame_cnts[id] > self.in_duration + self.seg_length - 1:
                tick = (self.frame_cnts[id] - self.seg_length + 1) / float(self.in_duration)
                offsets = [int(tick / 2.0 + tick * x) for x in range(self.in_duration)]
            else:
                offsets = [0 for i in range(self.in_duration)]

            final_offset = []
            for i in offsets:
                for bias in range(5):
                    final_offset.append(i + bias)
            return final_offset

    def _load_video(self, id):
        if self.phase == 'train':
            frame_list = self._sample_indices(id)
        else:
            frame_list = self._get_val_indices(id)
        frame_list = [int(x) + self.index_bias for x in frame_list]
        frames = []
        for fidx in frame_list:
            frames.append(self.load_frame(id, fidx))
        return frames

    def sample_negative(self, attr, obj):
        new_attr, new_obj = self.train_pairs[np.random.choice(
            len(self.train_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.train_pairs[np.random.choice(
                len(self.train_pairs))]


        return [self.attr2idx[new_attr], self.obj2idx[new_obj]]

    def sample_same_attribute(self, attr, obj, with_different_obj=True):
        if with_different_obj:
            i2 = np.random.choice(self.image_with_attr[attr])
            i = 1
            img1, attr1, obj1 = self.data[i2]
            while obj1 == obj and i <= 10:
                i += 1
                i2 = np.random.choice(self.image_with_attr[attr])
                img1, attr1, obj1 = self.data[i2]
            # if obj1 == obj:
            #     i2=-1

        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return i2

    def sample_same_object(self, attr, obj, with_different_attr=True):
        i2 = np.random.choice(self.image_with_obj[obj])
        i = 1
        if with_different_attr:
            img1, attr1, obj1 = self.data[i2]

            while attr1 == attr and i <= 10:
                i += 1
                i2 = np.random.choice(self.image_with_obj[obj])
                img1, attr1, obj1 = self.data[i2]
            # if  attr1 == attr:
            #     i2=-1
        return i2

    def sample_train_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])

        return self.attr2idx[new_attr]

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])

        return self.attr2idx[new_attr]

    def sample_neg_images(self, attr, obj):

        data_id_same_attr = np.random.choice(self.train_attr_set[attr])
        if self.train_attr_set_obj_num[attr] > 1:
            while self.train_data[data_id_same_attr][2] == obj:  # check obj
                data_id_same_attr = np.random.choice(self.train_attr_set[attr])

        data_id_same_obj = np.random.choice(self.train_obj_set[obj])
        if self.train_obj_set_attr_num[obj] > 1:
            while self.train_data[data_id_same_obj][1] == attr:  # check attr
                data_id_same_obj = np.random.choice(self.train_obj_set[obj])

        # print("diff obj comp: attr={} and obj={}".format(self.train_data[data_id_same_attr][1]==attr, self.train_data[data_id_same_attr][2]==obj))
        # print("diff attr comp: attr={} and obj={}".format(self.train_data[data_id_same_obj][1]==attr, self.train_data[data_id_same_obj][2]==obj))

        return self.train_data[data_id_same_attr][0], self.train_data[data_id_same_obj][0]

    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        data1 = []
        for i in range(len(self.train_data)):
            if phase == 'attr':
                if (self.train_data[i][1] == label_A) & (self.train_data[i][2] != label_B):
                    data1.append(self.train_data[i])
            else:
                if (self.train_data[i][2] == label_A) & (self.train_data[i][1] != label_B):
                    data1.append(self.train_data[i])

        if len(data1) == 0:
            for i in range(len(self.train_data)):
                if phase == 'attr':
                    if (self.train_data[i][1] == label_A):
                        data1.append(self.train_data[i])
                else:
                    if (self.train_data[i][2] == label_A):
                        data1.append(self.train_data[i])
        data2 = choice(data1)
        return data2

    def __getitem__(self, index):
        id, attr, obj = self.data[index]
        vid = self._load_video(id)
        vid = self.transform(vid)
        if self.phase == 'train':
            data = [
                vid, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]

            if self.return_n_matrix:
                n_c_v = torch.zeros(size=(1, len(self.attrs)))
                n_c_v[:, self.attr2idx[attr]] = 1.0
                n_c_o = torch.zeros(size=(1, len(self.objs)))
                n_c_o[:, self.obj2idx[obj]] = 1.0

                data.append(n_c_v)
                data.append(n_c_o)

            if self.ade_input:
                img_diff_obj_path, img_diff_attr_path = self.sample_neg_images(attr, obj)

                img_diff_obj = self._load_video(img_diff_obj_path)
                img_diff_obj = self.transform(img_diff_obj)
                img_diff_attr = self._load_video(img_diff_attr_path)
                img_diff_attr = self.transform(img_diff_attr)

                data += [img_diff_obj, img_diff_attr]

            if self.aux_input:
                i1 = self.sample_same_attribute(attr, obj, with_different_obj=True)
                id1, attr1, obj1 = self.data[i1]
                vid1 = self._load_video(id1)
                vid1 = self.transform(vid1)
                data.extend([vid1, self.obj2idx[obj1]])

                mask_task = 1
                if i1 == -1:
                    mask_task = 0

                i2 = self.sample_same_object(attr, obj, with_different_attr=True)
                id2, attr2, obj2 = self.data[i2]
                vid2 = self._load_video(id2)
                vid2 = self.transform(vid2)
                data.extend([vid2, self.attr2idx[attr2]])

                if i2 == -1:
                    mask_task = 0

                if self.use_composed_pair_loss:
                    if (attr2, obj1) in self.unseen_pair2idx:
                        composed_unseen_pair = self.unseen_pair2idx[(attr2, obj1)]
                        composed_seen_pair = 2000
                    elif (attr2, obj1) in self.train_pair_to_idx:
                        composed_seen_pair = self.train_pair_to_idx[(attr2, obj1)]
                        composed_unseen_pair = 2000
                    else:
                        composed_unseen_pair = 2000
                        composed_seen_pair = 2000

                    data.extend([composed_seen_pair, composed_unseen_pair, mask_task])


        else:
            data = [
                vid, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
