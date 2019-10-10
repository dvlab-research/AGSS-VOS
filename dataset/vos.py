import cv2
import json
import logging
import math
import numpy as np
import numpy.random as npr
import os
import os.path as osp
import random
from random import shuffle
import sys
from torch.utils.data import Dataset

sys.path.append('./tools')
import preprocess

class Trainset(Dataset):
    def __init__(self, root_data, json_meta_list,
               mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
               sample_size=5, test_mode=False, random_flip=True,
               fix_size=False, spec_vid=None, spec_range=None, spec_obj_ind=None,
               abandon_len1=False, min_obj_num=1, step=5, half_size=True,
              random_ref=True, random_skip=False):
        self.mean = mean
        self.std = std
        self.files = []
        self.files_obj_comb = []
        self.sample_size = sample_size
        self.test_mode = test_mode
        self.random_flip = random_flip
        self.fix_size = fix_size
        self.spec_range = spec_range
        self.half_size = half_size
        self.random_ref = random_ref
        self.random_skip = random_skip

        if spec_vid is None:
            assert spec_range is None
            assert spec_obj_ind is None

        meta_list = json.loads(open(json_meta_list).read())['videos']

        for name, vid in meta_list.items():
            if spec_vid is not None and name != spec_vid:
                continue
            min_idx = 1000
            max_idx = -1
            for k, obj_ind in vid['objects'].items():
                min_idx = min(min_idx, int(obj_ind['frames'][0]))
                max_idx = max(max_idx, int(obj_ind['frames'][-1]))

            vid_seq = []
            obj_num = len(vid['objects'])
            idx_map = dict()
            for idx in range(min_idx, max_idx+1, step):
                img_file = osp.join(root_data, 'JPEGImages/%s/%05d.jpg' % (name, idx))
                idx_map['%05d' % idx] = len(vid_seq)
                vid_seq.append({
                    'img': img_file,
                    'lab': [None for _ in range(obj_num)],
                    'attr': [0 for _ in range(obj_num)]
                })
            obj_cnt = 0
            for k, obj_ind in vid['objects'].items():
                if spec_obj_ind is not None and int(k) not in spec_obj_ind:
                    continue
                for frame in obj_ind['frames']:
                    id = idx_map[frame]
                    lab_file = osp.join(root_data, 'Split_Annotations/%s/%s/%s.png' % (name, k, frame))
                    vid_seq[id]['lab'][obj_cnt] = lab_file
                    vid_seq[id]['attr'][obj_cnt] = 1 #if cv2.imread(lab_file, cv2.IMREAD_GRAYSCALE).sum()>0 else 0
                obj_cnt += 1
            self.files.append(vid_seq)
            obj_attr = dict()
            for vid_f in vid_seq:
                if sum(vid_f['attr']) >= min_obj_num:
                    tmp_vid = tuple(vid_f['attr'])
                if tmp_vid not in obj_attr:
                    obj_attr[tmp_vid] = 0
                    obj_attr[tmp_vid] += 1
            for k,v in obj_attr.items():
                if abandon_len1 and v == 1:
                    continue
                self.files_obj_comb.append((len(self.files)-1,k))
        self.files_obj_comb = self.files_obj_comb
        random.seed(20170624)
        npr.seed(20170624)

    def __len__(self):
        return len(self.files_obj_comb)

    def __getitem__(self, item):
        vid_obj_comb = self.files_obj_comb[item] # (vid_name, obj_ind)
        video_list = self.files[vid_obj_comb[0]].copy()
        if self.spec_range is not None:
            video_list = video_list[self.spec_range[0]:self.spec_range[1]]
        ### flip the video is also an augmentation ###
        if not self.test_mode and random.uniform(0,1) > 0.5:
            video_list = video_list[::-1]
        assert len(video_list) > 0
        if len(video_list) >= 2:
            ### make sure must all the obj involved ###
            vid_attr = vid_obj_comb[1]
            cand_start_list = [i for i in range(len(video_list)) if tuple(video_list[i]['attr']) == vid_attr]
            video_select = []
            ### random skip some frames ###
            if self.random_skip:
                skip = random.randint(1,5)
            else:
                skip = 1
            if len(cand_start_list) > self.sample_size*skip:
                start = random.randint(0, len(cand_start_list)-2)
                for k in cand_start_list[start:start+self.sample_size*skip][::skip]:
                    video_select.append(video_list[k])
            else:
                for k in cand_start_list:
                    video_select.append(video_list[k])
            ### random select ref frame ###
            if self.random_ref:
                k = random.randint(0, len(cand_start_list)-1)
                k = cand_start_list[k]
                video_select = video_list[k:k+1] + video_select
        else:
            video_select = video_list


        img_set = []
        lab_set = []
        ori_img_set = []

        flip_flag = random.uniform(0,1) > 0.5 and self.random_flip
        attr = video_select[0]['attr']

        for idx,datafiles in enumerate(video_select):
            image = cv2.imread(datafiles['img'], cv2.IMREAD_COLOR)
            labels = []
            for i,lab_name in enumerate(datafiles['lab']):
                if attr[i] == 1:
                    if lab_name is not None:
                        lab = cv2.imread(lab_name, cv2.IMREAD_GRAYSCALE)
                        labels.append(lab)
                    else:
                        labels.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))

            if self.fix_size:
                dsize = (640,320)
            else:
                dsize = preprocess.get_dsize(image, half_size=self.half_size, scale=64)

            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            for i in range(len(labels)):
                if labels[i] is not None:
                    labels[i] = cv2.resize(labels[i], dsize, interpolation=cv2.INTER_NEAREST)

            ori_img = image.copy()

            image = preprocess.norm(image, self.mean, self.std)

            ### flip each frame ###
            if flip_flag:
                image = image[:,::-1]
                for i in range(len(labels)):
                    if labels[i] is not None:
                        labels[i] = labels[i][:,::-1]
                ori_img = ori_img[:,::-1]

            image = image.transpose((2, 0, 1))
            image_cat = []
            label_cat = []
            for i in range(len(labels)):
                image_cat.append(image[np.newaxis,:])
                label_cat.append(labels[i][np.newaxis,:])

            image_cat = np.concatenate(image_cat, axis=0)
            label_cat = np.concatenate(label_cat, axis=0)
            img_set.append(image_cat[:,np.newaxis,:])
            lab_set.append(label_cat[:,np.newaxis,:])
            ori_img_set.append(ori_img[np.newaxis,:])

        img_set = np.concatenate((img_set), axis=1)
        lab_set = np.concatenate((lab_set), axis=1)
        ori_img_set = np.concatenate((ori_img_set), axis=0)

        return img_set, lab_set, ori_img_set


class Valset(Dataset):
    def __init__(self, root_data, root_all_data, json_meta_list,
                 mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                 sample_size=5, all_frames=True, test_mode=False, fix_size=True,
                 spec_vid=None, spec_obj_ind=None, half_size=True):
        self.mean = mean
        self.std = std
        self.files = []
        if all_frames:
            self.sample_size = (sample_size-1)*5+1
        else:
            self.sample_size = sample_size
        self.test_mode = test_mode
        self.fix_size = fix_size
        self.half_size = half_size

        meta_list = json.loads(open(json_meta_list).read())['videos']
        if spec_obj_ind is not None:
            assert spec_vid is not None

        for name, vid in meta_list.items():
            if spec_vid is not None and name not in spec_vid:
                continue
            min_idx = 1000
            max_idx = -1
            for k, obj_ind in vid['objects'].items():
                min_idx = min(min_idx, int(obj_ind['frames'][0]))
                max_idx = max(max_idx, int(obj_ind['frames'][-1]))

            vid_info = dict()
            vid_seq = []
            idx_map = dict()
            step = 1 if all_frames or test_mode else 5
            for idx in range(min_idx, max_idx+1, step):
                img_file = osp.join(root_all_data, 'JPEGImages/%s/%05d.jpg' % (name, idx))
                idx_map['%05d' % idx] = len(vid_seq)
                vid_seq.append(img_file)

            vid_info['imgs'] = vid_seq
            vid_info['name'] = name
            vid_info['min_idx'] = min_idx
            vid_lab = []
            obj_num = 0
            for k, obj_ind in vid['objects'].items():
                if spec_obj_ind is not None:
                    if int(k) not in spec_obj_ind:
                        continue
                obj_num = max(obj_num, int(k))
                frame = obj_ind['frames'][0]
                lab_file = osp.join(root_data, 'Split_Annotations/%s/%s/%s.png' % (name, k, frame))
                obj_lab = dict()
                obj_lab['obj_ind'] = k
                obj_lab['lab_file'] = lab_file
                obj_lab['start_idx'] = int(frame)-min_idx
                vid_lab.append(obj_lab)
            vid_info['obj_num'] = obj_num
            vid_info['labs'] = vid_lab

            self.files.append(vid_info)
        random.seed(20170624)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        vid_info = self.files[item].copy()

        img_set = []
        lab_set = []
        ori_img_set = []
        ori_shape = None
        dsize = None

        ### gen img_set & ori_img_set ###
        for img_name in vid_info['imgs']:
            image = cv2.imread(img_name, cv2.IMREAD_COLOR)
            if ori_shape is None:
                ori_shape = image.shape[:2]

            if self.fix_size:
                # dsize = (640,320) ## for ytv
                dsize = (832,448) ## for davis
            else:
                dsize = preprocess.get_dsize(image, half_size=self.half_size, scale=64)

            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            ori_img = image.copy()

            image = preprocess.norm(image, self.mean, self.std)
            image = image.transpose((2, 0, 1))

            img_set.append(image[np.newaxis,:])
            ori_img_set.append(ori_img[np.newaxis,:])

        img_set = np.concatenate((img_set), axis=0)
        ori_img_set = np.concatenate((ori_img_set), axis=0)

        obj_ind = []
        obj_start_idx = []
        ### gen lab_set ###
        for obj_lab in vid_info['labs']:
            lab = cv2.imread(obj_lab['lab_file'], cv2.IMREAD_GRAYSCALE)
            lab = cv2.resize(lab, dsize, interpolation=cv2.INTER_NEAREST)
            lab_set.append(lab[np.newaxis,:])
            obj_ind.append(obj_lab['obj_ind'])
            obj_start_idx.append(obj_lab['start_idx'])

        lab_set = np.concatenate((lab_set), axis=0)


        return img_set, lab_set, ori_img_set, vid_info['name'], vid_info['min_idx'], vid_info['obj_num'], \
               obj_ind, obj_start_idx, ori_shape

