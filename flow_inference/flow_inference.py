import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse, os

import numpy as np

import models
import time
import timeit
from networks_flow.resample2d_package.modules.resample2d import Resample2d
import cv2

class Inference_flow(nn.Module):
    def __init__(self, args_flow, model_name="FlowNet2C",
                 restore_path="flow_inference/models/FlowNet2-C_checkpoint.pth.tar",
                 train_flow=False, resume=None):
        super(Inference_flow, self).__init__()
        args_flow.rgb_max = 255.0
        args_flow.fp16 = False
        args_flow.grads = {}

        if model_name == 'FlowNet2':
            self.model = models.FlowNet2(args_flow).cuda()
        elif model_name == 'FlowNet2C':
            self.model = models.FlowNet2C(args_flow).cuda()
        elif model_name == 'FlowNet2S':
            self.model = models.FlowNet2S(args_flow).cuda()
        elif model_name == 'FlowNet2SD':
            self.model = models.FlowNet2SD(args_flow).cuda()
        elif model_name == 'FlowNet2CS':
            self.model = models.FlowNet2CS(args_flow).cuda()
        elif model_name == 'FlowNet2CSS':
            self.model = models.FlowNet2CSS(args_flow).cuda()
        else:
            assert False, "No such model %s" % (model_name)
        print("loading %s pretrained model..." % (model_name))
        if train_flow:
            self.model.train()
        else:
            self.model.eval()
        if resume is not None:
            self.model.load_state_dict(torch.load(resume)['flow'])
        else:
            self.model.load_state_dict(torch.load(restore_path)['state_dict'])

    ## flow order: img2 -> img1 ##
    def infer(self, img1, img2, scale=1):
        #assert img1.shape[0] % 64 == 0 and img1.shape[1] % 64 == 0, "shape should be n*64, but got shape (%d, %d, %d)" \
        #                                                            % (img1.shape[0], img1.shape[1], img1.shape[2])
        # resize flow to appropriate shape

        ori_h = img1.shape[0]
        ori_w = img1.shape[1]
        resize_h = ori_h
        resize_w = ori_w
        if scale != 1:
            resize_h = resize_h // scale
            resize_w = resize_w // scale

        resize_h = resize_h // 64 * 64
        resize_w = resize_w // 64 * 64
        if resize_h == 0:
            resize_h = 64
        if resize_w == 0:
            resize_w = 64
        if ori_h != resize_h or ori_w != resize_w:
            ratio_h = float(ori_h) / resize_h
            ratio_w = float(ori_w) / resize_w
            img1 = cv2.resize(img1, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        assert img1.shape == img2.shape, "image1 and image2 should be same!"

        # concat and forward
        images = [img1, img2]
        images = np.array(images).transpose(3, 0, 1, 2)  # C 2 H W
        images = images[np.newaxis, :, :, :, :]
        images = Variable(torch.from_numpy(images.astype(np.float32)).cuda(), requires_grad=True)

        out_flo = self.model(images)[0].data.cpu().numpy().transpose(1, 2, 0)

        # resize back to original size
        if ori_h != resize_h or ori_w != resize_w:
            out_flo = cv2.resize(out_flo, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            out_flo[:, :, 0] = out_flo[:, :, 0] * ratio_w
            out_flo[:, :, 1] = out_flo[:, :, 1] * ratio_h

        return torch.FloatTensor(out_flo).cuda()


