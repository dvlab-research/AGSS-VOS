import sys

import os
import os.path as osp
import time
import logging
import numpy as np
import argparse
from copy import deepcopy
from tqdm import tqdm

import torch
assert torch.__version__ == '0.4.0'
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset.vos import Valset
from networks.agssvos import AGSSVOS
import sys
sys.path.append('./flow_inference')
from flow_inference.flow_inference import Inference_flow
from tools import preprocess, visualize, utils
import timeit
import cv2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--root-data', type=str)
    parser.add_argument('--root-all-data', type=str)
    parser.add_argument('--list-path', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--fix-size', action='store_true')
    parser.add_argument('--stop-iou', type=float, default=0.8)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--sample-dir', type=str)
    parser.add_argument('--snapshot-dir', type=str)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--resize_h', type=int, default=360)
    parser.add_argument('--resize_w', type=int, default=640)
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--div_flow', type=int, default=20)
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--scale_min', type=float, default=0.5, help='minimum random scale')
    parser.add_argument('--scale_max', type=float, default=2.0, help='maximum random scale')
    parser.add_argument('--rotate_min', type=float, default=-10, help='minimum random rotate')
    parser.add_argument('--rotate_max', type=float, default=10, help='maximum random rotate')
    parser.add_argument('--flow_checkpoint_path', type=str, default='models/FlowNet2-C_checkpoint.pth.tar',
                        help='pretrained model for flownetC')
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--test-mode', type=str)
    parser.add_argument('--spec-vid', type=str, default=None)
    parser.add_argument('--spec-obj-ind', type=str, default=None)
    parser.add_argument('--show-img', action='store_true', help='whether save the visualized image result')
    return parser

# get logger
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def show(images, labels, preds):
    os.system('rm %s/*' % args.sample_dir)
    for i_bs in range(images.shape[0]):
        for j_bs in range(labels.shape[1]):
            path = args.sample_dir + '/' + str(i_bs)+'_'+str(j_bs) + '#'
            image = visualize.denorm_image(images[i_bs, j_bs, :])
            label = visualize.vis_label(labels[i_bs, j_bs], 1, 128)
            cv2.imwrite(path + 'img.jpg', image)
            cv2.imwrite(path + 'lab.jpg', label)

        for j_bs in range(preds.shape[1]):
            path = args.sample_dir + '/' + str(i_bs)+'_'+str(j_bs) + '#'
            pred = preds[i_bs,j_bs]*255
            cv2.imwrite(path + 'pred.jpg', pred)

def main():
    global args, logger, writer
    args = get_parser().parse_args()
    logger = get_logger()
    print(args)

    test_mode = int(args.test_mode) > 0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # setting up model
    model = AGSSVOS().cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    flow_infer = Inference_flow(args, train_flow=False, resume=args.restore)

    for m in model.module.Encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


    if args.restore != None:
        assert os.path.isfile(args.restore), "no restore file found at %s" % (args.restore)
        logger.info("loading from %s" % (args.restore))

        checkpoint = torch.load(args.restore)['seg']
        model.load_state_dict(checkpoint)

        del checkpoint
        torch.cuda.empty_cache()

    spec_vid = args.spec_vid if not test_mode else None
    spec_obj_ind = [1] if not test_mode else None
    # print('spec_vid, spec_obj_ind', spec_vid, spec_obj_ind)

    testloader = data.DataLoader(
        Valset(root_data=args.root_data, root_all_data=args.root_all_data, json_meta_list=args.list_path,
                   sample_size=args.sample_size, fix_size=args.fix_size, half_size=False,
                   test_mode=True, spec_vid=spec_vid, spec_obj_ind=spec_obj_ind),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # training
    tot_iter = len(testloader)
    logger.info("Total iteration per epoch is %d" % (tot_iter))
    vid_time = []
    vid_time_cv2 = []
    vid_frm_num = []

    for i_iter, batch in enumerate(testloader):
        img, lab, ori_img, vid_name, min_idx, obj_num, obj_ind, obj_start_idx, ori_shape = batch

        vid_name = vid_name[0]

        logger.info('vid_name = %s' % vid_name)
        sample_dir = args.sample_dir+'/'+vid_name+'/'
        save_dir = args.save_dir+'/'+vid_name+'/'

        if not osp.exists(sample_dir):
             os.makedirs(sample_dir)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        start_time = timeit.default_timer()
        start_time_cv2 = cv2.getTickCount()

        img = img.cuda().float()
        lab = lab[0].cuda().float() # KHW
        img = img.expand(lab.shape[0],-1,-1,-1,-1) # KT3HW
        ori_img = ori_img[0].numpy() # THW3
        min_idx = min_idx.item()
        obj_num = obj_num.item()
        obj_ind = [int(a[0]) for a in obj_ind]
        obj_start_idx = [a.item() for a in obj_start_idx]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())

        preds = []
        preds.append(lab[:,0:1].contiguous())

        save_lab = torch.zeros(img.shape[1], obj_num+1, img.shape[-2], img.shape[-1]).cuda().float()
        save_lab[:,0,:] += 0.5

        obj_start_idx_set = set(obj_start_idx)
        for start in range(img.shape[1]):
            if start not in obj_start_idx_set:
                continue

            ref_lab = []
            obj_ind_tmp = []
            for k,s in enumerate(obj_start_idx):
                if s == start:
                    obj_ind_tmp.append(obj_ind[k])
                    ref_lab.append(lab[k:k+1])
            ref_lab = torch.cat(ref_lab, dim=0)
            ref_img = img[:ref_lab.shape[0],start,:]

            preds = [torch.zeros(ref_lab.shape).cuda().float().unsqueeze(1)]
            cnt = 0
            for k,s in enumerate(obj_start_idx):
                if s == start:
                    preds[0][cnt,0] = lab[k]
                    save_lab[s,obj_ind[k]] = lab[k]
                    cnt += 1

            ref_lab = ref_lab.max(0,keepdim=True)[0]
            with torch.no_grad():
                ms = model.forward(ref_img, ref_lab)

            for i in tqdm(range(start+1, img.shape[1])):
                with torch.no_grad():
                    flow = flow_infer.infer(ori_img[i], ori_img[i-1])

                prev_lab = utils.flow_warp_tensor(preds[-1], flow)

                merge_preds = prev_lab.max(0)[0]
                with torch.no_grad():
                    output, _ = model.forward(img[:ref_lab.shape[0],i], merge_preds, prev_lab.squeeze(1), ref=ms)

                output = output.detach()

                preds.append(output.contiguous())
                for idx,ind in enumerate(obj_ind_tmp):
                    save_lab[i,ind,:] = output[idx,0]
        end_time = timeit.default_timer()
        end_time_cv2 = cv2.getTickCount()

        vid_time.append(end_time-start_time)
        vid_time_cv2.append((end_time_cv2-start_time_cv2)/cv2.getTickFrequency())
        vid_frm_num.append(img.shape[1])

        step = 1
        for i,lab in enumerate(save_lab):
            if i % step == 0:
                img = cv2.resize(ori_img[i], (ori_shape[1],ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                visualize.show_save_lab_savesmall(i+min_idx, img, lab.cpu().numpy(), sample_dir, save_dir,
                                                  show=args.show_img, save=True)

    logger.info(('sum', sum(vid_time), sum(vid_time_cv2), sum(vid_frm_num)))
    logger.info(('time', sum(vid_time)/sum(vid_frm_num), sum(vid_time_cv2)/sum(vid_frm_num)))


if __name__ == '__main__':
    main()



