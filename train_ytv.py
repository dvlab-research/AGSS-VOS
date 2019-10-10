import sys
sys.path.insert(0, '/home/linhj/anaconda2/envs/pytorch3/lib/python3.6/site-packages')
sys.path.insert(0, '/data/linhj/anaconda2/envs/mypy3/lib/python3.6/site-packages')

import os
import time
import logging
import numpy as np
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset.vos import Trainset
from networks.agssvos import AGSSVOS
sys.path.append('flow_inference')
from flow_inference.flow_inference import Inference_flow
from tools import preprocess, visualize, utils
import timeit
import cv2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--root-data', type=str)
    parser.add_argument('--root-all-data', type=str)
    parser.add_argument('--meta-list', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--init-atn', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--set-bn-no-update', action='store_true')
    parser.add_argument('--random-crop', action='store_true')
    parser.add_argument('--iou-thr-per-obj', action='store_true')
    parser.add_argument('--lr-atn', action='store_true')
    parser.add_argument('--lr-after-atn', action='store_true')
    parser.add_argument('--three-frames-data', action='store_true')
    parser.add_argument('--loss-iou-maxmin', action='store_true')
    parser.add_argument('--random-ref', action='store_true')
    parser.add_argument('--random-skip', action='store_true')
    parser.add_argument('--step-size', type=float, default=4)
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
    parser.add_argument('--fix-lr', type=int, default=0)
    parser.add_argument('--show-img', action='store_true', help='show intermediate result')
    return parser

# get logger
def get_logger():
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def show(images, labels, preds, prev_labs):
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

        for j_bs in range(prev_labs.shape[1]):
            path = args.sample_dir + '/' + str(i_bs)+'_'+str(j_bs+2) + '#'
            prev_lab = prev_labs[i_bs,j_bs]*255
            cv2.imwrite(path + 'parev_lab.jpg', prev_lab)

def main():
    global args, logger, writer
    args = get_parser().parse_args()
    logger_train = get_logger()
    random.seed(20170624)
    logger_train.info((args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    utils.mkdir(args.snapshot_dir, args.sample_dir)

    # setting up model
    model = AGSSVOS(init_atn=args.init_atn, freeze=args.freeze).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    for m in model.module.Encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if args.set_bn_no_update:
                for p in m.parameters():
                    p.requires_grad = False

    if args.restore != None:
        assert os.path.isfile(args.restore), "no restore file found at %s" % (args.restore)
        logger_train.info("loading from %s" % (args.restore))

        state = model.state_dict()
        checkpoint = torch.load(args.restore)
        if args.finetune:
            checkpoint = checkpoint['seg']
        checkpoint = {k: v for k, v in checkpoint.items() if k in state}
        state.update(checkpoint)
        model.load_state_dict(state)

        del checkpoint
        torch.cuda.empty_cache()

    if args.finetune:
        flow_infer = Inference_flow(args, train_flow=True, resume=args.restore)
    else:
        flow_infer = Inference_flow(args, train_flow=True)

    params = []
    scale_lr = []
    assert args.lr_atn != args.lr_after_atn
    for key, value in dict(model.module.named_parameters()).items():
        if args.lr_atn and ('atn' in key or 'pred2' in key or 'RF2' in key):
            flag = True
        elif args.lr_after_atn and ('atn' in key or 'pred2' in key or 'RF2' in key):
            flag = True
        else:
            flag = False
        if value.requires_grad:
            if flag:
                scale_lr.append(True)
                print('lrx10', key)
            else:
                scale_lr.append(False)
            params += [{'params':[value],'lr':args.lr*10 if flag else args.lr , 'weight_decay': 4e-5}]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=4e-5)
    spec_vid = None
    spec_obj_ind = None

    trainloader = data.DataLoader(
            Trainset(root_data=args.root_data, json_meta_list=args.meta_list,
                         sample_size=args.sample_size, test_mode=False, spec_vid=spec_vid, spec_obj_ind=spec_obj_ind,
                         step=5, fix_size=False, half_size=True, random_ref=args.random_ref, random_skip=args.random_skip),
            batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)


    # training
    tot_iter = len(trainloader)
    logger_train.info("Total iteration per epoch is %d" % (tot_iter))
    tot_time = []
    loss_set = []
    iou_set = []
    optimizer.zero_grad()

    for epoch in range(args.start_epoch, args.epoch):
        for i_iter, batch in enumerate(trainloader):
            start_time = timeit.default_timer()

            img, lab, ori_img = batch

            img = img[0].cuda().float()
            lab = lab[0].cuda().float()
            ori_img = ori_img.numpy()
            # img KT3HW, lab KTHW, ori_img, KTHW3#

            ### It may be better to move this augmentation into the dataset preprocessing ##
            if random.uniform(0,1)>0.5 and args.random_crop:
                ### random resize ###
                coord = [1e4,1e4,0,0]
                lab_agno = lab.sum(0)
                val_cnt = 0
                for i in range(lab_agno.shape[0]):
                    idx = torch.nonzero(lab_agno[i]>0)
                    if idx.shape[0] == 0:
                        continue
                    val_cnt += 1
                    h0 = idx[:,0].min().item()
                    w0 = idx[:,1].min().item()
                    h1 = idx[:,0].max().item()
                    w1 = idx[:,1].max().item()
                    coord[0] = min(coord[0], h0)
                    coord[1] = min(coord[1], w0)
                    coord[2] = max(coord[2], h1)
                    coord[3] = max(coord[3], w1)
                if val_cnt < 2:
                    logger_train.info(('The number of frames that have label is less than 2, continue..'))
                    continue
                ori_shape = lab.shape[-2:]
                rand_coord = [0]*4

                if random.uniform(0,1) > 0.3:
                    scale = random.uniform(0,1)
                else:
                    scale = 1
                rand_coord[0] = coord[0] * scale
                rand_coord[1] = coord[1] * scale
                rand_coord[2] = (ori_shape[0]-coord[2]-1)*(1-scale)+coord[2]+1
                rand_coord[3] = (ori_shape[1]-coord[3]-1)*(1-scale)+coord[3]+1
                for j in range(4):
                    rand_coord[j] = int(rand_coord[j])

                old_img = img.clone()
                old_lab = lab.clone()
                ori_img = torch.FloatTensor(ori_img).cuda().transpose(-1,-2).transpose(-2,-3)
                old_ori_img = ori_img.clone()

                old_lab = old_lab[:,:,rand_coord[0]:rand_coord[2]+1,rand_coord[1]:rand_coord[3]+1]
                lab = F.upsample(old_lab, ori_shape, mode='bilinear', align_corners=True)
                lab = (lab>0.5).float()
                for i in range(img.shape[0]):
                    img_obj = old_img[i,:,:,rand_coord[0]:rand_coord[2]+1,rand_coord[1]:rand_coord[3]+1]
                    img[i] = F.upsample(img_obj, ori_shape, mode='bilinear', align_corners=True)
                img_obj = old_ori_img[0,:,:,rand_coord[0]:rand_coord[2]+1,rand_coord[1]:rand_coord[3]+1]
                ori_img[0] = F.upsample(img_obj, ori_shape, mode='bilinear', align_corners=True)
                ori_img = ori_img.transpose(-2,-3).transpose(-1,-2).cpu().numpy().astype(np.uint8)

                ### end of random resize ###

            if lab.shape[1] == 1:
                logger_train.info('lab.shape[1](vid_len) == 1, continue..')
                continue

            lr = utils.lr_poly(args.lr, i_iter, tot_iter, epoch, args.epoch)
            utils.adjust_optim_all(optimizer, lr, scale_lr)
            preds = []
            prev_labs = []
            preds.append(lab[:,0:1].contiguous())
            preds.append(lab[:,1:2].contiguous())
            merge_preds_ref = lab[:,0:1].contiguous().sum(0)
            for i in range(2, img.shape[1], 1):
                ms = model.forward(img[:,0], merge_preds_ref)
                flow = flow_infer.infer(ori_img[0,i], ori_img[0,i-1])
                prev_lab = utils.flow_warp_tensor(preds[i-1], flow)

                prev_labs.append(prev_lab.detach())
                merge_preds = prev_lab.max(0)[0]

                output, _ = model.forward(img[:,i], merge_preds, prev_lab.squeeze(1), ref=ms)

                cur_lab = lab[:,i].contiguous()

                if args.loss_iou_maxmin:
                    cur_loss = utils.loss_calc_iou(output, cur_lab.unsqueeze(1), unify=False, optim_hard=False,
                                                      square=False) # try this
                else:
                    cur_loss = utils.loss_calc_iou_v2(output, cur_lab.unsqueeze(1), unify=False, optim_hard=False,
                                                      square=False) # try this

                loss_set.append(cur_loss.item())

                iou = utils.calc_iou(output.data, cur_lab.long(), merge=False)
                iou_set.append(np.mean(iou))

                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

                if args.iou_thr_per_obj:
                    output = output.detach()
                    new_output = torch.zeros_like(output).cuda().float()
                    for j in range(new_output.shape[0]):
                        if iou[j] > 0.5:
                            new_output[j] = output[j]
                        else:
                            new_output[j] = lab[j:j+1,i]
                    new_output = new_output.contiguous()
                    preds.append(new_output.detach())
                else:
                    if np.mean(iou) > 0.5:
                        preds.append(output.detach())
                    else:
                        preds.append(cur_lab.unsqueeze(1).detach())

            end_time = timeit.default_timer()
            tot_time.append(end_time - start_time)

            if i_iter % 200 == 0:
                logger_train.info('show at %s' % args.sample_dir)
                try:
                    preds = torch.cat(preds, dim=1)
                    prev_labs = torch.cat(prev_labs, dim=1)
                except Exception as e:
                    print(e)
                    print('Ignore.. Continue..')
                    continue
                if args.show_img:
                    show(img.data.cpu().numpy(), lab.data.cpu().numpy(), preds.data.cpu().numpy().astype(np.float),
                        prev_labs.data.cpu().numpy().astype(np.float32))

            if i_iter % 20 == 0:
                run_time = np.mean(tot_time)
                rem_time = utils.calc_remain_time(run_time, i_iter, tot_iter, epoch, args.epoch)
                logger_train.info('iter = %d of %d in epoch = %d of %d, remain_time = %s' %
                            (i_iter, tot_iter, epoch, args.epoch, rem_time))
                tot_time = []
                logger_train.info('lr = %f, loss = %f, iou = %f' % (lr, np.mean(loss_set), np.mean(iou_set)))
                loss_set = []
                iou_set = []


        if epoch % (args.epoch//5) == 0 or epoch == args.epoch - 1:
            path = os.path.join(args.snapshot_dir, 'model_' + str(epoch) + '.pth')
            logger_train.info('save model at %s' % path)
            torch.save({'seg':model.state_dict(), 'flow':flow_infer.model.state_dict()}, path)


if __name__ == '__main__':
    main()



