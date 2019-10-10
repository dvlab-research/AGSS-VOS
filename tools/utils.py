import cv2
import os
import logging
import math
import numpy as np
import numpy.random as npr
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import timeit

def mkdir(*lis):
  r""" mkdir -p a new path
  :param lis: a dir set, like mkdir('aaa','bbb')
  :return: void
  """
  for s in lis:
    if not type(s) == type(None) and not os.path.exists(s):
      os.makedirs(s)


def lr_poly(base_lr, i_iter, max_iter, epoch, tot_epoch, power=0.9):
  r"""
  calc the lr of poly
  :param base_lr:
  :param i_iter:
  :param max_iter:
  :param epoch:
  :param tot_epoch:
  :param power:
  :return:
  """
  cur = epoch*max_iter + i_iter
  tot = max_iter * tot_epoch
  lr = base_lr * ((1 - float(cur) / tot) ** (power))
  return lr


def adjust_optim(optimizer, lr, start, end, scale=1):
  r"""
  optimizer[start~end-1] = lr*scale
  :param optimizer:
  :param lr:
  :param start:
  :param end:
  :param scale:
  :return:
  """
  for i in range(start, end):
    optimizer.param_groups[i]['lr'] = lr * scale

def adjust_optim_all(optimizer, lr, scale_lr=None, scale=10.):
  '''
  lr*scale for the specified param
  :param optimizer:
  :param lr:
  :param scale_lr:
  :param scale:
  :return:
  '''
  for idx, param_group in enumerate(optimizer.param_groups):
    if scale_lr is not None and scale_lr[idx]:
      param_group['lr'] = lr * scale
    else:
      param_group['lr'] = lr

def calc_remain_time(run_time, i_iter, max_iter, epoch, tot_epoch):
  r"""
  run_time is the running time of a unit of operation
  :param run_time:
  :param i_iter:
  :param max_iter:
  :param epoch:
  :param tot_epoch:
  :return:
  """
  remain = run_time*(max_iter-i_iter + max_iter*(tot_epoch-epoch-1))
  remain = int(remain) / 60
  min = remain % 60
  remain /= 60
  return '%.2f hour %.2f min' % (remain, min)


def flow_warp_tensor(mask, flow, coord=None):
  """

  :param mask: 1xCxHxW
  :param flow: HxWx2
  :return:
  """
  if coord is None:
    shape = mask.shape[-2:]
    coord = torch.ones(shape).cuda().long()
    coord = torch.nonzero(coord).float()
  flow = flow2coord(flow, coord)
  flow = flow.unsqueeze(0)
  if mask.shape[0] > 1:
    flow = flow.expand(mask.shape[0], -1, -1, -1)
  mask = f.grid_sample(mask, flow)
  return mask


def flow2coord(flow, coord, norm=True):
  if not isinstance(flow, torch.Tensor):
    flow = torch.Tensor(flow).cuda().float()
  shape = flow.shape[:2]

  coord2 = coord.clone()
  coord[:, 0] = coord2[:, 1]
  coord[:, 1] = coord2[:, 0]

  coord = coord.view(shape[0],shape[1],2)

  coord += flow
  if norm:
    coord[:,:,0] = (coord[:,:,0]-shape[1]/2.) / (shape[1]/2.)
    coord[:,:,1] = (coord[:,:,1]-shape[0]/2.) / (shape[0]/2.)

  return coord


def loss_calc_iou(pred, label, unify=False, optim_hard=False, square=False, eps=1e-7, ignore_index=255):
  """
  IoU = |min(P,Y)|/|max(P,Y)|
  :param pred: N1HW, variable, must bel [0,1]
  :param label: N1HW, tensor
  :return: variable with one value
  """
  if not unify:
    pred = pred.view(pred.shape[0], -1)
    label = label.view(label.shape[0], -1)
  gt = Variable((label==1), requires_grad=False).cuda().float()
  mask = Variable((label!=ignore_index), requires_grad=False).cuda().float()
  if unify:
    loss_seg = 1. - ((torch.min(gt, pred)*mask).sum()) / ((torch.max(gt, pred)*mask).sum()+eps)
  else:
    ones = Variable(torch.ones(pred.shape[0]), requires_grad=False).cuda().float()
    loss_seg = ones - ((torch.min(gt, pred)*mask).sum(1)) / ((torch.max(gt, pred)*mask).sum(1)+eps)
    if square:
      loss_seg = (loss_seg**2)/2
  if not unify and optim_hard:
    return torch.max(loss_seg)
  else:
    return torch.mean(loss_seg)


def loss_calc_iou_v2(pred, label, unify=False, optim_hard=False, square=False, eps=1e-7, ignore_index=255):
  """
  IoU = |P*Y|/|P+Y-P*Y|
  :param pred: N1HW, variable, must bel [0,1]
  :param label: N1HW, tensor
  :return: variable with one value
  """
  if not unify:
    pred = pred.view(pred.shape[0], -1)
    label = label.view(label.shape[0], -1)
  gt = Variable((label==1), requires_grad=False).cuda().float()
  mask = Variable((label!=ignore_index), requires_grad=False).cuda().float()
  if unify:
    loss_seg = 1. - (((gt*pred)*mask).sum()) / (((gt+pred-gt*pred)*mask).sum()+eps)
  else:
    ones = Variable(torch.ones(pred.shape[0]), requires_grad=False).cuda().float()
    intsec = gt*pred
    union = gt+pred-intsec
    loss_seg = ones - ((intsec*mask).sum(1)) / ((union*mask).sum(1)+eps)
    if square:
      loss_seg = (loss_seg**2)/2
  if not unify and optim_hard:
    return torch.max(loss_seg)
  else:
    return torch.mean(loss_seg)


def calc_iou(pred, label, threshold=0.5, ignore_index=255, merge=False):
  """
  calc the intersection over union
  :param pred: N1HW or NHW, Tensor
  :param label: NHW or N1HW, Tensor
  :return: intsec & union sum
  """
  pred = pred.view(pred.shape[0], -1)
  label = label.view(label.shape[0], -1)
  pred = (pred > threshold).long()
  mask = (label != ignore_index).long()
  if mask.sum() == 0:
    assert 1<0, (mask.sum())
  intsec = (label * pred) * mask
  union = (label + pred - intsec) * mask
  if merge:
    if union.sum().item() == 0:
      if intsec.sum().item() == 0:
        return 1.
      else:
        return 0.
    else:
      return intsec.sum().item()*1./union.sum().item()
  else:
    iou = []
    for i in range(intsec.shape[0]):
      if union[i].sum() == 0:
        iou.append(1. if intsec[i].sum()==0. else 0.)
      else:
        iou.append((intsec[i].sum().item()*1./union[i].sum().item()))
    return iou
