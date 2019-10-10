import cv2
import numpy as np
import random
from scipy.ndimage import distance_transform_edt as Dte
import torch


def resize(image, new_size, label=False):
  r"""
  resize a image to make the longer size match the new size
  :param image: both HW3 or HW1 ok
  :param new_size: an int, s, denote the value of the longer size
  :return: resized image
  """
  if max(image.shape[0], image.shape[1]) == new_size:
    return image.copy()
  else:
    if image.shape[0] >image.shape[1]:
      dh = new_size
    else:
      dw = new_size
    if label:
      img = cv2.resize(image, dsize=(dw, dh), interpolation=cv2.INTER_NEAREST)
    else:
      img = cv2.resize(image, dsize=(dw, dh), interpolation=cv2.INTER_LINEAR)
    return img

def resize_tuple(image, new_size, label=False):
  r"""
  resize a image to make the longer size match the new size
  :param image: both HW3 or HW1 ok
  :param new_size: tuple, (h,w)
  :return: resized image
  """
  if (image.shape[0], image.shape[1]) == new_size:
    return image.copy()
  else:
    if image.shape[0]*1./new_size[0] > image.shape[1]*1./new_size[1]:
      dh = new_size[0]
      dw = dh * image.shape[1] // image.shape[0]
    else:
      dw = new_size[1]
      dh = dw * image.shape[0] // image.shape[1]
    if label:
      img = cv2.resize(image, dsize=(dw, dh), interpolation=cv2.INTER_NEAREST)
    else:
      img = cv2.resize(image, dsize=(dw, dh), interpolation=cv2.INTER_LINEAR)
    return img


def resize_square(image, new_size, label=False):
  r"""
  resize a image to make the both size match the new size
  :param image: both HW3 or HW1 ok
  :param new_size: an int, s, denote the value of the new size
  :return: resized image
  """
  if label:
    img = cv2.resize(image, dsize=(new_size, new_size), interpolation=cv2.INTER_NEAREST)
  else:
    img = cv2.resize(image, dsize=(new_size, new_size), interpolation=cv2.INTER_LINEAR)
  return img

def resize_scale(image, scale, label=False):
  r"""
  resize a image to make the longer size match the new size
  :param image: both HW3 or HW1 ok
  :param scale: an int, s, denote the resize scale of image
  :return: resized image
  """
  if image.shape[0] % scale != 0 or image.shape[1] % scale != 0:
    assert image.shape[0] % scale == 0
    assert image.shape[1] % scale == 0
  dh = image.shape[0] / scale
  dw = image.shape[1] / scale
  if label:
    img = cv2.resize(image, dsize=(dw, dh), interpolation=cv2.INTER_NEAREST)
  else:
    img = cv2.resize(image, dsize=(dw, dh), interpolation=cv2.INTER_LINEAR)
  return img


def crop_tensor(label, lab_idx=1, shift=10):
  """

  :param label: tensor, HW
  :param lab_idx:
  :param shift:
  :return:
  """
  coord = torch.nonzero((label>0.5) == lab_idx)
  if coord.shape[0] == 0:
    return None, None, None, None
  h1,h2 = coord[:,0].min(), coord[:,0].max()
  w1,w2 = coord[:,1].min(), coord[:,1].max()
  det_h1 = det_h2 = det_w1 = det_w2 = shift
  h1 = max(h1.item()-det_h1, 0)
  h2 = min(h2.item()+det_h2, label.shape[0]-1)
  w1 = max(w1.item()-det_w1, 0)
  w2 = min(w2.item()+det_w2, label.shape[1]-1)
  return h1, h2, w1, w2


def crop(label, lab_idx=255, scale=0.2, shift=10, mode='scale', in_scale=0.1, out_scale=1.0, out_shift=300):
  r"""
  return a croped roi with a given scale/shift
  :param label: HW 2D, numpy
  :param lab_idx:
  :param scale:
  :param mode: 'scale' or 'shift'
  :return: the leftest coord and rightest coord, closed interval
  """
  coord = np.nonzero((label>0.5) == lab_idx)
  h1,h2 = coord[0].min(), coord[0].max()
  w1,w2 = coord[1].min(), coord[1].max()
  if mode == 'scale':
    det_h = int((h2-h1)*scale)
    det_w = int((w2-w1)*scale)
    det_h1 = det_h2 = det_h
    det_w1 = det_w2 = det_w
  elif mode == 'shift':
    det_h = shift
    det_w = shift
    det_h1 = det_h2 = det_h
    det_w1 = det_w2 = det_w
  elif mode == 'jitter':
    s = random.uniform(-scale, scale)
    det_h1 = int((h2-h1)*s) + random.uniform(0, out_shift)
    s = random.uniform(-scale, scale)
    det_h2 = int((h2-h1)*s) + random.uniform(0, out_shift)
    s = random.uniform(-scale, scale)
    det_w1 = int((w2-w1)*s) + random.uniform(0, out_shift)
    s = random.uniform(-scale, scale)
    det_w2 = int((w2-w1)*s) + random.uniform(0, out_shift)
  elif mode == 'none':
    det_h1 = det_w1 = det_h2 = det_w2 = 0
  elif mode == 'jitter_in_out':
    s = random.uniform(-in_scale, out_scale)
    det_h1 = int((h2-h1)*s)
    s = random.uniform(-in_scale, out_scale)
    det_h2 = int((h2-h1)*s)
    s = random.uniform(-in_scale, out_scale)
    det_w1 = int((w2-w1)*s)
    s = random.uniform(-in_scale, out_scale)
    det_w2 = int((w2-w1)*s)
  else:
    assert False, mode
  h1 = max(h1-det_h1, 0)
  h2 = min(h2+det_h2, label.shape[0]-1)
  w1 = max(w1-det_w1, 0)
  w2 = min(w2+det_w2, label.shape[1]-1)
  return h1, h2, w1, w2


def crop_wh_ratio(label, lab_idx=255, shift=10, ratio=2):
  """
  give a crop box wrt given ratio of wh
  :param label:
  :param lab_idx:
  :param shift:
  :param ratio: w:h = ratio:1
  :return:
  """
  coord = np.nonzero((label>0.5) == lab_idx)
  h1,h2 = coord[0].min(), coord[0].max()
  w1,w2 = coord[1].min(), coord[1].max()

  det_h = shift
  det_w = shift
  det_h1 = det_h2 = det_h
  det_w1 = det_w2 = det_w

  h1 = max(h1-det_h1, 0)
  h2 = min(h2+det_h2, label.shape[0]-1)
  w1 = max(w1-det_w1, 0)
  w2 = min(w2+det_w2, label.shape[1]-1)


  return h1, h2, w1, w2


def norm(image,
         mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
         scale=255.):
  r"""

  :param image: numpy
  :param mean:
  :param std:
  :param scale:
  :return: numpy, float
  """
  img = image[:, :, ::-1].astype(np.float)
  img /= scale
  img -= mean
  img /= std
  return img

def prepare(img, label=False):
  """

  :param img: img(3d) or lab(2d), np
  :param label: whether label
  :return:
  """
  if not label:
    img = norm(img.copy())
    img = torch.FloatTensor(img).cuda().float()
    img = img.transpose(1,2).transpose(0,1)
    return img.unsqueeze(0)
  else:
    lab = img.copy()
    lab = torch.FloatTensor(lab).cuda().float()
    return lab.unsqueeze(0)


def norm_4d(image, mean, std, scale=255.):
  r"""

  :param image: numpy, nchw
  :param mean:
  :param std:
  :param scale:
  :return: numpy, float
  """
  img = image[:, :, :, ::-1].astype(np.float)
  img /= scale
  img -= mean
  img /= std
  return img


def add_edge(label, kernel_erode, kernel_dilate, edge_value=255):
  r"""
  if kernel=0, means it do not need to dilate/erode
  :param label: HW, numpy, uint8
  :param kernel_erode: >=0
  :param kernel_dilate: >=0
  :return: uint8
  """
  if kernel_dilate > 0:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dilate, kernel_dilate))
    lab_dilate = cv2.dilate(label, kernel)
  else:
    lab_dilate = label.copy()
  if kernel_erode > 0:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_erode, kernel_erode))
    lab_erode = cv2.erode(label, kernel)
  else:
    lab_erode = label.copy()
  edge = lab_dilate - lab_erode
  lab = label * (edge == 0) + edge * edge_value
  return lab

def get_gaussian_mask(shape, point_set, rho=10., reverse=False, bg_zero=False):
  """
  :param shape:
  :param point_set:
  :param rho:
  :param reverse: if True, means point in edge is (x,y), else (h,w)
  :return:
  """
  mask = np.ones(shape).astype(np.float)
  for p in point_set:
    if reverse:
      mask[p[1], p[0]] = 0
    else:
      mask[p[0], p[1]] = 0
  mask = Dte(mask)
  mask /= rho
  mask = np.exp(-mask * mask / 2.)
  if bg_zero:
    mask = np.minimum(mask, 1.)
  else:
    mask = 1 - np.minimum(mask, 1.)
  return mask

def gen_gaussian_map(label, num, edge_mask=255, bg_zero=False):
  """

  :param label:
  :param num:
  :param edge_mask:
  :param bg_zero: if True, the bg is filled with zero, else one
  :return:
  """
  if num > 0:
    edge = (label == edge_mask).astype(np.uint8)
    edge = np.nonzero(edge)
    edge = np.array(edge).transpose()
    if edge.shape[0] == 0:
      assert 'no edge'
    step = edge.shape[0] / num
    if step == 0:
      step +=1
    edge = edge[::step]
    mask = get_gaussian_mask(label.shape, edge, bg_zero=bg_zero)
  else:
    if bg_zero:
      mask = np.zeros(label.shape).astype(np.float)
    else:
      mask = np.ones(label.shape).astype(np.float)
  return mask

def get_center_point(label_ori, lab_idx=1):
  """
  get a center point of a 2D map
  :param label_ori: HW, 2D, Tensor
  :param lab_idx: value of the object
  :return: list with length=1, [(h,w)]
  """
  label = (label_ori == lab_idx).cpu().numpy().astype(np.float)
  shape = label.shape
  lab_pad = cv2.copyMakeBorder(label, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                               value=(0.))
  lab_pad = Dte(lab_pad)
  lab_pad = lab_pad[1:-1, 1:-1].view()
  idx = np.argmax(lab_pad)
  coord = (idx / shape[1], idx % shape[1])
  return [coord]

def mask2box(mask, mode='none'):
  h1,h2,w1,w2 = crop((mask>0.5).astype(np.uint8), lab_idx=1, mode=mode)
  return np.array([w1,h1,w2,h2])

def get_dsize(image, half_size=False, scale=8, max_size=700):
  if not half_size or max(image.shape[0], image.shape[1])<=max_size:
    dh = image.shape[0]//scale*scale
    dw = image.shape[1]//scale*scale
  else:
    dh = image.shape[0]//2//scale*scale
    dw = image.shape[1]//2//scale*scale
  if dh*dw > 1280*720:
    dh = image.shape[0]//4*3
    dw = image.shape[1]//4*3
    dh = dh//8*8
    dw = dw//8*8
  if dh*dw > 1280*720:
    if image.shape[0] > image.shape[1]:
      dh = 640
      dw = 320
    else:
      dw = 640
      dh = 320
  dw = max(dw, scale)
  dh = max(dh, scale)
  dsize = (dw,dh)
  return dsize

def get_dsize_align(image, longsize=640, scale=32):
  if max(image.shape[0], image.shape[1]) == longsize:
    dh = image.shape[0]
    dw = image.shape[1]
  else:
    if image.shape[0] > image.shape[1]:
      dh = longsize
      dw = image.shape[1] * longsize // image.shape[0]
    else:
      dw = longsize
      dh = image.shape[0] * longsize // image.shape[1]
  dsize = (dw//scale*scale,dh//scale*scale)
  return dsize


def get_dsize_ratio(image, ratio=0.5, scale=8):
  h = image.shape[0]
  w = image.shape[1]
  dh = int(h*ratio) // scale*scale
  dw = int(w*ratio) // scale*scale
  dsize = (dw,dh)
  return dsize


def mask2rect(mask, context=50):
  box = mask2box(mask)
  box[0] = max(box[0]-context, 0)
  box[1] = max(box[1]-context, 0)
  box[2] = min(box[2]+context, mask.shape[1]-1)
  box[3] = min(box[3]+context, mask.shape[0]-1)
  ret = np.zeros(mask.shape, dtype=np.uint8)
  ret[box[1]:box[3]+1, box[0]:box[2]+1] = 1
  return ret
