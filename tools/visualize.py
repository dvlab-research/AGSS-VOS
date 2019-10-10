import cv2
import numpy as np


def denorm_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale=255.):
  r"""
  img_nrom = (img/scale-mean)/std
  show a image that has been normalized
  :param image: 3xHxW, numpy, float
  :param mean:
  :param std:
  :return: original image, np.uint8
  """
  img = image.transpose((1,2,0))
  img = (img*std+mean)*scale
  img = img[:,:,::-1].astype(np.uint8)
  return img


def vis_label(label, vis_index, vis_value):
  r"""
  label[label==vis_index] = vis_value
  :param label: HxW, numpy
  :param vis_index: [0,255]
  :param vis_value: [0,255]
  :return:
  """
  lab = label.copy()
  lab[label == vis_index] = vis_value
  lab = lab.astype(np.uint8)
  return lab

def vis_mask(mask, scale):
  """
  mask *= scale
  :param mask: HW 2D, numpy
  :param scale: [0,255], int
  :return:
  """
  ret = mask*scale
  ret = ret.astype(np.uint8)
  return ret

def vis_flow(flow, scale=50.):
  flow = flow.copy()*scale
  hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
  hsv[...,1] = 255

  mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

  return bgr


def show_save_lab_savesmall(idx, img_ori, cur_lab, sample_dir=None, save_dir=None, show=True, save=False):
  lab_c = np.argmax(cur_lab, axis=0)
  lab_c = cv2.resize(lab_c, (img_ori.shape[1], img_ori.shape[0]), interpolation=cv2.INTER_NEAREST)
  if save:
    cv2.imwrite('%s/%05d.png' % (save_dir,idx), lab_c)
  if not show:
    return
  img = img_ori.copy().astype(np.float)
  max_c = lab_c.max()
  rat = 0.3
  base = 255
  ### k==0 denote background ###
  for k in range(1, max_c+1):
    if k == 1:
      # img[lab_c==k, 0] = 255
      img[lab_c==k, 0] *= rat
      img[lab_c==k, 0] += (1-rat)*base
    elif k == 2:
      img[lab_c==k, 1] *= rat
      img[lab_c==k, 1] += (1-rat)*base
      # img[lab_c==k, 1] = 255
    elif k == 3:
      # img[lab_c==k, 0] = 255
      img[lab_c==k, 2] *= rat
      img[lab_c==k, 2] += (1-rat)*base
      # img[lab_c==k, 2] *= rat
      # img[lab_c==k, 2] += (1-rat)*base
    elif k == 4:
      # img[lab_c==k, 0] = 0
      img[lab_c==k, 1] *= rat
      img[lab_c==k, 1] += (1-rat)*128
    elif k == 5:
      img[lab_c==k, 1] = 0
    elif k == 6:
      img[lab_c==k, 2] = 0
    elif k == 7:
      img[lab_c==k, 0] = 128
    elif k == 8:
      img[lab_c==k, 1] = 128
    elif k == 9:
      img[lab_c==k, 2] = 128
    elif k == 10:
      img[lab_c==k, 0] = 90
    else:
      assert 1<0
  cv2.imwrite('%s/%05d.jpg' % (sample_dir, idx), img)

