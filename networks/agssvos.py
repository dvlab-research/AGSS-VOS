import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from torchvision import models
import logging
import cv2
import timeit

class Encoder(nn.Module):
    def __init__(self, init_atn=False, freeze=False):
        super(Encoder, self).__init__()

        self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024
        self.res5 = resnet.layer4 # 1/32, 2048

        self.atn = nn.Sequential(
            nn.Conv2d(257, 256, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, bias=True)
        )

        if init_atn:
            self._initialize_weights([self.atn])

        if freeze:
            # freeze BNs
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False
                        p.track_running_stats = False

    def _initialize_weights(self, mods, zero=False):
        for s in mods:
            for m in s:
                if isinstance(m, nn.Conv2d):
                    if not zero:
                        m.weight.data.normal_(0, 0.01)
                    else:
                        m.weight.data.zero_()
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.zero_()
                    m.weight.data = interp_surgery(m)


    def forward(self, in_f, in_p, objr2=False):
        f = in_f
        p = torch.unsqueeze(in_p, dim=1).float() # add channel dim
        x = self.conv1(f) + self.conv1_p(p)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 64

        if objr2:
            p_s4 = F.upsample(p, r2.shape[-2:], mode='bilinear', align_corners=True)
            r2_atn = self.atn(torch.cat((r2, p_s4), dim=1))
            return r2, r2_atn

        r3 = self.res3(r2) # 1/8, 128
        r4 = self.res4(r3) # 1/16, 256
        r5 = self.res5(r4) # 1/32, 512

        return r5, r4, r3, r2

class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr
        if s.shape[-1] == pm.shape[-1]:
            m = s + pm
        else:
            m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m


class Decoder(nn.Module):
    def __init__(self, output_dim=1):
        super(Decoder, self).__init__()
        mdim = 256
        self.GC = GC(4096, mdim) # 1/32 -> 1/32
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(1024, mdim) # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, output_dim, kernel_size=(3,3), padding=(1,1), stride=1)

    def _p_norm(self, p):
        p = F.softmax(p,1)
        p = p[:,1:]
        bg = 1-p.max(0,keepdim=True)[0]
        p = torch.cat((bg,p),dim=0)
        p = torch.clamp(p, 1e-7, 1-(1e-7))
        p = p/(1-p)
        p = p/(p.sum(0,keepdim=True))
        p = p[1:]
        return p

    def forward(self, r5, x5, r4, r3, r2, r2_obj, r2_atn):
        x = torch.cat((r5, x5), dim=1)

        x = self.GC(x) 
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r            # out: 1/32, 64
        m4 = self.RF4(r4, m5) # out: 1/16, 64
        m3 = self.RF3(r3, m4) # out: 1/8, 64

        m3 = m3.expand(r2_atn.shape[0],-1,-1,-1)

        m3_cat = m3 * r2_atn

        m2_cat = self.RF2(r2_obj, m3_cat) # out: 1/4, 64

        p2 = self.pred2(F.relu(m2_cat))
        p = F.upsample(p2, scale_factor=4, mode='bilinear')
        p = self._p_norm(p)

        return p


class AGSSVOS(nn.Module):
    def __init__(self, output_dim=2, init_atn=True, freeze=True):
        super(AGSSVOS, self).__init__()
        self.Encoder = Encoder(init_atn=init_atn, freeze=freeze)
        self.Decoder = Decoder(output_dim=output_dim)

    def forward(self, x, l_merge, l_obj=None, ref=None):
        if ref is None:
            r5, r4, r3, r2 = self.Encoder.forward(x[0:1], l_merge)
            return r5
        else:
            r5, r4, r3, r2 = self.Encoder.forward(x[0:1], l_merge)
            r2_obj, r2_atn = self.Encoder.forward(x, l_obj, objr2=True)
            p = self.Decoder.forward(r5, ref, r4, r3, r2, r2_obj, r2_atn)

            return p, r5

