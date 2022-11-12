import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary

class AttnNet(nn.Module):
  def __init__(self, pretrained, num_bins):
    super(AttnNet, self).__init__()


    #feature extraction
    self.face_res = pretrained
    self.eye_res = pretrained
    self.face_res.fc = nn.Identity()
    self.eye_res.fc = nn.Identity()

    #Attention
    self.hidden_dim = 256

    self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.tanh = nn.Tanh()
    self.v = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
    nn.init.normal_(self.v, 0, 0.1)

    self.softmax = nn.Softmax(dim=-1)

    #Estimate
    self.facefc_pitch = nn.Linear(2048, num_bins)
    self.facefc_yaw = nn.Linear(2048, num_bins)

    self.eyefc_pitch = nn.Linear(2048, num_bins)
    self.eyefc_yaw = nn.Linear(2048, num_bins)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x1, x2, x3):
    ff = self.face_res(x1)
    lf = self.eye_res(x2)
    rf = self.eye_res(x3)

    #Attention Component
    #AdditiveAttention
    ml = self.W1(ff) + self.W2(lf)
    ml = self.tanh(ml)
    ml = ml @ self.v

    mr = self.W1(ff) + self.W2(rf)
    mr = self.tanh(mr)
    mr = mr @ self.v

    m = ml + mr
    w = self.softmax(m)

    wl, wr = ml/w, mr/w

    #Gaze Estimation
    gb_pitch = self.facefc_pitch(ff)
    gb_yaw = self.facefc_yaw(ff)

    fe = wl*lf + wr*rf
    gr_pitch = self.eyefc_pitch(fe)
    gr_yaw = self.eyefc_yaw(fe)

    return gb_pitch, gb_yaw, gr_pitch, gr_yaw


model = AttnNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
