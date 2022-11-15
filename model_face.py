import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary

class FaceRes(nn.Module):
  def __init__(self,pretrained, num_bins):
    super(FaceRes, self).__init__()

    self.face_res = pretrained

    self.face_res.fc = nn.Identity()

    self.pitch_fc = nn.Linear(2048, num_bins)
    self.yaw_fc = nn.Linear(2048, num_bins)

  def forward(self, x):
    x = self.face_res(x)

    pre_yaw_gaze = self.pitch_fc(x)
    pre_pitch_gaze = self.yaw_fc(x)

    return pre_pitch_gaze, pre_yaw_gaze

#if you check this network, try to start the code.
#model = GC(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
#summary(model, (1, 3, 60, 36)) #Input shape is your size.
