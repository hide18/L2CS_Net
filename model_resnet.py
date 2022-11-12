import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary

class Res(nn.Module):
  def __init__(self, pretrained, num_bins):
    super(Res, self).__init__()

    self.face_res = pretrained
    self.eye_res = pretrained
    self.face_res.fc = nn.Identity()
    self.eye_res.fc = nn.Identity()

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    self.eyefc_pitch = nn.Sequential(
      nn.Linear(2048, 256),
      nn.ReLU(inplace=True)
    )
    self.facefc_pitch = nn.Sequential(
      nn.Linear(2048, 256),
      nn.ReLU(inplace=True)
    )
    self.eyefc_yaw = nn.Sequential(
      nn.Linear(2048, 256),
      nn.ReLU(inplace=True)
    )
    self.facefc_yaw = nn.Sequential(
      nn.Linear(2048, 256),
      nn.ReLU(inplace=True)
    )
    self.fc_yaw_gaze = nn.Linear(256 + 256 + 256, num_bins)
    self.fc_pitch_gaze = nn.Linear(256 + 256 + 256, num_bins)

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

    p_x1 = self.facefc_pitch(ff)
    p_x2 = self.eyefc_pitch(lf)
    p_x3 = self.eyefc_pitch(rf)

    y_x1 = self.facefc_yaw(ff)
    y_x2 = self.eyefc_yaw(lf)
    y_x3 = self.eyefc_yaw(rf)

    p_features = torch.cat((p_x1, p_x2, p_x3), 1)
    y_features = torch.cat((y_x1, y_x2, y_x3), 1)

    pre_pitch_gaze = self.fc_pitch_gaze(p_features)
    pre_yaw_gaze = self.fc_yaw_gaze(p_features)

    pre_pitch_gaze = self.fc_pitch_gaze(y_features)
    pre_yaw_gaze = self.fc_yaw_gaze(y_features)

    return pre_pitch_gaze, pre_yaw_gaze


model = Res(models.resnet50(pretrained=True), 180)
print(model)
