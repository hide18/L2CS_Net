import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary
from collections import OrderedDict

class Res(nn.Module):
  def __init__(self, block, layers, image_channels, num_bins):
    super(Res, self).__init__()

    self.in_channels = 64

    self.face_conv = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
      ('bn1', nn.BatchNorm2d(64)),
      ('relu', nn.ReLU(inplace=True)),
      ('maxppol', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
      ('layer1', self._make_layer(block, layers[0], first_conv_out_channels=64, stride=1)),
      ('layer2', self._make_layer(block, layers[1], first_conv_out_channels=128, stride=2)),
      ('layer3', self._make_layer(block, layers[2], first_conv_out_channels=256, stride=2)),
      ('layer4', self._make_layer(block, layers[3], first_conv_out_channels=512, stride=2)),
      ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    ]))

    self.in_channels = 64

    self.eye_conv = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
      ('bn1', nn.BatchNorm2d(64)),
      ('relu', nn.ReLU(inplace=True)),
      ('maxppol', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
      ('layer1', self._make_layer(block, layers[0], first_conv_out_channels=64, stride=1)),
      ('layer2', self._make_layer(block, layers[1], first_conv_out_channels=128, stride=2)),
      ('layer3', self._make_layer(block, layers[2], first_conv_out_channels=256, stride=2)),
      ('layer4', self._make_layer(block, layers[3], first_conv_out_channels=512, stride=2)),
      ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    ]))

    self.eyefc_pitch = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.ReLU(inplace=True)
    )
    self.facefc_pitch = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.ReLU(inplace=True)
    )
    self.eyefc_yaw = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.ReLU(inplace=True)
    )
    self.facefc_yaw = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.ReLU(inplace=True)
    )
    self.fc_yaw_gaze = nn.Linear(256 + 256 + 256, num_bins)
    self.fc_pitch_gaze = nn.Linear(256 + 256 + 256, num_bins)


    #self.fc_yaw_gaze = nn.Linear(512 * block.expansion * 3, num_bins)
    #self.fc_pitch_gaze = nn.Linear(512 * block.expansion * 3, num_bins)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


  def forward(self, x1, x2, x3):
    #Get Face Features
    ff = self.face_conv(x1)
    ff = ff.view(ff.shape[0], -1)
    #print(x1.shape)

    #Get Eye Featuresfrom tkinter.colorchooser import askcolor
    lf = self.eye_conv(x2)
    lf = lf.view(lf.shape[0], -1)
    #print(x2.shape)

    rf = self.eye_conv(x3)
    rf = rf.view(rf.shape[0], -1)
    #print(x3.shape)

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

  def _make_layer(self, block, num_res_blocks, first_conv_out_channels, stride):
    identity_conv = None
    layers = []
    if stride != 1 or self.in_channels != first_conv_out_channels*block.expansion:
      identity_conv = nn.Sequential(
        nn.Conv2d(self.in_channels, first_conv_out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(first_conv_out_channels * block.expansion)
      )

    layers.append(
      block(self.in_channels, first_conv_out_channels, stride, identity_conv)
    )

    self.in_channels = first_conv_out_channels * block.expansion

    for i in range(num_res_blocks - 1):
      layers.append(block(self.in_channels, first_conv_out_channels))

    return nn.Sequential(*layers)

#if you check this network, try to start the code.
model = Res(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
print(model)
#y = torch.rand(4, 3, 224, 224)
#print(model(y)[0].shape)
#summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
