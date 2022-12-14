import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary

class GN(nn.Module):
  def __init__(self, block, layers, image_channels, num_bins):
    super(GN, self).__init__()

    self.in_channels = 64

    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu =nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(
      block, layers[0], first_conv_out_channels=64, stride=1
    )
    self.layer2 = self._make_layer(
      block, layers[1], first_conv_out_channels=128, stride=2
    )
    self.layer3 = self._make_layer(
      block, layers[2], first_conv_out_channels=256, stride=2
    )
    self.layer4 = self._make_layer(
      block, layers[3], first_conv_out_channels=512, stride=2
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    self.eyefc_pitch = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5)
    )
    self.facefc_pitch = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(256, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5)
    )
    self.eyefc_yaw = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5)
    )
    self.facefc_yaw = nn.Sequential(
      nn.Linear(512 * block.expansion, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(256, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5)
    )
    self.fc_yaw_gaze = nn.Linear(256 + 256 + 32, num_bins)
    self.fc_pitch_gaze = nn.Linear(256 + 256 + 32, num_bins)


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
    x1 = self.conv1(x1)
    x1 = self.bn1(x1)
    x1 = self.relu(x1)
    x1 = self.maxpool(x1)
    x1 = self.layer1(x1)
    x1 = self.layer2(x1)
    x1 = self.layer3(x1)
    x1 = self.layer4(x1)
    x1 = self.avgpool(x1)
    x1 = x1.view(x1.shape[0], -1)
    #print(x1.shape)

    #Get Eye Featuresfrom tkinter.colorchooser import askcolor
    x2 = self.conv1(x2)
    x2 = self.bn1(x2)
    x2 = self.relu(x2)
    x2 = self.maxpool(x2)
    x2 = self.layer1(x2)
    x2 = self.layer2(x2)
    x2 = self.layer3(x2)
    x2 = self.layer4(x2)
    x2 = self.avgpool(x2)
    x2 = x2.view(x2.shape[0], -1)
    #print(x2.shape)

    x3 = self.conv1(x3)
    x3 = self.bn1(x3)
    x3 = self.relu(x3)
    x3 = self.maxpool(x3)
    x3 = self.layer1(x3)
    x3 = self.layer2(x3)
    x3 = self.layer3(x3)
    x3 = self.layer4(x3)
    x3 = self.avgpool(x3)
    x3 = x3.view(x3.shape[0], -1)
    #print(x3.shape)

    p_x1 = self.facefc_pitch(x1)
    p_x2 = self.eyefc_pitch(x2)
    p_x3 = self.eyefc_pitch(x3)

    y_x1 = self.facefc_yaw(x1)
    y_x2 = self.eyefc_yaw(x2)
    y_x3 = self.eyefc_yaw(x3)

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
#model = GN(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
#y = torch.rand(4, 3, 224, 224)
#print(model(y)[0].shape)
#summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
