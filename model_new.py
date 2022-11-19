import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary

class GC(nn.Module):
  def __init__(self, block, layers, image_channels, num_bins):
    super(GC, self).__init__()

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

    #new model plus the fc layer
    self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
    self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

    '''
    self.fc_yaw_gaze = nn.Sequential(
      nn.Linear(512 * block.expansion, 1000),
      nn.ReLU(inplace=True),
      nn.Linear(1000, num_bins)
    )
    self.fc_pitch_gaze = nn.Sequential(
      nn.Linear(512 * block.expansion, 1000),
      nn.ReLU(inplace=True),
      nn.Linear(1000, num_bins)
    )
    '''

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.shape[0], -1)

    pre_yaw_gaze = self.fc_yaw_gaze(x)
    pre_pitch_gaze = self.fc_pitch_gaze(x)

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
#model = GC(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
#summary(model, (1, 3, 60, 36)) #Input shape is your size.
