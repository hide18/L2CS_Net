import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary

class AttnNet(nn.Module):
  def __init__(self, block, layers, image_channels, num_bins):
    super(AttnNet, self).__init__()


    #feature extraction
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

    self.feature_fc = nn.Linear(512 * block.expansion, 256)


    #Attention
    self.hidden_dim = 256

    self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.tanh = nn.Tanh()
    self.v = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
    nn.init.normal_(self.v, 0, 0.1)

    self.softmax = nn.Softmax(dim=-1)

    #Estimate
    self.facefc_pitch = nn.Linear(256, num_bins)
    self.facefc_yaw = nn.Linear(256, num_bins)

    self.eyefc_pitch = nn.Linear(256, num_bins)
    self.eyefc_yaw = nn.Linear(256, num_bins)

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
    face = self.relu(self.feature_fc(x1))

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
    left = self.relu(self.feature_fc(x2))

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
    right = self.relu(self.feature_fc(x3))

    #Attention Component
    #AdditiveAttention
    ml = self.W1(face) + self.W2(left)
    ml = self.tanh(ml)
    ml = ml @ self.v

    mr = self.W1(face) + self.W2(right)
    mr = self.tanh(mr)
    mr = mr @ self.v

    m = ml + mr
    w = self.softmax(m)

    wl, wr = ml/w, mr/w

    #Gaze Estimation
    gb_pitch = self.facefc_pitch(face)
    gb_yaw = self.facefc_yaw(face)

    fe = wl*left + wr*right
    gr_pitch = self.eyefc_pitch(fe)
    gr_yaw = self.eyefc_yaw(fe)

    return gb_pitch, gb_yaw, gr_pitch, gr_yaw

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

model = AttnNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
