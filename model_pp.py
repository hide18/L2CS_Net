import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary
from collections import OrderedDict

class ResVgg(nn.Module):
  def __init__(self, block, layers, cfg, batch_norm=False, num_bins=90):
    super(ResVgg, self).__init__()

    #face
    self.in_channels = 64
    self.face_res = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
      ('bn1', nn.BatchNorm2d(64)),
      ('relu', nn.ReLU(inplace=True)),
      ('maxppol', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
      ('layer1', self._make_res_layer(block, layers[0], first_conv_out_channels=64, stride=1)),
      ('layer2', self._make_res_layer(block, layers[1], first_conv_out_channels=128, stride=2)),
      ('layer3', self._make_res_layer(block, layers[2], first_conv_out_channels=256, stride=2)),
      ('layer4', self._make_res_layer(block, layers[3], first_conv_out_channels=512, stride=2)),
      ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    ]))
    self.facefc_pitch = nn.Linear(512 * block.expansion, 256)
    self.facefc_yaw = nn.Linear(512 * block.expansion, 256)

    #eyes
    self.eye_vgg = nn.Sequential(OrderedDict([
      ('features', self._make_vgg_layers(cfg, batch_norm)),
      ('avgpool', nn.AdaptiveAvgPool2d((7, 7)))
    ]))

    self.p_classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 256),
    )
    self.y_classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 256),
    )

    self.fc_pitch_gaze = nn.Linear(256 + 256 + 256, num_bins)
    self.fc_yaw_gaze = nn.Linear(256 + 256 + 256, num_bins)


    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


  def forward(self, x1, x2, x3):
    #Get Face Features
    face = self.face_res(x1)
    face = face.view(face.shape[0], -1)
    p_face = self.facefc_pitch(face)
    y_face = self.facefc_yaw(face)

    #Get Eye features
    left = self.eye_vgg(x2)
    right = self.eye_vgg(x3)
    left = left.view(left.shape[0], -1)
    right = right.view(right.shape[0], -1)
    p_left = self.p_classifier(left)
    y_left = self.y_classifier(left)
    p_right = self.p_classifier(right)
    y_right = self.y_classifier(right)

    p_features = torch.cat((p_face, p_left, p_right), 1)
    y_features = torch.cat((y_face, y_left, y_right), 1)

    pitch = self.fc_pitch_gaze(p_features)
    yaw = self.fc_yaw_gaze(y_features)

    return pitch, yaw

  def _make_res_layer(self, block, num_res_blocks, first_conv_out_channels, stride):
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

  def _make_vgg_layers(self, cfg, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
      if v == "M":
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)

#if you check this network, try to start the code.
cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
model = ResVgg(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], cfg["E"], num_bins=180)
#y = torch.rand(4, 3, 224, 224)
#print(model(y)[0].shape)
summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
