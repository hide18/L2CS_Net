import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary
from collections import OrderedDict

class AttnNet(nn.Module):
  def __init__(self, block, layers, image_channels, num_bins):
    super(AttnNet, self).__init__()

    #feature extraction
    self.in_channels = 64
    self.face_res = nn.Sequential(OrderedDict([
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
    self.eye_res = nn.Sequential(OrderedDict([
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
    self.p_face_fc = nn.Linear(2048, 256)
    self.p_eye_fc = nn.Linear(2048, 256)
    self.y_face_fc = nn.Linear(2048, 256)
    self.y_eye_fc = nn.Linear(2048, 256)

    #Attention
    self.hidden_dim = 256

    self.p_W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.p_W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.y_W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.y_W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.tanh = nn.Tanh()
    self.p_v = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
    nn.init.normal_(self.p_v, 0, 0.1)
    self.y_v = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
    nn.init.normal_(self.y_v, 0, 0.1)

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
    ff = self.face_res(x1)
    ff = ff.view(ff.shape[0], -1)
    lf = self.eye_res(x2)
    lf = lf.view(lf.shape[0], -1)
    rf = self.eye_res(x3)
    rf = rf.view(rf.shape[0], -1)

    p_ff = self.p_face_fc(ff)
    p_lf = self.p_eye_fc(lf)
    p_rf = self.p_eye_fc(rf)

    y_ff = self.y_face_fc(ff)
    y_lf = self.y_eye_fc(lf)
    y_rf = self.y_eye_fc(rf)

    #Attention Component
    #AdditiveAttention
    p_ml = self.p_W1(p_ff) + self.p_W2(p_lf)
    p_ml = self.tanh(p_ml)
    p_ml = p_ml @ self.p_v
    p_mr = self.p_W1(p_ff) + self.p_W2(p_rf)
    p_mr = self.tanh(p_mr)
    p_mr = p_mr @ self.p_v

    p_m = p_ml + p_mr
    p_w = self.softmax(p_m)
    p_wl, p_wr = p_ml/p_w, p_mr/p_w

    y_ml = self.y_W1(y_ff) + self.y_W2(y_lf)
    y_ml = self.tanh(y_ml)
    y_ml = y_ml @ self.y_v
    y_mr = self.y_W1(y_ff) + self.y_W2(y_rf)
    y_mr = self.tanh(y_mr)
    y_mr = y_mr @ self.y_v

    y_m = y_ml + y_mr
    y_w = self.softmax(y_m)
    y_wl, y_wr = y_ml/y_w, y_mr/y_w

    #Gaze Estimation
    gb_pitch = self.facefc_pitch(p_ff)
    gb_yaw = self.facefc_yaw(y_ff)

    p_fe = p_wl*p_lf + p_wr*p_rf
    y_fe = y_wl*y_lf + y_wr*y_rf
    gr_pitch = self.eyefc_pitch(p_fe)
    gr_yaw = self.eyefc_yaw(y_fe)

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


#model = AttnNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
#summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
