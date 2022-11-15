import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torchinfo import summary

class VGGGaze(nn.Module):
  def __init__(self, cfg, batch_norm=False, num_bins=90):
    super(VGGGaze, self).__init__()
    self.features = self._make_layers(cfg, batch_norm)
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.p_classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, num_bins),
    )
    self.y_classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(4096, num_bins),
    )

    self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    pitch = self.p_classifier(x)
    yaw = self.y_classifier(x)
    return pitch, yaw

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def _make_layers(self, cfg, batch_norm):
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


'''
cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

model = VGGGaze(cfg["E"])
print(model)
#summary(model, (1, 3, 448, 448))
'''
