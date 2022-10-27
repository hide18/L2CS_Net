import os
import numpy as np
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter

class Gaze360(Dataset):
  def __init__(self, label_path, image_root, transform_face, transform_eye, angle, binwidth, train=True):
    self.transform_face = transform_face
    self.transform_eye = transform_eye
    self.image_root = image_root
    self.orig_list_len = 0
    self.angle = angle
    if train==False:
      angle = 90
    self.binwidth = binwidth
    self.lines = []

    if isinstance(label_path, list):
      for i in label_path:
        with open(i) as f:
          print("here")
          line = f.readlines
          line.pop(0)
          self.lines.extend(line)

    else:
      with open(label_path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len = len(lines)
        for line in lines:
          gaze2d = line.strip().split(" ")[5]
          label = np.array(gaze2d.split(",")).astype("float")
          if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
            self.lines.append(line)

    print(len(self.lines))
    print("{} items removed from Gaze360 dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    face = line[0]
    lefteye = line[1]
    righteye = line[2]
    name = line[3]
    gaze2d  = line[5]
    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    pitch = label[0] * 180 / np.pi
    yaw = label[1] * 180 / np.pi

    face = Image.open(os.path.join(self.image_root, face))
    left = Image.open(os.path.join(self.image_root, lefteye))
    right = Image.open(os.path.join(self.image_root, righteye))

    if self.transform_face is not None:
      face = self.transform_face(face)

    if self.transform_eye is not None:
      left = self.transform_eye(left)
      right = self.transform_eye(right)

    bins = np.array(range(-1 * self.angle, self.angle, self.binwidth))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FlaotTensor([pitch, yaw])

    return face, left, right, labels, cont_labels, name
