import os, argparse, time, datetime, pathlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

import datasets_plus
from utils import select_device, natural_keys, gazeto3d, angular
from model_residualback2 import GResidual

def parse_args():
  parser = argparse.ArgumentParser(
    description='Gaze Estimation using mymodel.'
  )
  parser.add_argument(
    '--image_dir', dest='image_dir', help='Directory path for images.', default='datasets/Gaze360/Image', type=str
  )
  parser.add_argument(
    '--label_dir', dest='label_dir', help='Directory path for labels.', default='datasets/Gaze360/Label/test.label', type=str
  )
  parser.add_argument(
    '--dataset', dest='dataset', help='gaze360', default='gaze360', type=str
  )
  parser.add_argument(
    '--snapshot', dest='snapshot', help='Path to the folder contains models.', default='output/snapshots', type=str
  )
  parser.add_argument(
    '--evalpath', dest='evalpath', help='Path for the output evaluating gaze models.', default='evaluation/gaze360'
  )
  parser.add_argument(
    '--gpu', dest='gpu_id', help='GPU device id to use [0]', default='0', type=str
  )
  parser.add_argument(
    '--batch_size', dest='batch_size', help='Batch size.', default=100, type=int
  )
  parser.add_argument(
    '--arch', dest='arch', help='Network architecture using backbone.', default='ResNet50', type=str
  )
  parser.add_argument(
    '--bins', default='180', type=int
  )
  parser.add_argument(
    '--angle', default=180, type=int
  )
  args = parser.parse_args()
  return args

def getArch(arch, bins):
  if arch == 'ResNet18':
    model = GResidual(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 3, bins)
  elif arch == 'ResNet34':
    model = GResidual(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], 3, bins)
  elif arch == 'ResNet101':
    model = GResidual(torchvision.models.resnet.Botteleneck, [3, 4, 23, 3], 3, bins)
  elif arch == 'ResNet152':
    model = GResidual(torchvision.models.resnet.Botteleneck, [3, 8, 36, 3], 3, bins)
  else:
    model = GResidual(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, bins)

  return model

if __name__ == '__main__':
  args = parse_args()
  cudnn.enabled = True
  gpu = select_device(args.gpu_id, batch_size=args.batch_size)
  batch_size = args.batch_size
  arch = args.arch
  dataset = args.dataset
  evalpath = args.evalpath
  snapshot = args.snapshot
  bins = args.bins
  angle = args.angle

  transformation_face = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  transformation_eye = transforms.Compose([
    transforms.Resize((108, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  if dataset=='gaze360':
    gaze_dataset = datasets_plus.Gaze360(args.label_dir, args.image_dir, transformation_face, transformation_eye, 180, 2, train=False)
    test_loader = torch.utils.data.DataLoader(
      dataset=gaze_dataset,
      batch_size=int(batch_size),
      shuffle=False,
      num_workers=8,
      pin_memory=True
    )

    model_name = pathlib.Path(snapshot).stem
    evalpath = os.path.join(evalpath, model_name)
    if not os.path.exists(evalpath):
      os.makedirs(evalpath)

    folder = os.listdir(snapshot)
    folder.sort(key=natural_keys)
    softmax = nn.Softmax(dim=1)
    with open(os.path.join(evalpath, dataset+".log"), 'w') as outfile:
      configuration = f"\ntest config = gpu={gpu}, batch_size={batch_size}, model_arch={arch}\nStart testing model={model_name}\n"
      print(configuration)
      outfile.write(configuration)
      epoch_list = []
      avg_picth = []
      avg_yaw = []
      avg_MAE = []

      for epochs in folder:
        model = getArch(arch, 180)
        saved_state_dict = torch.load(os.path.join(snapshot, epochs))
        model.load_state_dict(saved_state_dict)
        model.cuda(gpu)
        model.eval()
        total = 0
        idx_tensor = [idx for idx in range(180)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
        avg_error = 0.0

        with torch.no_grad():
          for j, (face, left, right, lebls, cont_labels, name) in enumerate(test_loader):
            face = Variable(face).cuda(gpu)
            left = Variable(left).cuda(gpu)
            right = Variable(right).cuda(gpu)
            total += cont_labels.size(0)

            label_pitch = cont_labels[:, 0].float() * np.pi / 180
            label_yaw = cont_labels[:, 1].float() * np.pi / 180

            gb_pitch, gb_yaw, gr_pitch, gr_yaw = model(face, left, right)

            pre_gb_pitch = softmax(gb_pitch)
            pre_gb_yaw = softmax(gb_yaw)
            pre_gr_pitch = softmax(gr_pitch)
            pre_gr_yaw = softmax(gr_yaw)

            pre_gb_pitch = torch.sum(pre_gb_pitch * idx_tensor, 1).cpu() * 2 - 180
            pre_gb_yaw = torch.sum(pre_gb_yaw * idx_tensor, 1).cpu() * 2 - 180
            pre_gr_pitch = torch.sum(pre_gr_pitch * idx_tensor, 1).cpu() * 2 - 180
            pre_gr_yaw = torch.sum(pre_gr_yaw * idx_tensor, 1).cpu() * 2 - 180

            pre_gb_pitch = pre_gb_pitch * np.pi / 180
            pre_gb_yaw = pre_gb_yaw * np.pi / 180
            pre_gr_pitch = pre_gr_pitch * np.pi / 180
            pre_gr_yaw = pre_gr_yaw * np.pi / 180

            pitch_predicted = pre_gb_pitch + pre_gr_pitch
            yaw_predicted = pre_gb_yaw + pre_gr_yaw

            for p, y, pl, yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
              avg_error += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))

        x = ''.join(filter(lambda i: i.isdigit(), epochs))
        epoch_list.append(x)
        avg_MAE.append(avg_error/total)
        loger = f"[{epochs}---{args.dataset}] Total Num:{total}, MAE{avg_error/total}\n"
        outfile.write(loger)
        print(loger)

    fig = plt.figure(figsize=(14, 8))
    plt.xlabel('epoch')
    plt.ylabel('avg')
    plt.title('Gaze anguler error')
    plt.legend()
    plt.plot(epoch_list, avg_MAE, color='k', label='mae')
    fig.savefig(os.path.join(evalpath, dataset+".png"), format='png')
    plt.show()
