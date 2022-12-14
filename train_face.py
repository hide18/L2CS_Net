from cProfile import label
from cgi import test
import os, argparse, time, datetime
from random import shuffle

import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchsummary import summary

import datasets
from model_new import Face
from utils import gazeto3d, select_device, angular

def parse_args():
  parser = argparse.ArgumentParser(description='Gaze estimation using the Gazenet based CNN network.')
  parser.add_argument(
    '--gpu', dest='gpu_id', help='GPU device id to use [0]', default='0', type=str
  )
  parser.add_argument(
    '--arch', dest='arch', help='GC use the backbone network.', default='ResNet50', type=str
  )
  parser.add_argument(
    '--num_epochs', dest='num_epochs', help='Maximun number of training epochs.', default=50, type=int
  )
  parser.add_argument(
    '--batch_size', dest='batch_size', help='Batch size.', default=16, type=int
  )
  parser.add_argument(
    '--lr', dest='lr', help='Base learning rate.', default=0.00001, type=float
  )
  parser.add_argument(
    '--alpha', dest='alpha', help='Regression loss coefficient.', default=1, type=float
  )
  parser.add_argument(
    '--dataset', dest='dataset', help='Use dataset', default="gaze360", type=str
  )
  parser.add_argument(
    '--image_dir', dest='image_dir', help='Directory path for gaze360 images.', default='datasets/Gaze360/Image', type=str
  )
  parser.add_argument(
    '--label_dir', dest='label_dir', help='Directory path for gaze360 labels.', default='datasets/Gaze360/Label', type=str
  )
  parser.add_argument(
    '--snapshot', dest='snapshot', help='Path of pretrained models.', default='', type=str
  )
  parser.add_argument(
    '--output', dest='output', help='Path of output models.', default='output/snapshots/', type=str
  )

  args = parser.parse_args()
  return args


def get_ignored_params(model):
  #Generator function that yields ignored params.
  b = [model.conv1, model.bn1]
  for i in range(len(b)):
    for module_name, module in b[i].named_modules():
      if 'bn' in module_name:
        module.eval()
      for name, param in module.named_parameters():
        yield param

def get_non_ignored_params(model):
  #Ganerator function that yields params that will be optimized.
  b = [model.layer1, model.layer2, model.layer3, model.layer4]
  for i in range(len(b)):
    for module_name, module in b[i].named_modules():
      if 'bn' in module_name:
        module.eval()
      for name, param in module.named_parameters():
        yield param

def get_fc_params(model):
  #Generator function that yields fc layer params.
  b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
  for i in range(len(b)):
    for module_name, module in b[i].named_modules():
      if 'bn' in module_name:
        module.eval()
      for name, param in module.named_parameters():
        yield param


def load_filtered_state_dict(model, snapshot):
  #By user apaszke from discuss.pytorch.org
  model_dict = model.state_dict()
  snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
  model_dict.update(snapshot)
  model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
  if arch == 'ResNet18':
    model = Face(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
  elif arch == 'ResNet34':
    model = Face(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
  elif arch == 'ResNet101':
    model = Face(torchvision.models.resnet.Botteleneck, [3, 4, 23, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
  elif arch == 'ResNet152':
    model = Face(torchvision.models.resnet.Botteleneck, [3, 8, 36, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
  else:
    model = Face(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

  return model, pre_url




if __name__=='__main__':
  args = parse_args()

  cudnn.enabled = True
  num_epochs = args.num_epochs
  batch_size = args.batch_size
  gpu = select_device(args.gpu_id, batch_size=args.batch_size)
  dataset = args.dataset
  alpha = args.alpha
  output = args.output

  transformations = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])


  if dataset=="gaze360":
    model, pre_url = getArch_weights(args.arch, 180)
    if args.snapshot == '':
      load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    else:
      saved_state_dict = torch.load(args.snapshot)
      model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    summary(model, (3, 448, 448))
    print('Loading data.')

    label_path = args.label_dir

    #traindata dataloader
    train_label = os.path.join(label_path, "train.label")
    train_dataset = datasets.Gaze360(train_label, args.image_dir, transformations, 180, 2)
    train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=int(batch_size),
      shuffle=True,
      num_workers=8,
      pin_memory=True
    )

    torch.backends.cudnn.benchmark = True

    today = datetime.datetime.fromtimestamp(time.time())
    summary_name = '{}_{}'.format('Face-gaze360', str(today.strftime('%Y-%-m*%-d_%-H*%-M*%-S')))

    output = os.path.join(output, summary_name)
    if not os.path.exists(output):
      os.makedirs(output)

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    l1_loss = nn.L1Loss().cuda(gpu)
    huber_loss = nn.HuberLoss().cuda(gpu)
    softmax = nn.Softmax(dim=1).cuda(gpu)

    #Adam
    optimizer_gaze = torch.optim.Adam([
      {'params' : get_ignored_params(model), 'lr' : 0},
      {'params' : get_non_ignored_params(model), 'lr' : args.lr},
      {'params' : get_fc_params(model), 'lr' : args.lr}
    ], lr = args.lr)

    #SGD
    '''
    optimizer_gaze = torch.optim.SGD(
      [{'params' : get_ignored_params(model), 'lr' : 0},
      {'params' : get_non_ignored_params(model), 'lr' : args.lr},
      {'params' : get_fc_params(model), 'lr' : args.lr}],
      lr = args.lr, momentum = 0.9, weight_decay=0.0001
    )
    '''

    #RAdam
    '''
    optimizer_gaze = torch.optim.RAdam([
      {'params' : get_ignored_params(model), 'lr' : 0},
      {'params' : get_non_ignored_params(model), 'lr' : args.lr},
      {'params' : get_fc_params(model), 'lr' : args.lr}
    ], lr = args.lr)
    '''

    idx_tensor = [idx for idx in range(180)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)


    print('Ready to train and validation network.')
    configuration = f"\ntrain_validation configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n"

    epoch_list = []
    avg_MAE = []


    for epoch in range(num_epochs):
      sum_loss_pitch = sum_loss_yaw = iter_gaze = 0
      #train
      model.train()
      for i, (images_gaze, labels_gaze, cont_labels_gaze, name) in enumerate(train_loader):
        images_gaze = Variable(images_gaze).cuda(gpu)

        #Binned labels (Tensor shape)
        label_pitch = Variable(labels_gaze[:, 0]).cuda(gpu)
        label_yaw = Variable(labels_gaze[:, 1]).cuda(gpu)

        #Continuous labels
        label_pitch_cont = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
        label_yaw_cont = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

        #Calculate gaze angular
        pitch, yaw = model(images_gaze)

        #Cross Entropy Loss
        loss_pitch = criterion(pitch, label_pitch)
        loss_yaw = criterion(yaw, label_yaw)

        #Predict gaze angular
        pitch_predicted = softmax(pitch)
        yaw_predicted = softmax(yaw)
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 2 - 180
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 2 - 180

        #MSE Loss
        loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
        loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)

        #Total Loss
        loss_pitch += alpha * loss_reg_pitch
        loss_yaw += alpha * loss_reg_yaw
        sum_loss_pitch += loss_pitch
        sum_loss_yaw += loss_yaw


        loss_seq = [loss_pitch, loss_yaw]
        grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
        optimizer_gaze.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer_gaze.step()

        iter_gaze += 1

        if (i+1) % 100 == 0:
          print('Epoch [%d/%d], Iter [%d/%d], Losses : Gaze Pitch %.4f, Gaze Yaw %.4f' %
          (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, sum_loss_pitch/iter_gaze, sum_loss_yaw/iter_gaze)
          )

      if epoch % 1 == 0 and epoch < num_epochs:
        if torch.save(model.state_dict(), output +'/'+'_epoch_'+str(epoch+1)+'.pkl') == None:
          print('Taking snapshot... success')
