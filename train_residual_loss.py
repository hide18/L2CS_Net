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

import datasets_plus
from model_residual import GResidual
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
  parser.add_argument(
    '--valpath', dest='valpath', help='Path of validation results.', default='validation/gaze360/', type=str
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
  b = [model.face_pitch, model.face_yaw, model.eye_pitch, model.eye_yaw]
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
    model = GResidual(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
  elif arch == 'ResNet34':
    model = GResidual(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
  elif arch == 'ResNet101':
    model = GResidual(torchvision.models.resnet.Botteleneck, [3, 4, 23, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
  elif arch == 'ResNet152':
    model = GResidual(torchvision.models.resnet.Botteleneck, [3, 8, 36, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
  else:
    model = GResidual(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, bins)
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
  valpath = args.valpath
  output = args.output

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


  if dataset=="gaze360":
    model, pre_url = getArch_weights(args.arch, 180)
    if args.snapshot == '':
      face = model.face_res
      eye = model.eye_res
      load_filtered_state_dict(face, model_zoo.load_url(pre_url))
      load_filtered_state_dict(eye, model_zoo.load_url(pre_url))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    print('Loading data.')

    label_path = args.label_dir

    #traindata dataloader
    train_label = os.path.join(label_path, "train.label")
    train_dataset = datasets_plus.Gaze360(train_label, args.image_dir, transformation_face, transformation_eye, 180, 2)
    train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=int(batch_size),
      shuffle=True,
      num_workers=8,
      pin_memory=True
    )

    #validation dataloader
    val_label = os.path.join(label_path, "val.label")
    val_dataset = datasets_plus.Gaze360(val_label, args.image_dir, transformation_face, transformation_eye, 180, 2, train=False)
    val_loader = DataLoader(
      dataset=val_dataset,
      batch_size=int(batch_size),
      shuffle=False,
      num_workers=8,
      pin_memory=True
    )

    torch.backends.cudnn.benchmark = True

    today = datetime.datetime.fromtimestamp(time.time())
    summary_name = '{}_{}'.format('GResidual-gaze360', str(today.strftime('%Y-%-m*%-d_%-H*%-M*%-S')))

    output = os.path.join(output, summary_name)
    if not os.path.exists(output):
      os.makedirs(output)

    valpath = os.path.join(valpath, summary_name)
    if not os.path.exists(valpath):
      os.makedirs(valpath)


    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    l1_loss = nn.L1Loss().cuda(gpu)
    softmax = nn.Softmax(dim=1).cuda(gpu)

    optimizer_gaze = torch.optim.Adam([
      {'params' : get_ignored_params(model.face_res), 'lr' : 0},
      {'params' : get_ignored_params(model.eye_res), 'lr' : 0},
      {'params' : get_non_ignored_params(model.face_res), 'lr' : args.lr},
      {'params' : get_non_ignored_params(model.eye_res), 'lr' : args.lr},
      {'params' : get_fc_params(model), 'lr' : args.lr}
    ], lr = args.lr)


    idx_tensor = [idx for idx in range(180)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)


    print('Ready to train and validation network.')
    configuration = f"\ntrain_validation configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n"

    epoch_list = []
    avg_MAE = []

    with open(os.path.join(valpath, dataset+".log"), 'w') as outfile:
      outfile.write(configuration)
      for epoch in range(num_epochs):
        sum_loss_pitch = sum_loss_yaw = iter_gaze = 0

        #train
        model.train()
        for i, (face, left, right, labels, cont_labels, name) in enumerate(train_loader):
          #input image
          face = Variable(face).cuda(gpu)
          left = Variable(left).cuda(gpu)
          right = Variable(right).cuda(gpu)

          #Binned labels (Tensor shape)
          label_pitch = Variable(labels[:, 0]).cuda(gpu)
          label_yaw = Variable(labels[:, 1]).cuda(gpu)

          #Continuous labels
          label_pitch_cont = Variable(cont_labels[:, 0]).cuda(gpu)
          label_yaw_cont = Variable(cont_labels[:, 1]).cuda(gpu)

          #Calculate gaze angular
          face_picth, face_yaw, eyes_pitch, eyeys_yaw = model(face, left, right)
          pitch, yaw = face_picth+eyes_pitch, face_yaw+eyeys_yaw

          #Cross Entropy Loss
          cross_gb_pitch = criterion(face_picth, label_pitch)
          cross_gb_yaw = criterion(face_yaw, label_yaw)
          cross_g_pitch = criterion(pitch, label_pitch)
          cross_g_yaw = criterion(yaw, label_yaw)
          cross_pitch = cross_gb_pitch + alpha*cross_g_pitch
          cross_yaw = cross_gb_yaw + alpha*cross_g_yaw

          #Predict gaze angular
          pre_gb_pitch = softmax(face_picth)
          pre_gb_yaw = softmax(face_yaw)
          pre_gr_pitch = softmax(eyes_pitch)
          pre_gr_yaw = softmax(eyeys_yaw)

          pre_gb_pitch = torch.sum(pre_gb_pitch * idx_tensor, 1) * 2 - 180
          pre_gb_yaw = torch.sum(pre_gb_yaw * idx_tensor, 1) * 2 - 180
          pre_gr_pitch = torch.sum(pre_gr_pitch * idx_tensor, 1) * 2 - 180
          pre_gr_yaw = torch.sum(pre_gr_yaw * idx_tensor, 1) * 2 - 180

          #Gaze
          pre_pitch = pre_gb_pitch + pre_gr_pitch
          pre_yaw = pre_gb_yaw + pre_gr_yaw

          #MSE
          loss_gb_pitch = reg_criterion(pre_gb_pitch, label_pitch_cont)
          loss_gb_yaw = reg_criterion(pre_gb_yaw, label_yaw_cont)
          loss_g_pitch = reg_criterion(pre_pitch, label_pitch_cont)
          loss_g_yaw = reg_criterion(pre_yaw, label_yaw_cont)

          #Total Loss
          loss_pitch = loss_gb_pitch + alpha*loss_g_pitch
          loss_yaw = loss_gb_yaw + alpha*loss_g_yaw

          loss_pitch += cross_pitch
          loss_yaw += cross_yaw

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

        #validation
        total = 0
        avg_error = 0.0
        model.eval()
        with torch.no_grad():
          for j, (face, left, right, labels, cont_labels, name) in enumerate(val_loader):
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

        x = epoch + 1
        epoch_list.append(x)
        avg_MAE.append(avg_error/total)
        loger = f"---VAL--- Epoch [{x}/{num_epochs}], MAE : {avg_error/total}\n"
        print(loger)
        outfile.write(loger)

        if epoch % 1 == 0 and epoch < num_epochs:
          if torch.save(model.state_dict(), output +'/'+'_epoch_'+str(epoch+1)+'.pkl') == None:
            print('Taking snapshot... success')
