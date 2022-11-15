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
from model_vgg import VGGGaze
from utils import gazeto3d, select_device, angular

from itertools import islice

def parse_args():
  parser = argparse.ArgumentParser(description='Gaze estimation using the Gazenet based CNN network.')
  parser.add_argument(
    '--gpu', dest='gpu_id', help='GPU device id to use [0]', default='0', type=str
  )
  parser.add_argument(
    '--arch', dest='arch', help='GC use the backbone network.', default='VGG19_bn', type=str
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


def load_filtered_state_dict(model, snapshot):
  #By user apaszke from discuss.pytorch.org
  model_dict = model.state_dict()
  #model_dict = {k: v for k, v in model_dict.items() if 'classifier' not in k.lower()}
  snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
  model_dict.update(snapshot)
  model.load_state_dict(model_dict)

cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def getArch_weights(arch, bins):
  if arch == 'VGG11':
    model = VGGGaze(cfg["A"], num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg11-8a719046.pth'
  elif arch == 'VGG11_bn':
    model = VGGGaze(cfg["A"], batch_norm=True, num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
  elif arch == 'VGG13':
    model = VGGGaze(cfg["B"], num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg13-19584684.pth'
  elif arch == 'VGG13_bn':
    model = VGGGaze(cfg["B"], batch_norm=True, num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'
  elif arch == 'VGG16':
    model = VGGGaze(cfg["D"], num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
  elif arch == 'VGG16_bn':
    model = VGGGaze(cfg["D"], batch_norm=True, num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
  elif arch == 'VGG19':
    model = VGGGaze(cfg["E"], num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
  else:
    model = VGGGaze(cfg["E"], batch_norm=True, num_bins=bins)
    pre_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'

  return model, pre_url

def nth_keys(dict,n):
  it = iter(dict)
  next(islice(it, n, n), None)

  return next(it)


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
    summary(model, (3, 224, 224))
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

    #validation dataloader
    val_label = os.path.join(label_path, "val.label")
    val_dataset = datasets.Gaze360(val_label, args.image_dir, transformations, 180, 2, train=False)
    val_loader = DataLoader(
      dataset=val_dataset,
      batch_size=int(batch_size),
      shuffle=False,
      num_workers=8,
      pin_memory=True
    )

    torch.backends.cudnn.benchmark = True

    today = datetime.datetime.fromtimestamp(time.time())
    summary_name = '{}_{}'.format('VGG-gaze360', str(today.strftime('%Y-%-m*%-d_%-H*%-M*%-S')))

    output = os.path.join(output, summary_name)
    if not os.path.exists(output):
      os.makedirs(output)

    valpath = os.path.join(valpath, summary_name)
    if not os.path.exists(valpath):
      os.makedirs(valpath)


    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    softmax = nn.Softmax(dim=1).cuda(gpu)

    model_param = model.state_dict()
    model_features = {k: v for k, v in model_param.items() if 'features' in k.lower()}

    #param
    param_to_update_1 = []
    param_to_update_2 = []
    param_to_update_3 = []
    param_to_update_4 = []

    if args.arch == 'VGG16':
      update_param_names_1 = []
      update_param_names_2 = []
      for i in range(4):
        update_param_names_1.append(nth_keys(model_features, i))
      for j in range(4, 26):
        update_param_names_2.append(nth_keys(model_features, j))
    elif args.arch == 'VGG19':
      update_param_names_1 = []
      update_param_names_2 = []
      for i in range(4):
        update_param_names_1.append(nth_keys(model_features, i))
      for j in range(4, 32):
        update_param_names_2.append(nth_keys(model_features, j))
    else:
      print("choose other architecture")

    update_param_names_3 = ['p_classifier.0.weight', 'p_classifier.0.bias', 'p_classifier.3.weight', 'p_classifier.3.bias', 'y_classifier.0.weight', 'y_classifier.0.bias', 'y_classifier.3.weight', 'y_classifier.3.bias']
    update_param_names_4 = ['p_classifier.6.weight', 'p_classifier.6.bias', 'y_classifier.6.weight', 'y_classifier.6.bias']

    for name, param in model.named_parameters():
      if name in update_param_names_1:
        param.requires_grad = False
        param_to_update_1.append(param)
      elif name in update_param_names_2:
        param.requires_grad = True
        param_to_update_2.append(param)
      elif name in update_param_names_3:
        param.requires_grad = True
        param_to_update_3.append(param)
      elif name in update_param_names_4:
        param.requires_grad = True
        param_to_update_4.append(param)

    #Adam
    optimizer_gaze = torch.optim.Adam([
      {'params' : param_to_update_1, 'lr' : 0},
      {'params' : param_to_update_2, 'lr' : args.lr},
      {'params' : param_to_update_3, 'lr' : args.lr},
      {'params' : param_to_update_4, 'lr' : args.lr*10}
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

    with open(os.path.join(valpath, dataset+".log"), 'w') as outfile:
      outfile.write(configuration)
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

        #validation
        total = 0
        avg_error = 0.0
        model.eval()
        with torch.no_grad():
          for j, (images, labels, cont_labels, name) in enumerate(val_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)

            label_pitch = cont_labels[:, 0].float() * np.pi / 180
            label_yaw = cont_labels[:, 1].float() * np.pi / 180

            gaze_pitch, gaze_yaw = model(images)

            _, pitch_binpred = torch.max(gaze_pitch.data, 1)
            _, yaw_binpred = torch.max(gaze_yaw.data, 1)

            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 2 - 180
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 2 - 180

            pitch_predicted = pitch_predicted * np.pi / 180
            yaw_predicted = yaw_predicted * np.pi / 180

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
