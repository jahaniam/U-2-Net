import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from rgbd_data_loader import RGBD_RescaleT
from rgbd_data_loader import RGBD_RandomCrop
from rgbd_data_loader import RGBD_RandomFlipT
from rgbd_data_loader import RGBD_ToTensorLab
from rgbd_data_loader import RGBD_SalObjDataset

from model import U2NET
from model import U2NETP
from model.rgbh_u2net import RGBH_U2NET
import os

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))


	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'rgbh_u2net' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data2' + os.sep)
tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'im_aug' + os.sep)
tra_label_dir = os.path.join('DUTS', 'DUTS-TR',  'gt_aug' + os.sep)

image_ext = '.png'
label_ext = '.png'



epoch_num = 10000
batch_size_train = 6
batch_size_val = 1
train_num = 0
val_num = 0

from pathlib import Path
dataset_path = Path('/dataset')
tra_lbl_name_list = list(dataset_path.glob('**/annotation/*.png'))
tra_img_name_list = [str(path).replace('annotation','rgb') for path in tra_lbl_name_list]
tra_depth_name_list = [str(path).replace('annotation','height') for path in tra_lbl_name_list]


print("---")
print("train images: ", len(tra_img_name_list))
print("train depths: ", len(tra_depth_name_list))
print("train labels: ", len(tra_lbl_name_list))

print("---")

train_num = len(tra_img_name_list)

from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("rgbh_u2net_s"+str(train_num)+"_%Y-%m-%d_%H_%M_%S")
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep, timestampStr)
Path(model_dir).mkdir(exist_ok=True,parents=True)

np.savetxt(model_dir + '/train_filenames.txt', tra_img_name_list, delimiter="\n", fmt="%s")


salobj_dataset = RGBD_SalObjDataset(
    img_name_list=tra_img_name_list,
    depth_name_list=tra_depth_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([RGBD_RandomFlipT(),
        RGBD_RescaleT(320),
        RGBD_RandomCrop(288),
        RGBD_ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=3)

# ------- 3. define model --------
# define the net
if(model_name=='rgbh_u2net'):
    net = RGBH_U2NET()
elif(model_name=='u2netp'):
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.cuda()

# checkpoint = torch.load('saved_models/rgbh_u2net/rgbd_u2net_s31057_2021-04-09_02_19_05/rgbh_u2net_31057_bce_itr_72000_train_0.178833_tar_0.020285.pth')
# net.load_state_dict(checkpoint)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000 # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, depth, labels = data['image'], data['depth'], data['label']
        inputs = torch.cat((inputs,torch.unsqueeze(depth,dim=1)),dim=1) # H x W x 4

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), model_dir +'/' +model_name+"_"+str(train_num)+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
