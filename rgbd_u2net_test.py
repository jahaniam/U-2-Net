import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from rgbd_data_loader import RGBD_RescaleT
from rgbd_data_loader import RGBD_RandomCrop
from rgbd_data_loader import RGBD_RandomFlipT
from rgbd_data_loader import RGBD_ToTensorLab
from rgbd_data_loader import RGBD_SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,i_test):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    # predict_np[predict_np>0.10]=1.0
    im = Image.fromarray(predict_np*255).convert('RGB')
    
    confidence_threshold = 0.80

    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = np.array(im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR),dtype=np.uint8)
    ret, imo = cv2.threshold(imo,confidence_threshold*255, 255, cv2.THRESH_BINARY)



    imo[np.all(imo == (255, 255, 255), axis=-1)] = (0, 255, 0)
    alpha = 0.7
    beta = 1 - alpha
    image = np.array(image)
    image = image[...,::-1]

    res = cv2.addWeighted(image, alpha, imo, beta, 0)

    # Path(str(image_name).replace('rgb','auto_annotation_rgbd')).parents[0].mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(image_name).replace('rgb','auto_annotation_rgbd'),imo)
    Path(str(image_name).replace('rgb','auto_annotation_rgbd_overlay')).parents[0].mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_name).replace('rgb','auto_annotation_rgbd_overlay'),res)
    # image.save()

def main():

    # --------- 1. get image path and name ---------
    model_name='rgbd_u2net'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '_s6457_2021-02-18_12_36_53u2net_6457_bce_itr_38000_train_0.107382_tar_0.009159.pth')
    model_dir = 'saved_models/rgbd_u2net/rgbd_u2net_s30857_2021-04-06_03_52_47/rgbd_u2net_30857_bce_itr_220000_train_0.089360_tar_0.008674.pth'
    # path_files = Path('/pool/2021-03-31_22-11-41')
    path_files = Path('/pool/2021-03-31_22-32-41')
    img_name_list1 = sorted([str(x) for x in path_files.rglob('**/rgb/*.png')])
    img_name_list1 = [str(x) for x in img_name_list1 if '2021-03-31' in str(x) or '2021-04-01' in str(x)]

    # path_files2 = Path('/dataset')
    # img_name_list2 = sorted([str(x) for x in path_files2.rglob('**/rgb/*.png') if not Path(str(x).replace('rgb','annotation')).exists()])
    # img_name_list2 = [str(x) for x in img_name_list2 if '2021-03-10' in str(x)]
    
    img_name_list = img_name_list1 #+ img_name_list2
    print('img len',len(img_name_list))
    depth_name_list = [x.replace('rgb','aligned_depth') for x in img_name_list]
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = RGBD_SalObjDataset(img_name_list = img_name_list,
                                        depth_name_list = depth_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RGBD_RescaleT(320),
                                                                      RGBD_ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='rgbd_u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(4,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(tqdm(test_salobj_dataloader)):

        # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        depth = data_test['depth']
        inputs_test = torch.cat((inputs_test,torch.unsqueeze(depth,dim=1)),dim=1) # H x W x 4

        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        with torch.no_grad():
            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir,i_test)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
