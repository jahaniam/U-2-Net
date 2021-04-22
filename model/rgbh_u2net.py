import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from model.u2net import RSU4,RSU4F,RSU5,RSU6,RSU7,_upsample_like


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
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

	return loss0, loss


class DepthU2NETDecoder(nn.Module):

    def __init__(self,in_ch=1,out_ch=1):
        super(DepthU2NETDecoder,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        #stage 1
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)

        return hx1,hx2,hx3,hx4,hx5,hx6


##### U^2-Net ####
class RGBH_U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(RGBH_U2NET,self).__init__()

        self.depth_dec = DepthU2NETDecoder()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.fuse1 = nn.Conv2d(128,64,3,padding=1)
        self.fuse2 = nn.Conv2d(256,128,3,padding=1)
        self.fuse3 = nn.Conv2d(512,256,3,padding=1)
        self.fuse4 = nn.Conv2d(1024,512,3,padding=1)
        self.fuse5 = nn.Conv2d(1024,512,3,padding=1)
        self.fuse6 = nn.Conv2d(1024,512,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)
    
    def forward(self,x):

        rgb  = x[:,:3,:,:]
        depth = x[:,3:,:,:]
        depth_stage1, depth_stage2, depth_stage3, depth_stage4, depth_stage5, depth_stage6 = self.depth_dec(depth)
        #stage 1
        hx1 = self.stage1(rgb)
        hx1 = torch.cat((depth_stage1,hx1),1)
        hx1 = self.fuse1(hx1)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx2 = torch.cat((depth_stage2,hx2),1)
        hx2 = self.fuse2(hx2)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx3 = torch.cat((depth_stage3,hx3),1)
        hx3 = self.fuse3(hx3)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx4 = torch.cat((depth_stage4,hx4),1)
        hx4 = self.fuse4(hx4)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5 = torch.cat((depth_stage5,hx5),1)
        hx5 = self.fuse5(hx5)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6 = torch.cat((depth_stage6,hx6),1)
        hx6 = self.fuse6(hx6)
        
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        
    def get_input(self, sample_batched):
        rgb,dep = sample_batched['img'].cuda(),sample_batched['depth'].cuda()
        return rgb,dep
    
    def get_gt(self, sample_batched):
        gt = sample_batched['gt'].cuda()
        return gt

    def get_result(self, output, index=0):
        if isinstance(output, list):
            result = output[0].data.cpu().numpy()[index,0,:,:]
        else:
            result = output.data.cpu().numpy()[index,0,:,:]

        # if isinstance(output, list):
        #     result = torch.sigmoid(output[0].data.cpu()).numpy()[index,0,:,:]
        # else:
        #     result = torch.sigmoid(output.data.cpu()).numpy()[index,0,:,:]
        return result
    
    def get_train_params(self, lr):
        train_params = [{'params': self.parameters(), 'lr': lr}]
        return train_params