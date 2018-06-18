from pretrainedmodels import models
import torch
from torch.nn import Sequential,MaxPool2d,Conv2d,LeakyReLU,BatchNorm2d,Upsample,Module,ModuleList

class res50_pyramid(Module):
    def __init__(self):
        super(res50_pyramid, self).__init__()
        self.model = models.resnet50()
        del self.model.avgpool
        del self.model.fc
        del self.model.last_linear

    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        p3 = x
        x = self.model.layer3(x)
        p2 = x
        p1 = self.model.layer4(x)
        return p1,p2,p3

def Basic_block(c_in,c_out):
    return Sequential(
        Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
        LeakyReLU(0.1))

def Squeeze_block(c_in,c_out):
    return Sequential(
            Conv2d(c_in,c_out, kernel_size=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
            LeakyReLU(0.1))

def make_pyramids_classifier(c_in,c_out,num_classes):
    classifier_num = 3*(num_classes+5)
    Pyramid_net = Sequential(
        Squeeze_block(c_in,c_out),
        Basic_block(c_out,c_out*2),
        Squeeze_block(c_out*2,c_out),
        Basic_block(c_out,c_out*2),
        Squeeze_block(c_out*2,c_out)
        )
    Classifier_net = Sequential(
        Basic_block(c_out,c_out*2),
        Squeeze_block(c_out*2,classifier_num)
        )
    return Pyramid_net,Classifier_net

class Yolo_body(Module):
    def __init__(self,num_class):
        super(Yolo_body, self).__init__()
        self.num_class = num_class
        self.res50_pyramid = res50_pyramid()
        self.pyramid1,self.classifier1 = make_pyramids_classifier(2048,1024,self.num_class)
        self.squeeze1 = Squeeze_block(1024,512)
        self.pyramid2,self.classifier2 = make_pyramids_classifier(1536,512,self.num_class)
        self.squeeze2 = Squeeze_block(512,256)
        self.pyramid3,self.classifier3 = make_pyramids_classifier(768,256,self.num_class)
        self.upsample = Upsample(scale_factor=2, mode='bilinear',align_corners=False)
    
    def forward(self,x):
        p1,p2,p3 = self.res50_pyramid(x)
        x = self.pyramid1(p1)
        y1 = self.classifier1(x)
        x = self.squeeze1(x)
        x = self.upsample(x)
        x = torch.cat((x,p2),dim=1)
        x = self.pyramid2(x)
        y2 = self.classifier2(x)
        x = self.squeeze2(x)
        x = self.upsample(x)
        x = torch.cat((x,p3),dim=1)
        x = self.pyramid3(x)
        y3 = self.classifier3(x)
        return y1,y2,y3    