import torch
from torch.nn import Sequential,MaxPool2d,Conv2d,LeakyReLU,BatchNorm2d,Upsample,Module,ModuleList

def Basic_block(c_in,c_out):
    return Sequential(
        Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
        LeakyReLU(0.1))

def DownSample_block(c_in,c_out):
    return Sequential(
        Conv2d(c_in,c_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
        LeakyReLU(0.1))

def Squeeze_block(c_in,c_out):
    return Sequential(
            Conv2d(c_in,c_out, kernel_size=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True),
            LeakyReLU(0.1))

class Residual_forward(Module):
    def __init__(self,c_in,c_out):
        super(Residual_forward, self).__init__()
        self.basic = Basic_block(c_in,c_out)
        self.squeeze = Squeeze_block(c_out,c_in)
        
    def forward(self,x):
        x = self.basic(x)
        x = self.squeeze(x)
        return x        

def make_block(c_in,c_out,num_blocks):
    forward_blocks = ModuleList()
    for i in range(num_blocks):
        forward_blocks.append(Residual_forward(c_in,c_out))
    return forward_blocks

class Residual_block(Module):
    def __init__(self,c_in,c_out,num_blocks):
        super(Residual_block, self).__init__()
        self.num_blocks = num_blocks
        self.down_sample = DownSample_block(c_in,c_out)
        self.forward_blocks = make_block(c_out,c_in,num_blocks)     
        
    def forward(self,x):
        x = self.down_sample(x)
        for i in range(self.num_blocks):
            res = x
            x = self.forward_blocks[i](x)
            x = res + x
        return x

class Darknet_body(Module):
    def __init__(self):
        super(Darknet_body, self).__init__()
        self.basic_block = Basic_block(3,32)
        self.res_block1 = Residual_block(32,64,1)
        self.res_block2 = Residual_block(64,128,2)
        self.res_block3 = Residual_block(128,256,8)
        self.res_block4 = Residual_block(256,512,8)
        self.res_block5 = Residual_block(512,1024,4)
        
    def forward(self,x):
        x = self.basic_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        p3 = x
        x = self.res_block4(x)
        p2 = x
        p1 = self.res_block5(x)
        return p1,p2,p3

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
    def __init__(self):
        super(Yolo_body, self).__init__()
        self.darknet_body = Darknet_body()
        self.pyramid1,self.classifier1 = make_pyramids_classifier(1024,512,80)
        self.squeeze1 = Squeeze_block(512,256)
        self.pyramid2,self.classifier2 = make_pyramids_classifier(768,256,80)
        self.squeeze2 = Squeeze_block(256,128)
        self.pyramid3,self.classifier3 = make_pyramids_classifier(384,128,80)
        self.upsample = Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self,x):
        p1,p2,p3 = self.darknet_body(x)
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