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

def Darknet_body():
    return Sequential(
    Basic_block(3,32),
    Residual_block(32,64,1),
    Residual_block(64,128,2),
    Residual_block(128,256,8),
    Residual_block(256,512,8),
    Residual_block(512,1024,4)
    )