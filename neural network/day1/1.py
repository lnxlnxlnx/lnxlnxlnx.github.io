import torch
from torch import nn

class Network(nn.Module): #这个是继承
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(,256)    #这个是设计第一层的大小
        self.layer2=nn.Linear(256,10)
        
    def forward(self,x):
        x=x.view(,)
        x=self.layer1(x)
        x=torch.relu(x)     #上面两步做的都是对数据进行变换
        return self.layer2(x)