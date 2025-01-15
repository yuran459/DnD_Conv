import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DnD_Conv_branch(nn.Module):
    # Dimensional-Decomposed Conv
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.kernal = kernel_size
        if self.kernal>1:
            self.HConv = nn.Conv2d(in_channels, out_channels, (1,kernel_size), (1,stride), [0,autopad(kernel_size, padding, dilation)], dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
            self.VConv = nn.Conv2d(out_channels, out_channels, (kernel_size,1), (stride,1), [autopad(kernel_size, padding, dilation),0], dilation=dilation, groups=out_channels, bias=bias, padding_mode=padding_mode)
            self.PWConv = nn.Conv2d(in_channels, out_channels, 1, (1,stride), 0, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
            if autopad(kernel_size, padding, dilation) == autopad(kernel_size, None, dilation):
                self.resize_layer = nn.Identity()
            else:
                self.resize_layer = nn.AvgPool2d((1,kernel_size),stride=1,padding=[0,autopad(kernel_size, padding, dilation)])
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        

    def forward(self, x):
        if self.kernal>1:
            return self.VConv(self.HConv(x)+self.resize_layer(self.PWConv(x)))
        else:
            return self.conv(x)
    
    def switch_to_deploy(self):
        if self.kernal>1:
            default_pad = autopad(self.HConv.kernel_size[1], None, self.HConv.dilation[1])
            if autopad(self.HConv.kernel_size[1], self.HConv.padding[1], self.HConv.dilation[1]) == default_pad:
                ResConv_weight = F.pad(self.PWConv.weight.data,[default_pad,default_pad])
            else:
                ResConv_weight = self.PWConv.weight.data.repeat(1,1,1,self.kernal)/self.kernal
            HConv_weight_t = ResConv_weight+self.HConv.weight.data
            self.conv = nn.Conv2d(self.HConv.in_channels, self.HConv.out_channels, self.HConv.kernel_size[1], self.HConv.stride, self.HConv.padding[1], dilation=self.HConv.dilation, groups=self.HConv.groups, bias=self.HConv.bias, padding_mode=self.HConv.padding_mode)
            self.conv.bias = self.VConv.bias
            self.conv.weight.data = torch.mul(HConv_weight_t.repeat(1,1,self.VConv.kernel_size[0],1),
                                              self.VConv.weight.data.repeat(1,self.HConv.in_channels,1,self.HConv.kernel_size[1]))
            if hasattr(self, 'HConv'):
                self.__delattr__('HConv')
            if hasattr(self, 'VConv'):
                self.__delattr__('VConv')
            if hasattr(self, 'PWConv'):
                self.__delattr__('PWConv')
            if hasattr(self, 'resize_layer'):
                self.__delattr__('resize_layer')

    def get_weight(self):
        return self.conv.weight.data
    
    def get_bias(self):
        return self.conv.bias.data
    
    def get_conv(self):
        return self.conv

class DnDConv(nn.Module):
    # DnD Conv Block
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = DnD_Conv_branch(c1, c2, k, s, padding=autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv1(x)))

    def switch_to_deploy(self):
        self.conv1.switch_to_deploy()
        self.conv = self.conv1.get_conv()
        if hasattr(self, 'conv1'):
            self.__delattr__('conv1')