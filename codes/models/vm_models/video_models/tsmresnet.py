import torch.nn as nn
import torch
from .resnet_basemodule import *



class TSM_Net(nn.Module):

    def __init__(self, resnet_model, n_segments, temporal_pool=True,
                 spatial_pool=True):
        super(TSM_Net, self).__init__()

        self.n_segments = n_segments

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = list(resnet_model.children())[9]

        self.temporal_pool = temporal_pool
        self.spatial_pool = spatial_pool


    def forward(self, x):
        # short-term motion
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        # b, t = bt // self.n_segments, self.n_segments

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1_bak(x)
        x2 = self.layer2_bak(x1)
        x3 = self.layer3_bak(x2)
        x4 = self.layer4_bak(x3)  ## bt,c,h,w


        x = x4
        if self.temporal_pool:
            x = x.view((b, t) + x.size()[-3:])  # b,t,c,h,w
            x = x.mean(dim=1)  # b,c,h,w
            if self.spatial_pool:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
        else:
            if self.spatial_pool:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)  # bt,c
                x = x.view(b, t, -1).permute(0, 2, 1).contiguous()  # b,t,c,h,w
            else:
                x = x.view((b, t) + x.size()[-3:])  # b,t,c,h,w
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # b,t,c,h,w
        return x


def tsmresnet(base_model, num_segments=8, shift_start=0, temporal_pool=True,spatial_pool=True):
    if ("18" in base_model):
        resnet_model = resnet18(pretrained=True, shift_start=shift_start, num_segments=num_segments)
    elif ("50" in base_model):
        resnet_model = resnet50(pretrained=True, shift_start=shift_start, num_segments=num_segments)
    else:
        raise NotImplementedError
    model = TSM_Net(resnet_model, num_segments, temporal_pool=temporal_pool,spatial_pool=spatial_pool)

    return model
