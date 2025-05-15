import torch.nn as nn
import torch

#   定义18层网络和34层网络的残差结构
class BasicBlock(nn.Module):
    #   expansion对应残差结构中，主分支的卷积核数有没有发生变化
    #   18层和34层的网络没有变化，50层、101层和152层的网络发生变化
    expansion = 1

    #   downsample下采样参数，用于残差分支的尺寸维度缩放
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        #   BN层
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        #   下采样方法
        self.downsample = downsample

    def forward(self, x):
        #   分支线上的输出
        #   将x赋值给identity，捷径上不执行下采样的输出值
        identity = x
        #   判断downsample=None，对捷径执行下采样操作并输出
        if self.downsample is not None:
            identity = self.downsample(x)

        #   主支线上的输出
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #   主分支输出加上捷径分支输出
        out += identity
        out = self.relu(out)

        return out


#   定义50层网络、101层网络和152层网络的残差结构
class Bottleneck(nn.Module):
    #   expansion对应残差结构中，主分支的卷积核数有没有发生变化
    #   50层、101层和152层的网络发生变化,其残差结构中第三层的卷积核个数为前两层的四倍，例如64—64—256
    expansion = 4

    """与ResNet的区别:初始化中加入了 groups 和 width_per_group"""
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dowmsample = downsample

    def forward(self, x):
        identity = x
        if self.dowmsample is not None:
            identity = self.dowmsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

"""与ResNet的区别:初始化中加入了 groups 和 width_per_group"""
#   定义ResNet网络
class ResNeXt(nn.Module):
    #   block：残差结构 block_num(list)：残差结构数量 include_top=True：方便在ResNet上搭建其他网络
    def __init__(self, block, block_num, num_classes=1000, include_top=True,
                 groups=1, width_per_group=64):
        super(ResNeXt, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        #   输出层+全连接层
        if self.include_top:
            self.avepool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        #   对卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    #   定义生成残差结构的方法
    #   block：残差结构 channel：第一层卷积核的个数 block_num：残差结构数量
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        #   判断通道数是否发生变化，来执行下采样操作
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        #   添加第一层残差结构
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride,
                            groups=self.groups, width_per_group=self.width_per_group))
        #   根据expansion来生成实线和虚线的残差结构
        self.in_channel = channel * block.expansion

        #   残差结构中除了第一层均为实线结构，将其依次添加到layers中
        for _ in range(1, block_num):  #    从1开始，即实线残差结构从第二层开始
            layers.append(block(self.in_channel, channel,
                                groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    #   正向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avepool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnext50_32x4d(num_classes=13, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNeXt(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
                   groups=groups, width_per_group=width_per_group)
