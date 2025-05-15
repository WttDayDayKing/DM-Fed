from torchvision.models import resnext50_32x4d
import torch
import torch.nn as nn

class CustomResNeXt50(nn.Module):
    def __init__(self, dropout_prob=0.5,num_class=13):
        super(CustomResNeXt50, self).__init__()
        # 加载预训练的 ResNeXt50
        self.resnext = resnext50_32x4d(pretrained=True)

        # 在第二层和第三层后面添加 Dropout
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        # 替换原有的全连接层
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(num_features, num_class)  # 根据需要修改输出数量

    def forward(self, x):
        # 前向传播
        x = self.resnext.conv1(x)
        x = self.resnext.bn1(x)
        x = self.resnext.relu(x)
        x = self.resnext.maxpool(x)

        # 第一层
        x = self.resnext.layer1(x)

        # 第二层 + Dropout
        x = self.resnext.layer2(x)
        x = self.dropout1(x)

        # 第三层 + Dropout
        x = self.resnext.layer3(x)
        x = self.dropout2(x)

        # 最后，继续前向传播
        x = self.resnext.layer4(x)
        x = self.resnext.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnext.fc(x)

        return x