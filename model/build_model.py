import logging

import torch.nn as nn
from .all_models import get_model, modify_last_layer
import torchvision.models as models

def build_model(args):
    # choose different Neural network model for different args
    if args.use_uacl==True:
        model=CustomResNet18(args.n_classes)
    else:
        model = get_model(args.model, False)
        model, _ = modify_last_layer(args.model, model, args.n_classes)

    #model = model.to(args.device)

    return model


def build_model_new(args):
    # choose different Neural network model for different args
    model = get_model(args.model, False)
    model, _ = modify_last_layer(args.model, model, args.n_classes)

    #model = model.to(args.device)

    return model


class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()

        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(pretrained=False)

        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # 添加Dropout层
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # 保存原始的残差块
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3

        # 替换第二层和第三层的块
        self.resnet.layer2 = self._replace_with_dropout(self.layer2)
        self.resnet.layer3 = self._replace_with_dropout(self.layer3)

    def _replace_with_dropout(self, layer):
        # 在每个残差块后添加Dropout层
        new_layers = []
        for block in layer:
            new_layers.append(block)
            new_layers.append(self.dropout1)  # 在第二层的每个块后添加Dropout
        return nn.Sequential(*new_layers)

    def forward(self, x):
        return self.resnet(x)


