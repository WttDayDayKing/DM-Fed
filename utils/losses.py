import logging

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

# LA损失使类均匀
class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        if torch.cuda.is_available():
            cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        else:
            cls_num_list = torch.FloatTensor(cls_num_list)
        #cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_num_list=cls_num_list.cpu()
        logging.info(f"class_num_list:{cls_num_list}")
        cls_num_list = numpy.array(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        logging.info(f"cls_p_list:{cls_p_list}")
        cls_p_list=torch.from_numpy(cls_p_list)
        m_list = tau * torch.log(cls_p_list)
        logging.info(f"m_list:{m_list}")
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        # logging.info(f"x:{x}")
        # logging.info(f"x size:{x.shape}")
        # logging.info(f"target:{target}")
        # logging.info(f"m_list size:{self.m_list.shape}")
        # logging.info(f"target size:{target.shape}")
        # X就是logits
        #x_m = x + self.m_list
        return F.cross_entropy(x, target, weight=self.weight)

# 知识蒸馏使用的损失函数
class LA_KD(nn.Module):
    def __init__(self, cls_num_list, tau=1):
        super(LA_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x, target, soft_target, w_kd):
        x_m = x + self.m_list
        log_pred = torch.log_softmax(x_m, dim=-1)
        log_pred = torch.where(torch.isinf(log_pred), torch.full_like(log_pred, 0), log_pred)

        kl = F.kl_div(log_pred, soft_target, reduction='batchmean')

        return w_kd * kl + (1 - w_kd) * F.nll_loss(log_pred, target)

