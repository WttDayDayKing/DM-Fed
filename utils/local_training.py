import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from .losses import LogitAdjust, LA_KD


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            #logging.info(f"global-outputs:{outputs}")
            _, predicted = torch.max(outputs.data, 1)
            #logging.info(f"gloabl-predicted:{predicted}")
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred

def localtest(w_local,test_set,args):
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for id in range(args.num_users):
        pred = np.array([])
        w_local.eval()
        with torch.no_grad():
            for images,labels in test_loader:
                images = images.to(args.device)
                outputs = w_local(images)
                #logging.info(f"client:{id},outputs:{outputs}")
                _, predicted = torch.max(outputs.data, 1)
                #logging.info(f"client:{id}, predicted:{predicted}")
                pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred



class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, item):
        image, label = self.dataset['images'][item], self.dataset['labels'][item]
        return image, label

    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()



class LocalUpdate(object):
    def __init__(self, args, id, dataset):
        self.args = args
        self.id = id
        self.local_dataset = DatasetSplit(dataset)
        #self.class_num_list = self.local_dataset.get_num_of_each_class(self.args)
        # logging.info(
        #     f'client{id} each class num: {self.class_num_list}, total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr

    # 使用交叉熵函数进行模型训练
    def train_LA(self, net, writer):
        net.train()

        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #images,labels=images.to(device),labels.to(device)

                logits = net(images)
                #logging.info(f"logits:{logits}")
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()
    
    #噪声客户端使用知识蒸馏训练
    def train_FedNoRo(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        criterion = LA_KD(cls_num_list=self.class_num_list)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda.to(self.args.device), labels.to(self.args.device)

                logits = student_net(images)
                # 教师模型不使用计算梯度优化模型参数，即按照前10个round的base模型直接计算损失
                # 知识蒸馏是在教师模型（warm up模型）上输入图像计算损失，在学生模型上基于图像训练模型，再根据criterion计算教师模型和学生模型预测结果的损失进行优化即可。
                with torch.no_grad():
                    teacher_output = teacher_net(images)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1)

                loss = criterion(logits, labels, soft_label, weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

def train_base(args,net,writer,train_loader,client_id):
    net.train()
    epoch_loss = []
    # set the optimizer
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
    for epoch in range(args.local_ep):
        batch_loss = []
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            logits = net(images)
            logging.info(f"logits:{logits}")
            logging.info(f"labels:{labels}")
            # labels = torch.tensor(labels, dtype=torch.float)
            loss = F.cross_entropy(logits, labels)
            logging.info(f"loss:{loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            writer.add_scalar(
                f'client{client_id}/loss_train', loss.item())
            #self.iter_num += 1
        epoch = epoch + 1
        epoch_loss.append(np.array(batch_loss).mean())

    return net.state_dict(), np.array(epoch_loss).mean()
