import time

from torch.utils.data import Dataset,DataLoader
import torch
import logging
import torch.nn.functional as F
import numpy as np
import copy
from torchvision import transforms
import torch.nn as nn
from PIL import Image
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)


    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Dataset2(Dataset):
    def __init__(self,dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self,item):
        image, label = self.dataset[self.idxs[item]]
        image = Image.open(image).convert('RGB')
        classify_image = self.transform(image)
        total_image = classify_image
        return total_image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),batch_size=self.args.local_bs, shuffle=True, num_workers=8)
        #self.current_train_loader=DataLoader(DatasetSplit(args.current_trainset,args.current_user_groups[client]),batch_size=self.args.local_bs, shuffle=True, num_workers=8)
        self.lr=self.args.base_lr
        self.local_ep=self.args.local_ep
        self.device=self.args.device
        self.data_num=len(idxs)
        self.current_data_images=[dataset.data[i] for i in idxs]
        self.current_data_label=[dataset.targets[i] for i in idxs]
        #self.train(self,model=model,writer=writer,client_id=client_id)


    def train(self,model,writer,client_id,task):
        epoch_loss = []
        model.train()
        old_model=copy.deepcopy(model)
        old_model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                     weight_decay=5e-4)
        if torch.cuda.device_count() > 1:
            model= torch.nn.DataParallel(model)
        # 存储每张图像在每次迭代中的预测概率
        all_probabilities = {}
        R = torch.zeros((self.local_ep, self.data_num, self.args.n_classes))
        for epoch in range(self.local_ep):
            batch_loss = []
            prob_lab=0
            start_time=time.time()
            for batch_idx,(images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model(images)
                loss =  F.cross_entropy(logits, labels)

                # ####icarl需要在增两阶段加入蒸馏损失
                if self.args.use_carl and self.args.use_uacl is False and task>0 and self.args.known_class[client_id]>0:
                    print(f"task{task},known class:{self.args.known_class[client_id]}")
                    loss_kd = self._KD_loss(
                        logits,
                        old_model(images),
                        2,
                    )
                    loss = loss + loss_kd
                if self.args.use_uacl:
                    ####获取每个图像预测的每个类的概率
                    probs = torch.nn.functional.softmax(logits, dim=1)  # dim=1 表示对每一行进行 softmax,batch_sizeXclass
                    # 将每个图像的概率存储
                    for i, prob in enumerate(probs):
                        R[epoch, i+prob_lab*self.args.local_bs, :] = prob
                    ####计算损失
                    loss_uacl=self._KD_loss(logits,old_model(images),2)
                    loss=loss+loss_uacl
                if self.args.use_margin_regular:
                    softmax_output = torch.softmax(logits, dim=1)
                    magnitude = torch.norm(softmax_output, p=2)
                    loss=loss+self.args.lambda_value * torch.log(1 + magnitude ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{client_id}/loss_train', loss.item())

                prob_lab+=1
            end_time=time.time()
            cost_time=end_time-start_time
            print(f"batch:{batch_idx}")
            print(f"one epoch cost time:{cost_time}")
            epoch_loss.append(np.array(batch_loss).mean())
        #model,next_step_model=self.train_one_step(model)#### one-more step train
        optimizer.zero_grad(set_to_none=True)
        if self.args.use_uacl:
            ###计算样本不确定性，并存储
            self.cal_uncertion(R=R,client_id=client_id)
        return model.state_dict(), np.array(epoch_loss).mean(),None

    def cal_uncertion(self,R,client_id):
        # 计算均值
        mu = torch.mean(R, dim=0)  ###平均epochs次
        # 计算不确定性
        squared_diff = (R - mu.unsqueeze(0)) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=0)
        U = torch.sum(torch.sqrt(sum_squared_diff / self.args.local_ep), dim=1)
        # 对 U 进行排序，返回排序后的值和对应的索引
        sorted_U, indices = torch.sort(U, descending=True)
        sort_uacl_data=[self.current_data_images[idx] for idx in indices]
        sort_uacl_targets=[self.current_data_label[idx] for idx in indices]
        for i in (set(sort_uacl_targets)):
            if i in self.args.class_cache_num[client_id]:
                m=self.args.class_cache_num[client_id][i]####需要存储多少个
                if m == 0:
                    continue
                else:
                    # 获取该类别的索引
                    cls_indices = np.where(sort_uacl_targets == i)[0]
                    # 检查是否有足够的索引
                    if len(cls_indices) > m:
                        # 随机选择前 m 个索引（可以根据需求选择排序或其他方式）
                        top_m_indices = cls_indices[:m]
                    else:
                        top_m_indices = cls_indices  # 不足 m 个时，返回所有索引
                    if i not in self.args.class_cache[client_id]:
                        print(f"当前类:{i}不在class_cache")
                    else:
                        print(f"当前类:{i}在class_cache")
                        cache_data_var = self.args.class_cache[client_id][i]
                        cache_data_var.data.extend(sort_uacl_data[ins] for ins in top_m_indices.tolist())
                        cache_data_var.targets.extend(sort_uacl_targets[ins] for ins in top_m_indices.tolist())
                        self.args.class_cache[client_id][i] = cache_data_var

    def train_one_step(self, model):
        """
        train one step using the dataset(x,y) to obtain the new model parameter,
        but we don't replace the self.model by the new model parameter, we only want
        to calculate the new model parameter.
        """
        # save the old model parameter
        old_model = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.base_lr, betas=(0.9, 0.999),
                                     weight_decay=5e-4)
        model.train()
        for e in range(self.args.local_ep):
            for batch_idx,(images, labels) in enumerate(self.train_loader):
                inputs, labels = images.to(self.args.device), labels.to(self.args.device)
                # self.person_optimizer.zero_grad()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss =  F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
        self.args.next_step_model = {key: copy.deepcopy(value) for key, value in model.named_parameters()}
        next_step_model=copy.deepcopy(self.args.next_step_model)
        # restore the old model
        model.load_state_dict(old_model)
        return model,next_step_model

    # 定义知识蒸馏损失函数
    def distillation_loss(self,y_student, y_teacher, temperature=2.0):
        # 计算蒸馏损失
        soft_targets = F.softmax(y_teacher / temperature, dim=1)
        student_outputs = F.log_softmax(y_student / temperature, dim=1)
        return F.kl_div(student_outputs, soft_targets, reduction='batchmean') * (temperature ** 2)

    # 新客户端使用的训练函数
    def new_client_train_old(self, model1, model2, temperature=2.0):
        model1.eval()  # 教师模型在评估模式
        optimizer = torch.optim.Adam(model2.parameters(), lr=self.args.base_lr, betas=(0.9, 0.999),
                                     weight_decay=5e-4)
        for epoch in range(self.args.local_ep):
            for data, target in self.train_loader:
                data, target = data.to(self.args.device), target.to(self.args.device)

                # 教师模型的输出
                with torch.no_grad():
                    teacher_output = model1(data)

                # 学生模型的输出
                student_output = model2(data)

                # 计算损失
                loss = self.distillation_loss(student_output, teacher_output, temperature)

                # 优化学生模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model2, next_step_model = self.train_one_step(model2)  #### one-more step train
        optimizer.zero_grad(set_to_none=True)
        return model2.state_dict(), next_step_model

    # 新客户端使用的训练函数
    def new_client_train(self, model1, model2, temperature=2.0):
        model1.eval()  # 教师模型在评估模式
        model2.train()
        # 使用 DataParallel 包装模型
        if torch.cuda.device_count() > 1:
            model1 = nn.DataParallel(model1)
            model2=nn.DataParallel(model2)
        optimizer = torch.optim.Adam(model2.parameters(), lr=self.args.base_lr, betas=(0.9, 0.999),
                                     weight_decay=5e-4)
        for epoch in range(self.args.local_ep):
            start_time=time.time()
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)

                # 教师模型的输出
                with torch.no_grad():
                    teacher_output = model1(data)

                # 学生模型的输出
                student_output = model2(data)

                # 计算损失
                loss = F.cross_entropy(student_output,target)+self.distillation_loss(student_output, teacher_output, temperature)
                if self.args.use_margin_regular:
                    softmax_output = torch.softmax(student_output, dim=1)
                    magnitude = torch.norm(softmax_output, p=2)
                    loss=loss+self.args.lambda_value * torch.log(1 + magnitude ** 2)

                # 优化学生模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time=time.time()
            cost_time=end_time-start_time
            print(f"one epoch cost time:{cost_time}")
            print(f"batch:{batch}")
        #model2, next_step_model = self.train_one_step(model2)  #### one-more step train
        optimizer.zero_grad(set_to_none=True)
        return model2.state_dict(), None

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        # Adjust dimensions if applicable, be careful with this!
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

    ###UACL loss
    def _uacl_loss(self,logits, soft_labels):
        sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        S_theta_t_1 = sigmoid(logits)
        S_theta = sigmoid(soft_labels)
        first_term = torch.sum(S_theta_t_1 * torch.log(S_theta))
        second_term = torch.sum((1 - S_theta_t_1) * torch.log(1 - S_theta))
        return (first_term + second_term)

