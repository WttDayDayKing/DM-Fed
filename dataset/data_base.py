import math
import os.path

from torch.utils.data import DataLoader
from model.local_update import DatasetSplit,Dataset2
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import csv
from . import data_util

class cache_data(object):
    def __init__(self):
        self.data=[]
        self.targets=[]


def update_client_pim(self, current_task):
    if current_task == 0:
        return
    else:
        ng_server={}
        sample_pair_all={key:{} for key in range(self.args.num_users)}
        for idx in range(self.args.num_users):
            grad_norm_list = {}
            grad_norm_avg = {}
            sample_pair = {}
            #客户端在上一任务存在样本
            if idx not in self.args.local_train_old[current_task-1]:
                print(f"current:{current_task},client have no old sample:{idx}")
                continue
            else:
                train_old_sample = self.args.local_train_old[current_task - 1][idx]
            print(f"current:{current_task},client have old sample:{idx}")
            if len(self.args.cache_client_samples) > 0 and idx in self.args.cache_client_samples: ###已存在cache的旧数据
                train_cached_sample = self.args.cache_client_samples[idx]
                if len(train_cached_sample.data) > 0:
                    train_old_sample.data = train_old_sample.data + train_cached_sample.data
                    train_old_sample.targets = train_old_sample.targets + train_cached_sample.targets
            # print(f"train_old_sample.data:",train_old_sample.data)
            # print(f"train_old_sample.targets",train_old_sample.targets)
            # all_old_sample = train_old_sample
            idxs = [i for i in range(len(train_old_sample.data))]
            train_loader = DataLoader(Dataset2(train_old_sample, idxs),
                                      batch_size=self.args.local_pim_bs, shuffle=True, num_workers=8)
            optimizer = torch.optim.Adam(self.pim_model_list[idx].parameters(), lr=self.args.base_lr,
                                         betas=(0.9, 0.999),
                                         weight_decay=5e-4)
            for s in range(self.args.local_ep):
                for batch_idx, (images, labels) in enumerate(train_loader):
                    proximal_term = 0.0

                    model=self.pim_model_list[idx]
                    # 获取分类头的输出层
                    #num_classes = model.fc.out_features

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    images.requires_grad = True
                    model.to(self.args.device)
                    self.netglobals[idx].to(self.args.device)
                    output = model(images)
                    for w, w_t in zip(model.parameters(), self.netglobals[idx].parameters()):
                        proximal_term += (w - w_t).norm(2)
                    q_lambda = (1 - self.args.hyper) / (2 * self.args.hyper)
                    loss = F.cross_entropy(output, labels, reduction='none') + q_lambda * proximal_term
                    tocal_loss=loss.sum()
                    # 计算每个样本的梯度
                    grads = torch.autograd.grad(tocal_loss.sum(), images, create_graph=True)[0]  # 计算每个样本的梯度
                    # 计算每个样本梯度的 L2 范数
                    l2_norms = torch.norm(grads.view(grads.size(0), -1), p=2, dim=1)

                    optimizer.zero_grad()
                    tocal_loss.backward()
                    optimizer.step()

                    l2_norm_numpy=l2_norms.detach().cpu().numpy()
                    image_idx_range=range(batch_idx*self.args.local_pim_bs,batch_idx*self.args.local_pim_bs+len(l2_norm_numpy))
                    # 使用字典构建对应关系
                    correspondence = {idx: norm for idx, norm in zip(image_idx_range, l2_norm_numpy)}
                    for image_id in range(batch_idx*self.args.local_pim_bs,batch_idx*self.args.local_pim_bs+len(l2_norm_numpy)):
                        idx_image = train_old_sample.data[image_id]
                        idx_label = train_old_sample.targets[image_id]
                        if idx_image not in grad_norm_list:
                            grad_norm_list[idx_image] = torch.zeros(1)
                        grad_norm_list[idx_image] += correspondence[image_id]
                        if idx_image not in sample_pair:
                            sample_pair[idx_image] = idx_label
                sample_pair_all[idx] = sample_pair

            for imageid in grad_norm_list:
                # 求平均epoch的重要性指数
                grad_norm_avg[imageid] = grad_norm_list[imageid] / self.args.local_ep

            ng_server[idx]=grad_norm_avg
            #get_importent_sample(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all)  已计算出客户端cache size
            get_class_importent_sample(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all)

        # ##todo 将各客户端样本的grad范数上传至中间服务器，总体排序；
        #upload_proxy_server(self,ng_server,current_task,sample_pair_all) 将各客户端样本的grad范数上传至中间服务器，总体排序；
        return sample_pair

def get_class_importent_sample(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all):
    if self.args.use_dynamic_memory:
        if self.args.use_dynamic_class_memory:
           sample_pair=sample_pair_all[idx]
           # 1. 对梯度值进行排序
           sorted_grad_norms = sorted(grad_norm_avg.items(), key=lambda item: item[1], reverse=True)

           # 2. 筛选出每个类别的样本
           required_counts = self.args.avg_class_cache_size[idx]
           selected_samples = {class_id: [] for class_id in required_counts.keys()}

           for image, grad in sorted_grad_norms:
               if image in sample_pair:
                   class_id = sample_pair[image]
                   if len(selected_samples[class_id]) < required_counts[class_id]:
                       selected_samples[class_id].append((image, class_id))

           cache_data_var = cache_data()
           for samples in selected_samples.values():
               for image, label in samples:
                   cache_data_var.data.append(image)
                   cache_data_var.targets.append(label)
           self.args.cache_client_samples[idx] = cache_data_var

        else:
            get_importent_sample(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all)
    else:
        if len(grad_norm_list) < self.args.cached_sample_size:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
        else:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
            images_list = images_list[:self.args.cached_sample_size]
        cache_data_var = cache_data()
        for img in images_list:
            cache_data_var.data.append(img)
            cache_data_var.targets.append(sample_pair_all[idx][img])
        self.args.cache_client_samples[idx] = cache_data_var


def get_importent_sample(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all):
    if self.args.use_dynamic_memory:
        if len(grad_norm_list) < self.args.avg_client_memeory_size[idx]:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
        else:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
            memory_size = self.args.avg_client_memeory_size[idx]
            images_list = images_list[:memory_size]
    else:
        if len(grad_norm_list) < self.args.cached_sample_size:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
        else:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
            images_list = images_list[:self.args.cached_sample_size]
    cache_data_var = cache_data()
    for img in images_list:
        cache_data_var.data.append(img)
        cache_data_var.targets.append(sample_pair_all[idx][img])
    self.args.cache_client_samples[idx] = cache_data_var


def get_importent_sample_new(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all,task):
    if self.args.use_dynamic_memory:
        if task ==1:
            if len(grad_norm_list) < self.args.avg_client_memeory_size[idx]:
                images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
            else:
                images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
                memory_size = self.args.avg_client_memeory_size[idx]
                images_list = images_list[:memory_size]
        else:
            data_util.reduce_client_cache(args=self.args)
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
            memory_size = self.args.avg_client_memeory_size[idx]
            images_list = images_list[:memory_size]
    else:
        if len(grad_norm_list) < self.args.cached_sample_size:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
        else:
            images_list = sorted(grad_norm_avg, key=grad_norm_avg.get, reverse=True)
            images_list = images_list[:self.args.cached_sample_size]

    for img in images_list:
        self.args.cache_client_samples[idx].data.append(img)
        self.args.cache_client_samples[idx].targets.append(sample_pair_all[idx][img])



#将各客户端样本的grad范数上传至中间服务器，总体排序；
def upload_proxy_server(self,ng_server,current_task,sample_pair_all):
    if self.args.use_dynamic_memory:
        result_dict = proxy_server_cal_cache_size(ng_server)
        for client_id, images_pairs in result_dict.items():
            cache_data_var = cache_data()
            sample_pair = sample_pair_all[client_id]
            for image_dict in images_pairs:
                for image, value in image_dict.items():
                    cache_data_var.data.append(image)
                    cache_data_var.targets.append(sample_pair[image])
            self.args.cache_client_samples[client_id] = cache_data_var
            print(f"task:{current_task},client:{client_id},cache size:{len(cache_data_var.data)}")
            save_cache_sample_csv(cache_data_var, client_id, current_task, 'dynamic')
    else:
        for client_id, grad_pair in ng_server.items():
            sample_pair = sample_pair_all[client_id]
            if len(grad_pair) < self.args.cached_sample_size:
                images_list = sorted(grad_pair, key=grad_pair.get, reverse=True)
            else:
                images_list = sorted(grad_pair, key=grad_pair.get, reverse=True)
                images_list = images_list[:self.args.cached_sample_size]
            cache_data_var = cache_data()
            for img in images_list:
                cache_data_var.data.append(img)
                cache_data_var.targets.append(sample_pair[img])
            self.args.cache_client_samples[client_id] = cache_data_var
            save_cache_sample_csv(cache_data_var, client_id, current_task, 'fix')



def proxy_server_cal_cache_size(ng_server):
    # 提取 grad、image 和对应的 client_id
    grad_list = []
    result_dict={}
    for client_id, images in ng_server.items():
        for image, grad in images.items():
            grad_list.append((client_id, image, grad))

    # 按 grad 从大到小排序
    sorted_grads = sorted(grad_list, key=lambda x: x[2], reverse=True)

    # 获取前 100 个 grad 和对应的 image、client_id
    top_1200_grads = sorted_grads[:1200]

    # 输出结果
    for client_id, image, grad in top_1200_grads:
        if client_id not in result_dict:
            result_dict[client_id] = []  # 如果 client_id 不在字典中，初始化列表
        result_dict[client_id].append({image:grad})  # 添加 image 和 grad
    return result_dict

def save_cache_sample_csv(cache_data_var,client_id,current_task,log):
    # 将数据写入 CSV 文件
    csv_file_path = f'{log}_task_{current_task}_client_{client_id}_cache.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # 写入标题
        writer.writerow(['Index', 'Image', 'Label'])

        # 写入数据
        for index, (img, label) in enumerate(zip(cache_data_var.data, cache_data_var.targets)):
            writer.writerow([index, img, label])
def caculate_importence_score(self, train_loader, idx, grad_norm_list, train_old_sample):
    sample_pair = {}
    for batch_idx, (images, labels) in enumerate(train_loader):
        self.pim_model_list[idx].eval()
        images, labels = images.to(self.args.device), labels.to(self.args.device)
        images.requires_grad = True
        output = self.pim_model_list[idx](images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        grad_norm = torch.norm(images.grad, p=2)
        idx_image = train_old_sample.data[batch_idx]
        idx_label = train_old_sample.targets[batch_idx]
        if idx_image not in grad_norm_list:
            grad_norm_list[idx_image] =0
        grad_norm_list[idx_image] += grad_norm
        if idx_image not in sample_pair:
            sample_pair[idx_image]=idx_label
    return grad_norm_list, sample_pair

#icarl
def update_new_set(args,data_tmp,targets_tmp,user_groups):
    exemplar = []
    exemplar_targets = []
    for idx in range(args.num_users):
        index_list=user_groups[idx]
        class_index=set([targets_tmp[i] for i in index_list])
        if len(class_index)>0:
            if idx not in args.learned_numclass:
                args.learned_numclass[idx]=0
                args.learned_classes[idx]=[]
                args.known_class[idx]=0
            args.known_class[idx]=args.learned_numclass[idx]
            args.learned_numclass[idx] += len(class_index)
            args.learned_classes[idx].extend(class_index)
            if args.use_dynamic_memory:
                if idx >= len(args.avg_client_memeory_size):
                    args.avg_client_memeory_size.append(args.cached_sample_size)
                m = int(args.avg_client_memeory_size[idx] / args.learned_numclass[idx])#每个客户端为已经学习到的类别平分cache size
            else:
                m = int(args.cached_sample_size / args.learned_numclass[idx])
            _reduce_exemplar_sets(args,m)
            for i in class_index:
                indices = np.where(np.array(targets_tmp)==i)
                images = [data_tmp[id] for id in indices[0]]
                exemplar,exemplar_targets,m=_construct_exemplar_set(args,images,i, m,exemplar,exemplar_targets)
                cache_data_var = cache_data()
                cache_data_var.data = exemplar
                cache_data_var.targets = exemplar_targets
                if idx not in args.class_cache:
                    args.class_cache[idx]={}
                    args.class_cache_num[idx]={}
                if i not in args.class_cache[idx]:
                    args.class_cache[idx][i]=None
                    args.class_cache_num[idx][i]=0
                args.class_cache[idx][i]=cache_data_var#每个客户端为每个类的cache
                args.class_cache_num[idx][i]=m


def _reduce_exemplar_sets(args, m):
    for index in range(len(args.class_cache)):
        if index in args.class_cache:
            for class_id in args.class_cache[index]:
                length=len(args.class_cache[index][class_id].data)
                if length>0:
                    data_temp=args.class_cache[index][class_id].data
                    args.class_cache[index][class_id].data=data_temp[:m]
                    target_temp=args.class_cache[index][class_id].targets
                    args.class_cache[index][class_id].targets=target_temp[:m]

def _construct_exemplar_set(args, images,labels, m, exemplar,exemplar_targets):
    class_mean, feature_extractor_output = compute_class_mean(args, images, args.transform)
    dim=feature_extractor_output.shape[1]
    # print(f"feature_extractor_output:{feature_extractor_output.shape}")
    now_class_mean = np.zeros((1, dim))
    if args.use_uacl:
        m=math.floor(0.5*m)
    else:
        m=math.floor(m)
    for i in range(m):
        x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
        x = np.linalg.norm(x, axis=1)
        index = np.argmin(x)
        now_class_mean += feature_extractor_output[index]
        exemplar.append(images[index])
        exemplar_targets.append(labels)
    return exemplar,exemplar_targets,m

def compute_class_mean(args, images, transform):
    x = Image_transform(args,images, transform).cuda(args.device)
    torch.cuda.empty_cache()
    results = []
    batch_size=20
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        features = F.normalize(args.feature_extractor(x_batch).detach())
        results.append(features.cpu().numpy())
    feature_extractor_output = np.concatenate(results, axis=0)
        #feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
    class_mean = np.mean(feature_extractor_output, axis=0)
    return class_mean, feature_extractor_output


def Image_transform(args, images, transform):
    img_read=Image.open(images[0]).convert('RGB')
    data = transform(img_read).unsqueeze(0)
    for index in range(1, len(images)):
        new_image=Image.open(images[index]).convert('RGB')
        data = torch.cat((data, args.transform(new_image).unsqueeze(0)), dim=0)
    return data

#仅对当前任务的数据计算重要性
def update_client_pim_new(self, current_task):
    if current_task == 0:
        return
    else:
        ng_server={}
        sample_pair_all={key:{} for key in range(self.args.num_users)}
        for idx in range(self.args.num_users):
            grad_norm_list = {}
            grad_norm_avg = {}
            sample_pair = {}
            #客户端在上一任务存在样本
            if idx not in self.args.local_train_old[current_task-1]:

                continue
            else:
                train_old_sample = self.args.local_train_old[current_task - 1][idx]#每个客户端上一任务的样本


            idxs = [i for i in range(len(train_old_sample.data))]
            train_loader = DataLoader(Dataset2(train_old_sample, idxs),
                                      batch_size=self.args.local_pim_bs, shuffle=True, num_workers=8)
            optimizer = torch.optim.Adam(self.pim_model_list[idx].parameters(), lr=self.args.base_lr,
                                         betas=(0.9, 0.999),
                                         weight_decay=5e-4)
            for s in range(self.args.local_ep):
                for batch_idx, (images, labels) in enumerate(train_loader):
                    proximal_term = 0.0
                    model=self.pim_model_list[idx]
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    images.requires_grad = True
                    model.to(self.args.device)
                    self.netglobals[idx].to(self.args.device)
                    output = model(images)
                    for w, w_t in zip(model.parameters(), self.netglobals[idx].parameters()):
                        proximal_term += (w - w_t).norm(2)
                    q_lambda = (1 - self.args.hyper) / (2 * self.args.hyper)
                    loss = F.cross_entropy(output, labels, reduction='none') + q_lambda * proximal_term
                    tocal_loss=loss.sum()
                    # 计算每个样本的梯度
                    grads = torch.autograd.grad(tocal_loss.sum(), images, create_graph=True)[0]  # 计算每个样本的梯度
                    # 计算每个样本梯度的 L2 范数
                    l2_norms = torch.norm(grads.view(grads.size(0), -1), p=2, dim=1)

                    optimizer.zero_grad()
                    tocal_loss.backward()
                    optimizer.step()

                    l2_norm_numpy=l2_norms.detach().cpu().numpy()
                    image_idx_range=range(batch_idx*self.args.local_pim_bs,batch_idx*self.args.local_pim_bs+len(l2_norm_numpy))
                    # 使用字典构建对应关系
                    correspondence = {idx: norm for idx, norm in zip(image_idx_range, l2_norm_numpy)}
                    for image_id in range(batch_idx*self.args.local_pim_bs,batch_idx*self.args.local_pim_bs+len(l2_norm_numpy)):
                        idx_image = train_old_sample.data[image_id]
                        idx_label = train_old_sample.targets[image_id]
                        if idx_image not in grad_norm_list:
                            grad_norm_list[idx_image] = torch.zeros(1)
                        grad_norm_list[idx_image] += correspondence[image_id]
                        if idx_image not in sample_pair:
                            sample_pair[idx_image] = idx_label
                sample_pair_all[idx] = sample_pair

            for imageid in grad_norm_list:
                # 求平均epoch的重要性指数
                grad_norm_avg[imageid] = grad_norm_list[imageid] / self.args.local_ep

            ng_server[idx]=grad_norm_avg
            get_importent_sample_new(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all,current_task)  #已计算出客户端cache size
            #get_class_importent_sample(self,grad_norm_list,idx,grad_norm_avg,sample_pair_all)

        # ##todo 将各客户端样本的grad范数上传至中间服务器，总体排序；
        #upload_proxy_server(self,ng_server,current_task,sample_pair_all) 将各客户端样本的grad范数上传至中间服务器，总体排序；
        return sample_pair
