import numpy as np
from . import Matek19
from . import Acevedo20
from . import WBC
from . import data_base
import torch.nn.functional as F
from collections import defaultdict
import torch
from collections import Counter
import os
from torch.utils.data import Dataset
from utils.sampling import non_iid_dirichlet_sampling
import math
import copy
class SubsetDataset(Dataset):
    def __init__(self, dataset_A, indices):
        self.data = []
        self.targets = []

        # 提取指定索引的数据和目标
        for idx in indices:
            data, target = dataset_A.data[idx],dataset_A.targets[idx] # 从 A 中获取对应的 data 和 targets
            self.data.append(data)  # 添加到 B 的 data 中
            self.targets.append(target)  # 添加到 B 的 targets 中

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_dataloader_cil(args, session,trainset_client,testset):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_perclient(args,trainset_client,testset)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args,trainset_client,testset)
    return trainset, trainloader, testloader

###获取第一阶段所有训练数据集
def get_all_dataset(args,current_task,netglobals,pim_model_list):
    if args.il_type == 'CIL':
        if current_task == 0:
            class_index = np.arange(args.base_class)
            old_class_index = None
        else:
            class_index = np.arange(args.base_class + (current_task - 1) * args.way,
                                       args.base_class + current_task * args.way)
            old_class_index = np.arange(args.base_class + (current_task - 1) * args.way)

    ##　todo　待完善
    elif args.il_type == 'DIL':
         if current_task == 0:
             class_index=np.arange(args.base_class)
             old_class_index=None
    elif args.il_type == 'CIL+DIL':
         if current_task == 0:
             class_index = np.arange(args.base_class)
             old_class_index = None
         elif current_task == 1:
             class_index = np.arange(args.base_class + (current_task - 1) * args.way,args.base_class + current_task * args.way)
             old_class_index=None

    if current_task==0:
        base_session=True
    else:
        base_session=False

    if args.dataset =='Matek-19':
        trainset = Matek19.matek19(args, is_train=True, index=class_index,
                                         base_session=base_session,old_index=old_class_index,netglobals=netglobals,current_task=current_task,pim_model_list=pim_model_list)
        testset = Matek19.matek19(args, is_train=False, index=class_index,
                                        base_session=base_session,old_index=old_class_index)
    elif args.dataset == 'Acevedo-20':
        trainset = Acevedo20.acevedo20(args, is_train=True, index=class_index,
                                         base_session=base_session,old_index=old_class_index,netglobals=netglobals,current_task=current_task,pim_model_list=pim_model_list)
        testset=Acevedo20.acevedo20(args, is_train=False, index=class_index,
                                        base_session=base_session,old_index=old_class_index)
    elif args.dataset=='Labelled':
        trainset= WBC.wbc(args, is_train=True, index=class_index,
                                         base_session=base_session,old_index=old_class_index,netglobals=netglobals,current_task=current_task,pim_model_list=pim_model_list)
        testset=WBC.wbc(args, is_train=False, index=class_index,
                                        base_session=base_session,old_index=old_class_index)
    else:
        trainset = Matek19.matek19(args, is_train=True, index=class_index,
                                   base_session=base_session, old_index=old_class_index, netglobals=netglobals,
                                   current_task=current_task, pim_model_list=pim_model_list)
        testset = Matek19.matek19(args, is_train=False, index=class_index,
                                  base_session=base_session, old_index=old_class_index)
    return trainset,testset



def get_base_dataloader_perclient(args,trainset_client,testset):

    trainloader = torch.utils.data.DataLoader(dataset=trainset_client, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset_client, trainloader, testloader

def get_new_dataloader(args,session):
    if args.dataset == 'Matek-19':
        class_index = np.arange(args.base_class+(session-1)*args.way, args.base_class+session*args.way)
        old_class_index = np.arange(args.base_class+(session-1)*args.way)
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False,  crop_transform=None, secondary_transform=None, old_index=old_class_index)
    return

# 为每个客户端平均划分训练集，即得到每个客户端对应的样本索引
def avg_distribution_train_per_client(args,trainset):
    num_clients = args.n_clients
    value_indices = defaultdict(list)

    # Populate the dictionary with indices
    for index, value in enumerate(trainset.target):
        value_indices[value].append(index)

    # Distribute indices among clients
    clients_indices = [[] for _ in range(num_clients)]

    for indices in value_indices.values():
        for i, index in enumerate(indices):
            clients_indices[i % num_clients].append(index)

#根据客户端id获取样本数据
def get_dataset_by_clientid(trainset,client_id,clients_indices):
    trainset2=trainset.copy()
    trainset.data=[trainset2.data[i] for i in clients_indices[client_id]]
    trainset.target=[trainset2.target[i] for i in clients_indices[client_id]]
    return trainset


def get_client_dict_dataset(args,current_task,pim_model_list,netglobals):
    trainset,testset = get_all_dataset(args,current_task,netglobals,pim_model_list)
    if args.iid or current_task==0:
        user_groups = cifar_iid(trainset, args.num_users)
    else:
        ##
        #print(f"targets class:{set(trainset.targets)}")
        user_groups = non_iid_dirichlet_sampling(current_task,trainset.targets,args.num_class,args.non_iid_prob_class, args.num_users, seed=100, alpha_dirichlet=args.alpha_dirichlet)
    args.current_trainset=trainset
    args.current_user_groups=user_groups
    #args.task_cache_size[current_task]=math.floor((len(list(trainset.data))/11456)*args.server_all_cache_size)
    args.client_data_size=[len(user_groups[i]) for i in range(args.num_users)]
    args.all_data_size=len(trainset)
    print(f"current task :{current_task} all data size:{len(trainset)}")
    print(f"task:{current_task},client data num distribution:{args.client_data_size}")
    all_targets=trainset.targets
    # for client_id in range(args.num_users):
    #     client_target=[all_targets[i] for i in user_groups[client_id]]
        #print(f"task:{current_task},client:{client_id},client_class:{client_target},client_class_num:{Counter(client_target)}")
    if args.use_carl:
        data_base.update_new_set(args,trainset.data,trainset.targets,user_groups)####

    #print(f"session:{current_task},client_data:{user_groups}")
    if current_task not in args.local_train_old:
        args.local_train_old[current_task] = {}
    for i in (user_groups.keys()):
        old_train=SubsetDataset(trainset,user_groups[i])
        #print("train_list:", old_train)
        if i not in args.local_train_old[current_task]:
            args.local_train_old[current_task][i] = []  ####仅存储当前任务的各客户端样本，为训练个性化本地模型提供数据
            args.local_train_old[current_task][i] = old_train
            print(f"task:{current_task},client:{i},calss num:{Counter(old_train.targets)}")
    ####加载所有旧数据
    if current_task>0 and args.get_all_old_data:
       for task in range(0,current_task):
           for client_id in range(args.num_users):
               if client_id not in args.local_train_old[task]:
                   continue
               else:
                   old_data=args.local_train_old[task][client_id]
                   old_length=len(trainset.data)
                   trainset.data=trainset.data+old_data.data
                   trainset.targets=trainset.targets+old_data.targets
                   new_leng = len(trainset.data)
                   for i in range(old_length - 1, new_leng):
                       user_groups[client_id].add(i)
    ##加载示例集
    ###加载历史任务cached的样本
    if args.use_pim:
       for idx in range(args.num_users):
           if len(args.cache_client_samples)>0:
               if idx not in args.cache_client_samples:
                   continue
               else:
                   datalist=args.cache_client_samples[idx]
                   leng=len(trainset.data)
                   trainset.data=trainset.data+datalist.data
                   trainset.targets=trainset.targets+datalist.targets
                   new_leng=len(trainset.data)
                   for i in range(leng-1,new_leng):
                       user_groups[idx].add(i)
    elif args.use_carl:
        for idx in range(args.num_users):
            if len(args.class_cache) > 0:
                if idx not in args.class_cache:
                    continue
                else:
                    leng = len(trainset.data)
                    class_list=args.class_cache[idx]
                    for class_id in class_list:
                        if class_id in trainset.targets:
                           continue
                        else:
                            data_list = class_list[class_id]
                            trainset.data = trainset.data + data_list.data
                            trainset.targets = trainset.targets + data_list.targets
                    new_leng = len(trainset.data)
                    for i in range(leng - 1, new_leng):
                        user_groups[idx].add(i)
    sorted_user_groups = {key: user_groups[key] for key in sorted(user_groups.keys())}
    return sorted_user_groups,trainset,testset

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
###计算客户端的重要型指数：Shannon Entropy（客户端样本分布多样性）+客户端训练后与原全局模型参数差异
def cal_client_importent_score(netglobals,w_locals,args,user_groups,trainset,rnd):
    weight_diff={}
    shannon_Entropy={}
    combined = {}
    client_class_cache={}
    task_server_all_cache_size=args.server_all_cache_size
    for client_id in range(len(w_locals)):
        l2_norm=0.0
        for key in netglobals[client_id].state_dict().keys():
            if key in w_locals[client_id].state_dict().keys():
                net_param = netglobals[client_id].state_dict()[key].to(args.device)
                local_param = w_locals[client_id].state_dict()[key].to(args.device)
                diff=net_param.float()-local_param.float()
                l2_norm += torch.norm(diff).item() ** 2  # L2 范数的平方
        weight_diff[client_id]=l2_norm ** 0.5
    print(f"client_diff:{weight_diff}")
    for client in range(len(user_groups)):
        targets_class=[trainset.targets[id] for id in user_groups[client]]
        class_counts = Counter(targets_class)
        # 计算总样本数
        total_samples = sum(class_counts.values())
        # 计算每个类的概率
        class_probabilities = {count / total_samples for count in class_counts.values()}
        # 计算 Shannon Entropy
        entropy = -sum(p * np.log2(p) for p in class_probabilities if p > 0)
        shannon_Entropy[client]=entropy
    print(f"client data entropy:{shannon_Entropy}")
    # 相加对应的值
    for key in weight_diff:
        if key in shannon_Entropy:  # 确保在两个字典中都有这个键
            combined[key] = weight_diff[key] + shannon_Entropy[key]
    # 计算总和
    total_sum = sum(combined.values())
    # 归一化各客户端权重
    normalized = {key: value / total_sum for key, value in combined.items()}
    args.client_memeory_size[rnd] = [value * task_server_all_cache_size for value in
                                     normalized.values()]
    args.client_memeory_size[rnd] = [math.floor(x) for x in args.client_memeory_size[rnd]]
    print(f"----------------------------------------------")
    print(f"rnd:{rnd},client_cache_size:{args.client_memeory_size[rnd]}")
    ##若超过客户端能存储的最大size，则将值设置为最大size
    if any(x > args.client_max_cache_size for x in args.client_memeory_size[rnd]):
        args.client_memeory_size[rnd] = [min(x, args.client_max_cache_size) for x in args.client_memeory_size[rnd]]
        ##排除已达到最大size的客户端，剩余的size重新加权
        # if args.client_max_cache_size in args.client_memeory_size and sum(
        #         args.client_memeory_size) != args.server_all_cache_size:
        print(f"-----------------------------------------")
        print(f"超过最大cache,重加权")
        all_indices = [index for index, value in enumerate(args.client_memeory_size[rnd])]
        max_indices = [index for index, value in enumerate(args.client_memeory_size[rnd]) if
                       value == args.client_max_cache_size]
        task_server_all_cache_size = task_server_all_cache_size - len(
            max_indices) * args.client_max_cache_size
        a = 0
        while True:

            # 获取需要重新加权的索引
            need_reweight_indices = [index for index in all_indices if index not in max_indices]

            if not need_reweight_indices:
                break  # 如果没有需要重新加权的索引，退出循环

            # 获取需要加权的值并计算总和
            values = [args.client_memeory_size[rnd][idx] for idx in need_reweight_indices]
            values_sum = sum(values)

            # 重新加权值
            values = [int(round((x / values_sum) * task_server_all_cache_size)) for x in values]

            # 更新列表
            for index, new_value in zip(need_reweight_indices, values):
                args.client_memeory_size[rnd][index] = new_value
            a = a + 1
            print(f"第{a}次重加权结果：{args.client_memeory_size[rnd]}")
            # 限制最大缓存大小
            args.client_memeory_size[rnd] = [min(x, args.client_max_cache_size) for x in
                                             args.client_memeory_size[rnd]]

            # 更新 max_indices 和检查是否继续加权
            new_max_indices = [index for index, value in enumerate(args.client_memeory_size[rnd])
                               if value == args.client_max_cache_size]
            is_go_on_reweight = [item for item in new_max_indices if item not in max_indices]

            # 如果没有需要重新加权的索引，退出循环
            if not is_go_on_reweight:
                break

            # 更新服务器缓存大小
            # 计算将要减少的值
            to_reduce = len(is_go_on_reweight) * args.client_max_cache_size

            # 确保 task_server_all_cache_size 在减少后仍大于 0
            if task_server_all_cache_size > to_reduce:
                task_server_all_cache_size -= to_reduce
            else:
                print(f"Warning: no cache size to reduce")
            # task_server_all_cache_size -= len(new_max_indices) * args.client_max_cache_size
            max_indices = new_max_indices  # 更新 max_indices 以便下一次检查

####按照类别计算动态cache
def cal_class_cache(netglobals,w_locals,args,user_groups,trainset,rnd,task):
    weight_diff = {}
    combined = {}
    client_class_cache = {}
    task_server_all_cache_size = args.server_all_cache_size
    if task==2:
        task_server_all_cache_size+=args.cached_sample_size
    if task ==1 or task==3:
       client_length= args.num_users-1
    else:
        client_length = args.num_users
    ####使用参数直接求差
    for client_id in range(client_length):
        l2_norm = 0.0
        for key in netglobals[client_id].state_dict().keys():
            if key in w_locals[client_id].state_dict().keys():
                net_param = netglobals[client_id].state_dict()[key].to(args.device)
                local_param = w_locals[client_id].state_dict()[key].to(args.device)
                diff = net_param.float() - local_param.float()
                l2_norm += torch.norm(diff).item() ** 2  # L2 范数的平方
        weight_diff[client_id] = l2_norm ** 0.5
    print(f"client_diff:{weight_diff}")

    for key in weight_diff:
        combined[key] = weight_diff[key]
    # 计算总和
    total_sum = sum(combined.values())
    # 归一化各客户端权重
    normalized = {key: value / total_sum for key, value in combined.items()}
    #####使用cos
    # similarities, normalized_similarities=calculate_cosine_similarity(netglobals[0], w_locals,args)
    # total_similarity = sum(normalized_similarities)
    # storage_proportions = [(sim / total_similarity) for sim in normalized_similarities]
    # args.client_memeory_size[rnd] = [value * task_server_all_cache_size for value in
    #                                  storage_proportions]
    args.client_memeory_size[rnd] = [value * task_server_all_cache_size for value in
                                     normalized.values()]
    args.client_memeory_size[rnd] = [math.floor(x) for x in args.client_memeory_size[rnd]]
    print(f"----------------------------------------------")
    print(f"rnd:{rnd},client_cache_size:{args.client_memeory_size[rnd]}")
    ##若超过客户端能存储的最大size，则将值设置为最大size
    if any(x > args.client_max_cache_size for x in args.client_memeory_size[rnd]):
        args.client_memeory_size[rnd] = [min(x, args.client_max_cache_size) for x in args.client_memeory_size[rnd]]
        ##排除已达到最大size的客户端，剩余的size重新加权
        # if args.client_max_cache_size in args.client_memeory_size and sum(
        #         args.client_memeory_size) != args.server_all_cache_size:
        print(f"-----------------------------------------")
        print(f"超过最大cache,重加权")
        all_indices = [index for index, value in enumerate(args.client_memeory_size[rnd])]
        max_indices = [index for index, value in enumerate(args.client_memeory_size[rnd]) if
                       value == args.client_max_cache_size]
        task_server_all_cache_size = task_server_all_cache_size - len(
            max_indices) * args.client_max_cache_size
        a = 0
        while True:

            # 获取需要重新加权的索引
            need_reweight_indices = [index for index in all_indices if index not in max_indices]

            if not need_reweight_indices:
                break  # 如果没有需要重新加权的索引，退出循环

            # 获取需要加权的值并计算总和
            values = [args.client_memeory_size[rnd][idx] for idx in need_reweight_indices]
            values_sum = sum(values)

            # 重新加权值
            values = [int(round((x / values_sum) * task_server_all_cache_size)) for x in values]

            # 更新列表
            for index, new_value in zip(need_reweight_indices, values):
                args.client_memeory_size[rnd][index] = new_value
            a = a + 1
            print(f"第{a}次重加权结果：{args.client_memeory_size[rnd]}")
            # 限制最大缓存大小
            args.client_memeory_size[rnd] = [min(x, args.client_max_cache_size) for x in
                                             args.client_memeory_size[rnd]]

            # 更新 max_indices 和检查是否继续加权
            new_max_indices = [index for index, value in enumerate(args.client_memeory_size[rnd])
                               if value == args.client_max_cache_size]
            is_go_on_reweight = [item for item in new_max_indices if item not in max_indices]

            # 如果没有需要重新加权的索引，退出循环
            if not is_go_on_reweight:
                break

            # 更新服务器缓存大小
            # 计算将要减少的值
            to_reduce = len(is_go_on_reweight) * args.client_max_cache_size

            # 确保 task_server_all_cache_size 在减少后仍大于 0
            if task_server_all_cache_size > to_reduce:
                task_server_all_cache_size -= to_reduce
            else:
                print(f"Warning: no cache size to reduce")
            # task_server_all_cache_size -= len(new_max_indices) * args.client_max_cache_size
            max_indices = new_max_indices  # 更新 max_indices 以便下一次检查
    # 为每个客户端的每个类别动态计算cache size
    if args.use_dynamic_class_memory:
        for client_id in range(len(w_locals)):
            # print(f"client:{client_id}")
            # print(f"args.client_memeory_size:{len(args.client_memeory_size)}")
            # print(f"args.client_memeory_size[rnd]:{args.client_memeory_size[rnd]}")
            if (task ==1 and client_id==5) or (task==3 and client_id==6):
                client_cache=args.cached_sample_size
                args.client_memeory_size[rnd].append(client_cache)
            else:
                print(f"task:{task},client_id:{client_id}")
                print(f"client_memeory_size:{args.client_memeory_size[rnd]}")
                client_cache = args.client_memeory_size[rnd][client_id]#####当前客户端被分配到的cache
            # print(f"------------------------------------------------------------------------")
            # print(f"当前客户端分配的cache:{client_cache}")
            class_ids = [trainset.targets[i] for i in user_groups[client_id]]
            # print(f"当前客户端的类别分布：{class_ids}")
            class_num = Counter(class_ids)  # 当前客户端每个类的个数
            # print(f"当前客户端每个类别样本个数：{class_num}")
            class_count = {class_id: trainset.targets.count(class_id) for class_id in set(class_ids)}  # 每个类别多少个样本
            # print(f"全局样本类别个数：{class_count}")
            # 计算总和
            total = sum(class_num.values())
            client_class_ratio = {k: v / total for k, v in class_num.items()}
            # print(f"当前客户端的每个类别占比：{client_class_ratio}")
            global_class_ratio = {key: class_num[key] / class_count[key] for key in class_count}
            # print(f"当前客户端每个类别占全局样本占比：{global_class_ratio}")
            # 计算结果
            result_ratio = {key: (1-args.data_dist_weight)  * client_class_ratio[key] + args.data_dist_weight * global_class_ratio[key] for key in
                            client_class_ratio}
            # print(f"当前客户端每个类别的综合占比：{result_ratio}")
            total = sum(result_ratio.values())
            # 归一化每个值
            normalized_result = {key: value / total for key, value in result_ratio.items()}
            # print(f"归一化后占比：{normalized_result}")
            result = {key: client_cache * normalized_result[key] for key in normalized_result}
            # print(f"当前客户端每个类别size:{result}")
            client_class_cache[client_id] = result

        args.client_class_memory_size[rnd] = client_class_cache

###计算每个客户端与全局的cos相似度
def calculate_cosine_similarity(global_model, client_models,args):
    # 获取全局模型的参数
    global_params = torch.cat([param.data.view(-1) for param in global_model.parameters()]).to(args.device)

    similarities = []

    for id in range(len(client_models)):
        # 获取客户端模型的参数
        client_params = torch.cat([param.data.view(-1) for param in client_models[id].parameters()]).to(args.device)

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(global_params.unsqueeze(0), client_params.unsqueeze(0))
        similarities.append((1-cosine_similarity.item()))

    # 归一化相似度值
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    normalized_similarities = [(sim - min_similarity) / (max_similarity - min_similarity) for sim in similarities]
    print("Cosine Similarities:", similarities)
    print("Normalized Similarities:", normalized_similarities)
    return similarities, normalized_similarities


def cal_avg_class_cache_size(args):
    # 初始化一个字典来存储总和
    total_cache_size = {}

    # 遍历每个 rnd
    for rnd in range(args.rounds):
        if rnd not in args.client_class_memory_size:
            continue  # 如果没有这个 rnd，则跳过
        for client_id, class_sizes in args.client_class_memory_size[rnd].items():
            if client_id not in total_cache_size:
                total_cache_size[client_id] = {}

            for class_id, size in class_sizes.items():
                if class_id not in total_cache_size[client_id]:
                    total_cache_size[client_id][class_id] = 0

                total_cache_size[client_id][class_id] += size

    # 计算平均值
    average_cache_size = {}
    for client_id, class_sizes in total_cache_size.items():
        average_cache_size[client_id] = {}
        for class_id, total_size in class_sizes.items():
            average_cache_size[client_id][class_id] = math.ceil(total_size / args.rounds)
    args.avg_class_cache_size=average_cache_size


###计算当前任务下各客户端cache size, 初始任务每个客户端直接存满，增量任务按比例计算
def new_cal_client_cache(args,client_models,netglobals,task,rnd):
    current_train=args.current_trainset
    current_user_groups=args.current_user_groups
    num_clients=[]
    weight_diff=[]
    for client_id in range(args.num_users):
        num_client=len(list(current_train.targets[id] for id in current_user_groups[client_id]))#计算客户端样本占比
        num_clients.append(num_client)
        l2_norm = 0.0
        for key in netglobals[client_id].state_dict().keys():
            if key in client_models[client_id].keys():
                net_param = netglobals[client_id].state_dict()[key].to(args.device)
                local_param = client_models[client_id][key].to(args.device)
                diff = net_param.float() - local_param.float()
                l2_norm += torch.norm(diff).item() ** 2  # L2 范数的平方
        weight_diff.append(l2_norm ** 0.5)

    total_sample_num = sum(num_clients)
    total_weight=sum(weight_diff)
    client_sample_ratio=[x/total_sample_num for x in num_clients]
    client_weight_ratio=[x/total_weight for x in weight_diff]
    if task ==0:
        task_cache_size=args.server_all_cache_size
    else:
        task_cache_size=args.task_cache_size[task]
    ration_client=[(args.a*client_sample_ratio[i]+(1-args.a)*client_weight_ratio[i]) for i in range(args.num_users)]
    ration_client_sum=sum(ration_client)
    ration_client=[(x/ration_client_sum)*task_cache_size for x in ration_client]
    args.client_memeory_size[rnd]=ration_client


####由于初始任务已经将cache存满，因此需要为新任务数据腾位置
def reduce_client_cache(args):
    for client_id in range(args.num_users):
        if client_id not in args.avg_client_memeory_size:
            continue
        else:
            m = args.avg_client_memeory_size[client_id]
            leng = len(list(args.cache_client_samples[client_id].data))
            if leng > args.init_client_size:
                print(f"为客户端：{client_id}，原始存储的样本个数为：{leng}  腾出：{m}个位置")
                args.cache_client_samples[client_id].data[:(args.init_client_size - m)]
                args.cache_client_samples[client_id].targets[:(args.init_client_size - m)]
            args.init_client_size -= m

#####UACL-uncertainty_estimation to choose example data
def epistemic_uncertainty_estimation(model, images, T, num_classes):
    # 初始化结果张量
    R = torch.zeros((T, len(images), num_classes))
    for t in range(T):
        # 在模型的第二层和第三层之间启用Dropout，需要在模型里加入对应的dropout层
        model.train()
        outputs = model(images)
        # 获取概率值
        probs = torch.softmax(outputs, dim=1)
        for i, prob in enumerate(probs):
            R[t, i, :] = prob

    # 计算均值
    mu = torch.mean(R, dim=0)
    # 计算不确定性
    squared_diff = (R - mu.unsqueeze(0)) ** 2
    sum_squared_diff = torch.sum(squared_diff, dim=0)
    U = torch.sum(torch.sqrt(sum_squared_diff / T), dim=1)
    return U





