# This file is borrowed from https://github.com/Xu-Jingyi/FedCorr/blob/main/util/sampling.py
import copy
import logging
import os

import numpy as np

# 独立同分布采样
def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    # 每个客户端分配多少个样本
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        #在总样本集种随机选择num_items个分配给各客户端
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])
    # 转换为列表
    for key in dict_users.keys():
        dict_users[key] = list(dict_users[key])
    return dict_users

# 非独立同分布采样
def non_iid_dirichlet_sampling(current_task,y_train, num_classes, p, num_users, seed, alpha_dirichlet,args):
    np.random.seed(seed)
    # 使用NumPy库中的随机模块生成二项分布样本的函数调用。Phi为二维数组，其中行为客户端，列为类别，即20个客户端，5个类，每个元素为0或者1，1代表客户端选择了该类，0代表客户端未选择该类。
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client  指明每个客户端选择的类别
    # 按第一个维度相加，即按行（客户端）相加可得到一个客户端有几个类
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class 指出选择每个类别的客户端
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    # 将每个类别的样本按概率分配给各客户端
    for class_i in range(num_classes):
        y_train = np.array(y_train)
        if current_task>0:
            inc_class_i=class_i + current_task*num_classes+args.way
            all_idxs = np.where(y_train == inc_class_i)[0]
        else:
            all_idxs = np.where(y_train == class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
    
    # for key in dict_users.keys():
    #     dict_users[key] = list(dict_users[key])
    return dict_users

# new 非独立同分布采样，即每个客户端分配不同的样本个数
def cl_non_iid_sampling(y_train, p, num_clients, seed, alpha_dirichlet):
    np.random.seed(seed)
    # 10X1的二维数组，表示10个客户端，在当前任务
    Phi = np.random.binomial(1, p, size=(num_clients, 1))


def cl_iid_sampling(n_train, num_clients, seed, session,train_dataset):
    np.random.seed(seed)
    # 每个客户端分配多少个样本
    num_client_items = int(n_train / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(n_train)]  # initial user and index for current dataset
    dict_per_client=[]
    for i in range(num_clients):
        if i == num_clients-1:
            dict_clients[i]=set(all_idxs)
        else:
            # 在总样本集种随机选择num_items个分配给各客户端
            dict_clients[i] = set(
                np.random.choice(all_idxs, num_client_items,
                                 replace=False))  # 'replace=False' make sure that there is no repeat
            all_idxs = list(set(all_idxs) - dict_clients[i])  # 更新待选集合
    for idx in range(num_clients):
        images_list=[]
        lable_list=[]
        target_list=[]
        for id in list(dict_clients[idx]):
            images_id=train_dataset.images[id]
            lable=train_dataset.labels[id]
            target=train_dataset.targets[id]
            images_list.append(images_id)
            lable_list.append(lable)
            target_list.append(target)
        dict_per_client[idx].images=images_list
        dict_per_client[idx].labels=lable_list
        dict_per_client[idx].targets=target_list

    # 转换为列表
    for key in dict_clients.keys():
        dict_per_client[key].images = list(dict_per_client[key].images)
        dict_per_client[key].labels = list(dict_per_client[key].labels)
        dict_per_client[key].targets = list(dict_per_client[key].targets)
        logging.info(f"session:{session},  客户端索引：{key}, 样本个数：{len(dict_clients[key])},样本：{dict_per_client[key].images}")
    return dict_clients,dict_per_client


def get_client_index(n_train, num_clients, self):
    #校验是否已分配各客户端当前任务
    path="/data/wtt/demo/own/FL_CL/sample"
    txt_name=f"sample_client_index_{self.session}.txt"
    txt_path=os.path.join(path,txt_name)
    dict_clients = {}
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                key=int(key)
                value=set(eval(value))
                dict_clients[key] = value
    else:
        np.random.seed(self.args.seed)
        # 每个客户端分配多少个样本
        num_client_items = int(n_train / num_clients)
        all_idxs = [i for i in range(n_train)]
        for i in range(num_clients):
            if i == num_clients - 1:
                dict_clients[i] = set(all_idxs)
            else:
                # 在总样本集种随机选择num_items个分配给各客户端
                dict_clients[i] = set(
                    np.random.choice(all_idxs, num_client_items,
                                     replace=False))  # 'replace=False' make sure that there is no repeat
                all_idxs = list(set(all_idxs) - dict_clients[i])  # 更新待选集合
        if not os.path.exists(path):
            os.mkdir(path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for key, value in dict_clients.items():
                f.write(f"{key}\t{value}"+"\n")

    for key in dict_clients.keys():
        dict_clients[key]=list(dict_clients[key])
    return dict_clients
