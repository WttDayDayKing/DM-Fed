import copy
import torch
import numpy as np
import math

def FedAvg(w, dict_len,args,rnd):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    # if args.use_dynamic_memory:
    #    cal_client_memory_size(w_avg,w,args,rnd)
    return w_avg

# 距离感知融合函数，为干净客户端和噪声客户端计算不同的融合权重
def DaAgg(w, dict_len, clean_clients, noisy_clients):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()
    distance = np.zeros(len(dict_len))
    for n_idx in noisy_clients:
        dis = []
        for c_idx in clean_clients:
            dis.append(model_dist(w[n_idx], w[c_idx]))
        distance[n_idx] = min(dis)
    distance = distance / distance.max()
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    # print(client_weight)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()


def cal_client_memory_size(w_avg, w,args,rnd):
    task_server_all_cache_size=args.server_all_cache_size
    ###直接利用剧局部模型参数和全局模型参数的L2
    for idx in range(len(w)):
        l1_norm = 0.0
        # for param1, param2 in zip(w[idx].parameters(),w_avg[idx].parameters()):
        #     l1_norm += torch.sum(torch.abs(param1.data - param2.data)).item()  # 计算 L1 范数
        l2_norm=compute_l2_norm(w_avg,w[idx])
        print(f"l2_norm:{l2_norm}")
        # n =  l2_norm + (
        #         args.client_data_size[idx] / args.all_data_size)
        n=l2_norm
        args.client_memeory_size[rnd].append(n)
    _sum = sum(args.client_memeory_size[rnd])
    args.client_memeory_size[rnd] = [(x / _sum) * task_server_all_cache_size for x in
                                     args.client_memeory_size[rnd]]
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
        a=0
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
            #task_server_all_cache_size -= len(new_max_indices) * args.client_max_cache_size
            max_indices = new_max_indices  # 更新 max_indices 以便下一次检查


def compute_cos(model1_state_dict, model2_state_dict):
    # 初始化 L2 范数

    cosine_similarity=0.0
    for name, param in model1_state_dict.items():
        if param.requires_grad and name in model2_state_dict:
            # # 获取参数
            # param1 = model1_state_dict[name]
            # param2 = model2_state_dict[name]
            # l2_norm += torch.norm(param1 - param2, p=1).item()

            # 获取参数
            param1 = model1_state_dict[name].view(-1)  # 展平为一维张量
            param2 = model2_state_dict[name].view(-1)  # 展平为一维张量

            # 计算余弦相似度
            dot_product = torch.dot(param1, param2)  # 点积
            norm_param1 = torch.norm(param1, p=2)  # param1 的 L2 范数
            norm_param2 = torch.norm(param2, p=2)  # param2 的 L2 范数

            if norm_param1.item() > 0 and norm_param2.item() > 0:  # 避免除以零
                cosine_similarity += dot_product / (norm_param1 * norm_param2)
    return cosine_similarity

def compute_l2_norm(model1_state_dict, model2_state_dict):
    l2_norm = 0.0
    for name, param in model1_state_dict.items():
        if name in model2_state_dict:
            param1 = model1_state_dict[name]
            param2 = model2_state_dict[name]
            l2_norm += torch.norm(param1 - param2, p=2).item()

    return l2_norm
###计算本地模型与其他客户端模型的相似度，相似度大的给与较小的权重
def cal_cos_sim(models_list_global,models_list_clients,args,rnd):
    params=[]
    cos_sim_clients={}
    task_server_all_cache_size = args.server_all_cache_size
    for id in range(len(models_list_clients)):
        for name,param in models_list_clients[id].named_parameters():
            if param.requires_grad:
                params.append(name)
    # 打印参数名称以调试
    #print("Collected parameter names:", params)
    for id in range(len(models_list_clients)):
        site_grad = []
        global_grad=[]
        # 打印客户端模型的 state_dict
        #print(f"Client {id} state_dict keys:", models_list_clients[id].state_dict().keys())
        for name in params:
            try:
               # 访问客户端和全局模型的参数
              site_param = models_list_clients[id].state_dict()[name]
              global_param = models_list_global[id].state_dict()[name]

              site_grad.append(torch.as_tensor(site_param).data.view(-1))
              global_grad.append(torch.as_tensor(global_param).data.view(-1))
            except:
                print(f"KeyError: Parameter '{name}' not found in model {id}.")
        site_grads_vec = torch.cat(site_grad).to("cpu")
        global_grad_vec=torch.cat(global_grad).to("cpu")
        # minus gradient
        minus_grads_vec = global_grad_vec - 0.2 * site_grads_vec
        fedce_cos_sim_site = (
            torch.cosine_similarity(site_grads_vec, minus_grads_vec, dim=0).detach().cpu().numpy().item()
        )
        cos_sim_client=1-fedce_cos_sim_site
        args.client_memeory_size[rnd].append(cos_sim_client)
    _sum = sum(args.client_memeory_size[rnd])
    args.client_memeory_size[rnd] = [(x / _sum) * task_server_all_cache_size for x in
                                     args.client_memeory_size[rnd]]
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

    #return torch.sqrt(torch.tensor(l2_norm))
    # # 遍历两个模型的参数
    # for key in model1_state_dict.keys():
    #     if param.requires_grad:
    #     # 确保两个模型有相同的参数
    #     if key in model2_state_dict:
    #         # 获取参数
    #         param1 = model1_state_dict[key]
    #         param2 = model2_state_dict[key]
    #
    #         # 计算参数的 L2 范数并累加
    #         l2_norm += torch.norm(param1 - param2, p=1).item()
    #
    # # 返回 L2 范数的平方根
    # return torch.sqrt(torch.tensor(l2_norm))



