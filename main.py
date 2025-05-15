import math
import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
import torch.backends.cudnn as cudnn

from utils.options import args_parser
from utils.local_training import globaltest,train_base,localtest
from utils.FedAvg import FedAvg, DaAgg,cal_cos_sim
from utils.FedDWA import FedDWA
from utils.util import add_noise, set_seed, set_output_files, get_output, get_current_consistency_weight
from utils.compute_SSIM import compute_SSIM
from dataset.data_util import get_client_dict_dataset,cal_client_importent_score,cal_class_cache,cal_avg_class_cache_size

from model.build_model import build_model
from model.local_update import LocalUpdate
from model.global_updateold import globaltesteda
from utils.util import average_models
from PIL import Image
from torchvision import models
from torchvision import transforms
from dataset.data import CL_dataset
from clients.DYCFCL import DYCFCL
np.set_printoptions(threshold=np.inf)
import glob
"""
Major framework of noise FL
"""
##读取txt文件的工具
# def read_txt(file_path):
#     result = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#         for line in lines:
#             if ("******** round:") in line:
#                 result.append(line)
#     return result



if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.learned_numclass = {idx: 0 for idx in range(args.num_users)}
    args.feature_extractor = models.resnet18(pretrained=False).to(args.device)
    args.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomAffine(degrees=10,translate=(0.02,0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
            ])
    args.learned_classes={idx:[] for idx in range(args.num_users)}
    print(torch.cuda.is_available())  # 应该返回 True
    print(torch.cuda.device_count())  # 应该返回你的 GPU 数量
    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # --------------------- Build Models ---------------------------
    #获取数据集
    dataset = CL_dataset(args)
    ds = dataset.train_dataset
    #初始化模型
    netglob = build_model(args)#ResNet18的模型
    pim_model = netglob  ###客户端信息化模型模型
    student_model=pim_model ###为新客户端提供
    netglobs = {}
    user_id = list(range(args.num_users))
    for id in user_id:
        netglobs[id] = netglob
    trainer_locals = []
    clients=[]
    #构建多个客户端
    for i in range(7):
        group = dataset.groups[i]
        client_id = i
        if args.method == 'FCLPF':
            client = DYCFCL(args.batch_size, args.local_ep, ds, group, args.dataset_name, client_id=client_id, device='cuda')
        clients.append(client)
    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    args.local_train_old={}
    args.cache_client_samples={}
    args.avg_client_memeory_size = [args.cached_sample_size]*args.num_users
    pim_model_list = {idx: pim_model for idx in range(args.num_users)}
    for task in range(args.task_num):
        if task == 1 or task == 3:
            args.num_users+=1
        if task ==0:
            args.num_class=4
        else:
            args.num_class=3
        # ------------------------------ output files ------------------------------
        writer, models_dir,logs_dir = set_output_files(args,task)
        best_performance = 0.
        best_performance_list={i:0 for i in range(args.num_users)}
        BACC=[]
        BACD={i:[] for i in range(args.num_users)}
        ####出现新客户端时，初始化模型
        if len(netglobs)<args.num_users:
            teacher_model=build_model(args)
            teacher_model.load_state_dict(netglob)
            netglobs[args.num_users-1]=student_model
            pim_model_list[args.num_users-1]=pim_model
        print(f"task:{task},client memory size:{args.avg_client_memeory_size}")
        user_groups, train_set, test_set = get_client_dict_dataset(args,task,pim_model_list,netglobs)
        args.client_memeory_size = {rnd: [] for rnd in range(args.rounds)}
        args.client_class_memory_size={rnd:{} for rnd in range(args.rounds)}
        args.client_avg_acc={idx:[] for idx in range(args.num_users)}
        args.global_avg_acc={idx:[] for idx in range(args.rounds)}
        global_acc_list=[]
        for rnd in range(args.rounds):
            w_locals = {}
            w_next_step_locals = {}
            w_glob_array = []
            for client in range(args.num_users):
                map_dict = {}
                for (key, value), new_key in zip(user_groups.items(), range(args.num_users)):
                    map_dict[new_key] = value
                user_groups = map_dict
                train_client_indx=user_groups[client]
                #w_locals[client]=netglobs[client]
                w_locals[client]=copy.deepcopy(netglobs[client]).to(args.device)
                local=LocalUpdate(args,train_set,train_client_indx)
                if task==1 or task==3:
                    if client == args.num_users - 1:  #
                        w_local, next_step_model = local.new_client_train(
                            model1=copy.deepcopy(teacher_model).to(args.device),
                            model2=copy.deepcopy(student_model).to(args.device))
                    else:
                        w_local, loss_local, next_step_model = local.train(
                            model=copy.deepcopy(netglobs[client]).to(args.device), writer=writer, client_id=client)

                else:
                    w_local, loss_local, next_step_model = local.train(
                        model=copy.deepcopy(netglobs[client]).to(args.device), writer=writer, client_id=client)
                new_state_dict = {}
                for k, v in w_local.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
                    else:
                        new_state_dict[k] = v
                w_local = new_state_dict
                w_locals[client].load_state_dict(copy.deepcopy(w_local))
                #w_next_step_locals[client]=copy.deepcopy(next_step_model)
                w_glob_array.append(w_local)
                #cal_client_importent_score(netglobals=netglobs,w_locals=w_locals,args=args,user_groups=user_groups,trainset=train_set,rnd=rnd)
                if args.use_dynamic_memory:
                     cal_class_cache(netglobals=netglobs,w_locals=w_locals,args=args,user_groups=user_groups,trainset=train_set,rnd=rnd)
            print("-------------------------------------------------------------------------------------------------")
            # cal_client_importent_score(netglobals=netglobs, w_locals=w_locals, args=args,
            #                                user_groups=user_groups, trainset=train_set, rnd=rnd)
            w_locals_last = copy.deepcopy(w_glob_array)

            ### 使用 FedAvg融合
            if args.combine=='FedAvg':
                dict_len = [len(user_groups[idx]) for idx in range(args.num_users)]
                w_glob_fl = FedAvg(w_glob_array, dict_len,args,rnd)
                # new_state_dict = {}
                # for k, v in w_glob_fl.items():
                #     if k.startswith('module.'):
                #         new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
                #     else:
                #         new_state_dict[k] = v
                # netglob = new_state_dict
                netglob=w_glob_fl
                for client in range(args.num_users):
                    netglobs[client].load_state_dict(copy.deepcopy(w_glob_fl))
                #cal_cos_sim(netglobs,w_locals,args,rnd)
                pred = globaltest(copy.deepcopy(netglobs[0]).to(args.device), test_set, args)
                global_acc = accuracy_score(test_set.targets, pred)###测试全局模型
                global_acc_list.append(global_acc)
                # local_acc_list={}
                # for client_id in range(args.num_users):
                #     pred = localtest(copy.deepcopy(w_locals[client_id]).to(args.device), test_set, args)
                #     local_acc = accuracy_score(test_set.targets, pred)###测试局部模型
                #     local_acc_list[client_id]=local_acc
                print("******** session: %d, round: %d, global_acc: %.4f" % (task, rnd, global_acc))

                writer.add_scalar(f'test/global_acc', global_acc, rnd)
                # for idx in range(args.num_users):
                #     lo=local_acc_list[idx]
                #     print("******** session:%d,round:%d,client:%d,client_acc:%.4f" % (task, rnd, idx, lo))
                #     writer.add_scalar(f'test/local_acc', local_acc_list[idx], rnd)
                #     args.client_avg_acc[idx].append(lo)
                # save model
                if global_acc > best_performance:
                    best_performance = global_acc
                    torch.save(netglob,
                                   models_dir + f'/fedavg_session_{task}_model.pth')  # 保持最好一个round的模型
                    print(f'best acc: {best_performance}，round:{rnd}')
                BACC.append(global_acc)
                print(BACC)
            elif args.combine=='FedDWA':
                server = FedDWA(args,w_locals=w_locals,w_more_locals=w_next_step_locals)
                server_netglobs = server.feddwa(rnd)
                local_acc_list = {}
                global_acc_list = {}
                for idx in range(args.num_users):
                    test_dlobal = server_netglobs.send_client_models[idx]
                    print(f"global_paramter_item:{test_dlobal}")
                    stat_dict = netglobs[idx].state_dict()
                    print(f"state_dict:{stat_dict}")
                    for k, v in test_dlobal.items():
                        stat_dict[k] = v
                    netglobs[idx].load_state_dict(stat_dict)
                    #netglobs[idx].load_state_dict(copy.deepcopy(test_dlobal))
                    preds_golbal = globaltesteda(copy.deepcopy(netglobs[idx]).to(args.device), test_set, args)
                    #test_local = w_locals[idx]
                    #netglobs[idx].load_state_dict(copy.deepcopy(test_local))
                    preds_client = localtest(copy.deepcopy(w_locals[idx]).to(args.device), test_set, args)
                    local_acc = accuracy_score(test_set.targets, preds_client)  ###测试局部模型
                    local_acc_list[idx] = local_acc
                    global_acc = accuracy_score(test_set.targets, preds_golbal)  ###测试全局模型
                    global_acc_list[idx] = global_acc
                for idx in range(args.num_users):
                    lo = local_acc_list[idx]
                    go=global_acc_list[idx]
                    args.client_avg_acc[idx].append(lo)
                    args.global_avg_acc[idx].append(go)
                    print("******** session: {}, round: {}, global_client: {}, global_acc: {:.4f}".format(task, rnd, idx,global_acc_list[idx]))
                    print("******** session: {}, round: {}, client: {}, client_acc: {:.4f}".format(task, rnd, idx, local_acc_list[idx]))
                    writer.add_scalar(f'test/local_acc', local_acc_list[idx], rnd)
                    writer.add_scalar(f'test/global_acc', global_acc_list[idx], rnd)
                for idx in range(args.num_users):
                    if global_acc_list[idx] > best_performance_list[idx]:
                        best_performance_list[idx]=global_acc_list[idx]
                        torch.save(netglobs[idx].state_dict(),models_dir+f'/dwa_client_{idx}_session_{task}_model.pth')
                        logging.info(f'client:{idx}, best acc: {best_performance_list[idx]}，round:{rnd}')
                        BACD[idx].append(global_acc_list[idx])
                        logging.info(BACD[idx])
        if args.use_dynamic_memory:
            # 初始化一个列表，用于存储每个round下客户端的cache size的总和
            sum_values = [0] * args.num_users
            count = 0
            # 遍历字典，求和
            for values in args.client_memeory_size.values():
                print(f"values:{values}")
                sum_values = [sum_values[i] + values[i] for i in range(args.num_users)]
                count += 1
            # 计算平均
            average_values = [s / count for s in sum_values]
            args.avg_client_memeory_size = [math.floor(x) for x in average_values]
        if args.client_class_memory_size:
            cal_avg_class_cache_size(args)

        # for idcd in range(args.rounds):
        #     do_array_acc=args.global_avg_acc[idcd]
        #     print(f"task:{task},clien:{idcd},client_avg_rouns_acc:{np.mean(array_acc)}")
        #     print(f"task:{task},client:{idcd},global_avg_rouns_acc:{np.mean(do_array_acc)}")
        average_accuracy = sum(global_acc_list) / len(global_acc_list)
        print("当前平均精度:", average_accuracy)
    torch.cuda.empty_cache()
