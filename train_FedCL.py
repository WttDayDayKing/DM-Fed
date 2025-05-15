import math
import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter
from utils import Dataloader_old
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader,TensorDataset
from dataset.data_preprocess import data_preprocess
from utils.options import args_parser
from utils.local_training import globaltest,train_base,localtest
from utils.FedAvg import FedAvg, DaAgg,cal_cos_sim
from utils.FedDWA import FedDWA
from utils.util import add_noise, set_seed, set_output_files, get_output, get_current_consistency_weight
from utils.compute_SSIM import compute_SSIM
from dataset.data_util import get_client_dict_dataset,cal_class_cache,cal_avg_class_cache_size
from dataset.dataset import get_dataset
from dataset.cl_dataset import get_cl_dataset,update_train_example_data_set
from model.build_model import build_model,build_model_new,CustomResNet18
from model.local_update import LocalUpdate
from model.global_updateold import globaltesteda
from model.all_models import modify_last_layer,Incremental_learning
from model.ResNet import ResNet18
from utils.util import average_models
from PIL import Image
from torchvision import models
from torchvision import transforms
from dataset.Matek19 import matek19
from dataset.Acevedo20 import acevedo20
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
    args.next_step_model = None
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.learned_numclass = {idx: 0 for idx in range(args.num_users)}
    args.known_class={idx:0 for idx in range(7)}
    args.feature_extractor = ResNet18(args.n_classes,False).to(args.device)
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
    ### 初始化全局模型
    ### 为每个客户端维护信息模型
    #netglob = CustomResNet18(num_classes=args.n_classes)
    netglob = build_model(args)
    ###为新客户端提供
    netglobs={}
    pim_model_list={}
    user_id = list(range(args.num_users))
    for id in user_id:
        netglobs[id]=netglob
        pim_model = build_model(args)
        pim_model_list[id]=pim_model
    trainer_locals = []

    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    args.local_train_old={}
    args.cache_client_samples={}
    args.class_cache={client:{} for client in range(args.num_users)}
    args.class_cache_num={client:{} for client in range(args.num_users)}
    args.avg_client_memeory_size = [args.cached_sample_size]*args.num_users
    #pim_model_list = {idx: pim_model for idx in range(args.num_users)}
    for task in range(args.task_num):
        if task == 1 or task == 3:
            args.num_users+=1
        if task ==0:
            args.num_class=args.base_class
        else:
            args.num_class=args.way

        # ------------------------------ output files ------------------------------
        writer, models_dir,logs_dir = set_output_files(args,task)
        best_performance = 0.
        best_performance_list={i:0 for i in range(args.num_users)}
        BACC=[]
        BACD={i:[] for i in range(args.num_users)}
        ####出现新客户端时，初始化模型
        if len(netglobs)<args.num_users:
            #averaged_model=average_models(netglobs)
            averaged_model=copy.deepcopy(netglobs[0].state_dict())
            #teacher_model=copy.deepcopy(averaged_model)
            #args.new_class=args.known_class+args.way
            teacher_model = build_model(args)
            teacher_model.load_state_dict(averaged_model)
            student_model = build_model(args)
            #student_model=build_model(args)
            #netglobs[args.num_users-1]=student_model
            pim_model=build_model(args)
            #pim_model_list[args.num_users-1]=pim_model
            netglobs[args.num_users-1] = student_model
            pim_model_list[args.num_users-1]= pim_model
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
                w_locals[client]=copy.deepcopy(netglobs[client]).to(args.device)
                local=LocalUpdate(args,train_set,train_client_indx)

                if args.use_kd_loss:
                    if task == 1 or task == 3:
                        if client == args.num_users - 1:  #
                            w_local, next_step_model = local.new_client_train(
                                model1=copy.deepcopy(teacher_model).to(args.device),
                                model2=copy.deepcopy(student_model).to(args.device))
                        else:
                            w_local, loss_local, next_step_model = local.train(
                                model=copy.deepcopy(netglobs[client]).to(args.device), writer=writer, client_id=client,
                                task=task)

                    else:
                        w_local, loss_local, next_step_model = local.train(
                            model=copy.deepcopy(netglobs[client]).to(args.device), writer=writer, client_id=client,
                            task=task)
                else:
                    w_local, loss_local, next_step_model = local.train(
                        model=copy.deepcopy(netglobs[client]).to(args.device), writer=writer, client_id=client,
                        task=task)
                new_state_dict = {}
                for k, v in w_local.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
                    else:
                        new_state_dict[k] = v
                w_local = new_state_dict
                w_locals[client].load_state_dict(copy.deepcopy(w_local))
                w_glob_array.append(w_local)
            if args.use_dynamic_memory:
                cal_class_cache(netglobals=netglobs,w_locals=w_locals,args=args,user_groups=user_groups,trainset=train_set,rnd=rnd,task=task)
            print("-------------------------------------------------------------------------------------------------")

            w_locals_last = copy.deepcopy(w_glob_array)

            ### 使用 FedAvg融合
            if args.combine=='FedAvg':
                dict_len = [len(user_groups[idx]) for idx in range(args.num_users)]
                w_glob_fl = FedAvg(w_glob_array, dict_len,args,rnd)

                netglob=w_glob_fl
                for client in range(args.num_users):
                    netglobs[client].load_state_dict(copy.deepcopy(w_glob_fl))
                pred = globaltest(copy.deepcopy(netglobs[0]).to(args.device), test_set, args)
                global_acc = accuracy_score(test_set.targets, pred)###测试全局模型
                global_acc_list.append(global_acc)

                print("******** session: %d, round: %d, global_acc: %.4f" % (task, rnd, global_acc))

                writer.add_scalar(f'test/global_acc', global_acc, rnd)
                if global_acc > best_performance:
                    best_performance = global_acc
                    torch.save(netglob,
                                   models_dir + f'/fedavg_session_{task}_model.pth')  # 保持最好一个round的模型
                    print(f'best acc: {best_performance}，round:{rnd}')
                BACC.append(global_acc)
                print(BACC)

        if args.use_dynamic_memory and task<args.task_num-1:
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
        if args.client_class_memory_size and task<args.task_num-1:
            cal_avg_class_cache_size(args)
        print(f"task:{task},client memory size:{args.avg_client_memeory_size}")
        print(f"task:{task},client class cache:{args.avg_class_cache_size}")
        # for idcd in range(args.rounds):
        #     do_array_acc=args.global_avg_acc[idcd]
        #     print(f"task:{task},clien:{idcd},client_avg_rouns_acc:{np.mean(array_acc)}")
        #     print(f"task:{task},client:{idcd},global_avg_rouns_acc:{np.mean(do_array_acc)}")
        average_accuracy = sum(global_acc_list) / len(global_acc_list)
        print("当前平均精度:", average_accuracy)
        #args.known_class+=args.num_class
        # netglobs[0] = Incremental_learning(netglobs[0], args.known_class + args.way)
        # for client in range(args.num_users):
        #     netglobs[client] = netglobs[0]
        #     pim_model_list[client] = Incremental_learning(pim_model_list[client],
        #                                                   args.known_class + args.way)
    torch.cuda.empty_cache()
