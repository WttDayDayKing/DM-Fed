import copy
import csv
import random
import os
import sys
import re
import shutil
import numpy as np
import pandas as pd
from collections import OrderedDict
import logging
from tensorboardX import SummaryWriter
from PIL import Image
import torch
import torch.nn.functional as F
import glob
from collections import defaultdict

from sklearn.metrics import balanced_accuracy_score

# 为样本添加噪声，构建噪声客户端
def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)
    gamma_s = np.array([0.] * args.num_users)
    gamma_s[:int(args.level_n_system*args.num_users)] = 1.
    np.random.shuffle(gamma_s)
    gamma_c_initial = np.random.rand(args.num_users) # 创建0到1的大小为20的随机数组[0.15432,0.2543,....]
    gamma_c_initial = (args.level_n_upperb - args.level_n_lowerb) * \
        gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)

    if args.n_type == "instance":
        if args.dataset == "isic2019":
            df = pd.read_csv("your csv")
        elif args.dataset == "ICH":
            df = pd.read_csv("/data/wtt/demo/FedNoRo/FedNoRo/data/ICH_softlabel.csv")
        else:
            raise

        soft_label = df.iloc[:, 1:args.n_classes+1].values.astype("float")
        real_noise_level = np.zeros(args.num_users)# 创建20个为0的数组[0.,0.,0.,...0.]
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))
            soft_label_this_client = soft_label[sample_idx]
            hard_label_this_client = y_train[sample_idx]

            p_t = copy.deepcopy(soft_label_this_client[np.arange(
                soft_label_this_client.shape[0]), hard_label_this_client])
            p_f = 1 - p_t
            p_f = p_f / p_f.sum()
            # Choose noisy samples base on the misclassification probability.
            noisy_idx = np.random.choice(np.arange(len(sample_idx)), size=int(
                gamma_c[i]*len(sample_idx)), replace=False, p=p_f)

            for j in noisy_idx:
                soft_label_this_client[j][hard_label_this_client[j]] = 0.
                soft_label_this_client[j] = soft_label_this_client[j] / \
                    soft_label_this_client[j].sum()
                # Choose a noisy label base on the classification probability.
                # The noisy label is different from the initial label.
                y_train_noisy[sample_idx[j]] = np.random.choice(
                    np.arange(args.n_classes), p=soft_label_this_client[j])

            noise_ratio = np.mean(
                y_train[sample_idx] != y_train_noisy[sample_idx])
            logging.info("Client %d, noise level: %.4f, real noise ratio: %.4f" % (
                i, gamma_c[i], noise_ratio))
            real_noise_level[i] = noise_ratio

    elif args.n_type == "random":
        real_noise_level = np.zeros(args.num_users)
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))
            prob = np.random.rand(len(sample_idx))
            noisy_idx = np.where(prob <= gamma_c[i])[0]
            y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(
                0, args.n_classes, len(noisy_idx))
            noise_ratio = np.mean(
                y_train[sample_idx] != y_train_noisy[sample_idx])
            logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
                i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
            real_noise_level[i] = noise_ratio

    else:
        raise NotImplementedError

    return (y_train_noisy, gamma_s, real_noise_level)


def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)


def get_output(loader, net, args, softmax=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if softmax == True:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def get_output_and_label(loader, net, args):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()

            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)

            if i == 0:
                output_whole = np.array(outputs.cpu())
                label_whole = np.array(labels.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                label_whole = np.concatenate(
                    (label_whole, labels.cpu()), axis=0)

    return output_whole, label_whole


def cal_training_acc(prediction, noisy_labels, true_labels):
    prediction = np.array(prediction)
    noisy_labels = np.array(noisy_labels)
    true_labels = np.array(true_labels)

    acc_noisy = balanced_accuracy_score(noisy_labels, prediction)
    acc_true = balanced_accuracy_score(true_labels, prediction)

    return acc_noisy, acc_true


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_output_files(args,task):
    outputs_dir = 'outputs_' + str(args.dataset) + '_session' + str(task)
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp + '_'+str(task))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    # shutil.copytree('.', code_dir, ignore=shutil.ignore_patterns('.git'))

    # logging.basicConfig(filename=os.path.join(logs_dir,'logs.txt'), level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)
    return writer, models_dir,logs_dir

# 根据标签划分数据集
def generate_group_dataset(args):
    csv_file = os.path.join(args.root, args.mode + ".csv")
    file = pd.read_csv(csv_file)
    # 创建一个字典，用于存储标签相同的样本组
    groups = defaultdict(list)

    for index, row in file.iterrows():
        # 找到值为1的列名，即真实标签
        true_label = row[row == 1].index[0]
        groups[true_label].append(row)
    # 将分类后的数据转换为DataFrame格式并保存到新的CSV文件中,
    group_path = '/data/wtt/demo/own/FL_CL/data/'
    prefix = args.mode+'_group'
    files = glob.glob(os.path.join(group_path, f"{prefix}*"))
    if not files:
        for label, group in groups.items():
            group_df = pd.DataFrame(group)
            group_df.to_csv(f'/data/wtt/demo/own/FL_CL/data/{args.mode}_group_{label}.csv', index=False)
    files = glob.glob(os.path.join(group_path, f"{prefix}*"))
    files.sort()
    return files

# 将示例集样本保存到txt文件
def save_to_txt(array, output_directory,args,client_id):
    images, labels, targets = array
    # 创建文件名
    file_name = f"session_{args.session}_example_client_{client_id}.csv"
    file_path = output_directory + '/' + file_name
    file_path_name = os.path.join(output_directory + '/', file_name)
    if os.path.isfile(file_path_name):
        os.remove(file_path)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    # 将示例集样本和标签读取路径写入txt文件
    with open(file_path, 'w') as file:
        csv_writer=csv.writer(file)
        csv_writer.writerow(['id','epidural','intraparenchymal','intraventricular','subarachnoid','subdural','target'])
        for id in range(len(images)):
            csv_writer.writerow([images[id],labels[id][0],labels[id][1],labels[id][2],labels[id][3],labels[id][4],targets[id]])



#从txt文件中读取示例集样本
def read_example_from_csv(path):
    file=pd.read_csv(path)
    example_images = file["id"].values
    example_labels = file.iloc[:, 1:].values.astype("int")
    return example_images,example_labels


def read_txt_to_array(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    numbers = re.findall(r'\d+', content)
    int_list = [int(num) for num in numbers]
    array = np.array(int_list)

    return array


def average_models(client_models):
    # 获取模型中参数的名称
    model= client_models[0].state_dict()
    model_keys=model.keys()

    # 初始化一个新的状态字典
    averaged_model = OrderedDict()

    for key in model_keys:
        # 将每个客户端的参数取出并求平均
        averaged_model[key] = torch.mean(torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))]), dim=0)

    return averaged_model