import logging
import os
import numpy as np
from PIL import Image
import pandas as pd
import glob

import torch
from torch.utils.data import Dataset
from utils.util import generate_group_dataset
from utils.sampling import get_client_index
from collections import defaultdict

class isic2019(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test"]
        self.transform = transform

        csv_file = os.path.join(self.root, self.mode+".csv")
        self.file = pd.read_csv(csv_file)
        self.images = self.file["image"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)
        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 8

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(
            self.root, "ISIC_2019_Training_Input", self.images[index]+".jpg")
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        label = self.targets[index]
        return img, label

    

class ICH(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train_ICH", "test_ICH"]
        self.transform = transform

        csv_file = os.path.join(self.root, self.mode+".csv")
        self.file = pd.read_csv(csv_file)
        self.images = self.file["id"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)
        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 5

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        id, target = self.images[index], self.targets[index]
        img = self.read_image(id)

        img = self.transform(img)
        return img, target

    def read_image(self, id):
        image_path = os.path.join(self.root, "stage_1_train_images", id+".png")
        image = Image.open(image_path).convert("RGB")
        return image

class CL_ICH(Dataset):
    def __init__(self,root,mode,transform=None,args=None,client=0):
        self.root = root
        self.mode = mode
        assert self.mode in ["train_ICH", "test_ICH"]
        self.transform = transform
        self.session=args.session#当前任务
        self.args=args
        self.client=client#当前客户端
        self.images=[]
        self.labels=np.zeros((1, 5))
        self.targets=[]
        csv_files=[]
        # 读取每个训练阶段session的数据集
        files = generate_group_dataset(self)
        logging.info(f"session:{args.session},files:{files}")
        if self.mode=='test_ICH' and self.session>0:
            for i in range(self.session+1):
                csv_file=files[i]
                csv_files.append(csv_file)
            logging.info(f"csv_files:{csv_files}")
            for i in range(len(csv_files)):
                csv_file = csv_files[i]
                # 获取路径下样本
                image, label, target = get_test_cliant_sample(csv_file)
                self.images=np.concatenate((self.images,image))
                self.labels=np.concatenate((self.labels,label))
                self.targets = np.concatenate((self.targets, target))
        elif self.mode=='test_ICH' and self.session==0:
            self.images, self.labels, self.targets=get_test_cliant_sample(files[self.session])
        logging.info(f"test_ICH,session:{self.session}")

        if self.mode=='train_ICH':
            csv_file = files[self.session]
            self.images,self.labels,self.targets=get_train_client_sample(self,csv_file)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        id, labels = self.images[index], self.targets[index]
        img = self.read_image(id)

        img = self.transform(img)
        return img, labels

    def read_image(self, id):
        image_path = os.path.join(self.root, "stage_1_train_images", id + ".png")
        image = Image.open(image_path).convert("RGB")
        return image
# 获取测试集
def get_test_cliant_sample(csv_file):
    file = pd.read_csv(csv_file)
    images = file["id"].values
    labels = file.iloc[:, 1:].values.astype("int")

    targets = np.argmax(labels, axis=1)

    return images,labels,targets

#获取每个客户端的训练数据集
def get_train_client_sample(self,csv_file):
    file = pd.read_csv(csv_file)
    images = file["id"].values
    labels = file.iloc[:, 1:].values.astype("int")
    targets= np.argmax(labels, axis=1)
    file_row = len(file)

    # 根据均匀分布获取分配的各客户端样本索引
    dict_per_client_index = get_client_index(file_row, self.args.n_clients, self)
    all_images=[images[i] for i in dict_per_client_index[self.client]]
    all_labels=[labels[i] for i in dict_per_client_index[self.client]]
    all_targets=[targets[i] for i in dict_per_client_index[self.client]]
    return all_images,all_labels,all_targets