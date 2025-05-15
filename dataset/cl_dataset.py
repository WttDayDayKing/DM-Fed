####generate CL dataset
import csv
import logging
import random

import pandas as pd
import torch
import torchvision.transforms as transforms
from .all_datasets import CL_ICH
import numpy as np

from utils.sampling import cl_iid_sampling,cl_non_iid_sampling
from utils.util import save_to_txt

def get_cl_dataset(args,client_id):
   if args.dataset=='ICH':
      root = "/data/wtt/demo/FedNoRo/FedNoRo/data"

      normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])
      train_transform = transforms.Compose([
         transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize,
      ])
      val_transform = transforms.Compose([
         transforms.ToTensor(),
         normalize,
      ])

      test_dataset = CL_ICH(root, "test_ICH", val_transform, args)  # 获取到CL任务的测试数据集
      train_dataset = CL_ICH(root, "train_ICH", train_transform, args,client_id)  # 获取到每个CL任务每个客户端的训练数据集
      #test_dataset = CL_ICH(root, "test_ICH", val_transform,args)  #获取到CL任务的测试数据集

      #更新示例集
      update_train_example_data_set(args,train_dataset,client_id)
      #增量任务时读取示例集并添加到当前训练样本集合
      if args.session>0 and train_dataset.images is not None and train_dataset.labels is not None and train_dataset.targets is not None:
         for idx in range(args.session):
            file_path = args.dataset_path + '/' + f'session_{str(idx)}_example_client_' + str(client_id) + '.csv'
            file = pd.read_csv(file_path)
            images = file["id"].values

            labels = file.iloc[:, 1:-1].values.astype("int")
            targets = file.iloc[:, -1].values.astype("int")
            new_images=[images[i] for i in range(len(images))]

            train_dataset.images.extend(new_images)
            train_dataset.labels = np.concatenate((train_dataset.labels, labels))
            train_dataset.targets = np.concatenate((train_dataset.targets, targets))

      print("### Datasets are ready ###")

      return train_dataset, test_dataset

#每个任务更新示例集
def update_train_example_data_set(args,train_dataset,client_id):
   images,labels,targets=train_dataset.images,train_dataset.labels,train_dataset.targets
   num_to_select = int(len(images) * args.ratio)  # 按比例计算选择多少个示例样本
   all_index = [i for i in range(len(images))]
   select_index = random.sample(all_index, num_to_select)

   select_images = [images[id] for id in select_index]
   select_labels = [labels[id] for id in select_index]

   select_targets = [targets[id] for id in select_index]
   select_example_sample = select_images, select_labels, select_targets

   # 将示例集保存
   save_to_txt(select_example_sample, args.dataset_path, args,client_id)



