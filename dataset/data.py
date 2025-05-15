import os
import PIL
from PIL import Image
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
from .common import create_lda_partitions
from .data_util import non_iid_dirichlet_sampling


def get_dataset(args):
    if args.dataset_name=='Matek-19':
        train_dataset=get_matek(args,True)
        test_dataset=get_matek(args,False)
        return train_dataset,test_dataset
    else:
        raise NotImplementedError


def get_matek(args):
    args.num_classes=13

    class Matek19(torch.utils.data.Dataset):
        def __init__(self, args, is_train=True):
            self.transform = None
            self.args = args
            self.is_train=is_train
            self._pre_operate(self)
            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
                ])

        def _pre_operate(self):
            split_file_train = os.path.join(self.args.save_train_dir, self.args.dataset_name, "train")
            split_file_test = os.path.join(self.args.save_test_dir, self.args.dataset_name, "test")
            self.data = []
            self.targets = []
            if self.is_train:
                for file_name in os.listdir(split_file_train):
                    images = os.listdir(os.path.join(split_file_train, file_name))
                    for k in range(len(images)):
                        image_path = os.path.join(split_file_train, file_name, images[k])
                        self.data.append(image_path)
                        self.targets.append(int(file_name))

            else:
                for file_name in os.listdir(split_file_test):
                    images = os.listdir(os.path.join(split_file_test, file_name))
                    for k in range(len(images)):
                        image_path = os.path.join(split_file_test, file_name, images[k])
                        self.data.append(image_path)
                        self.targets.append(int(file_name))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            path, targets = self.data[i], self.targets[i]
            try:
                with Image.open(path) as img:
                    # 尝试打开并处理 TIFF 图像
                    img.thumbnail((2048, 2048), Image.ANTIALIAS)  # 限制最大尺寸
                    img = img.convert('RGB')  # 转换为 RGB
            except Exception as e:
                print(f"捕获到异常无法正确加载图片", e)
                print(f"异常图片路径为：", path)

            classify_image = self.transform(img)
            total_image = classify_image
            return total_image, targets


class CL_dataset():
    def __init__(self, args):
        self.args = args
        self.name = args.dataset_name
        self.train_dataset, self.test_dataset = get_dataset(self.args)###所有数据
        self.classes = np.arange(len(np.unique(self.train_dataset.targets)))
        
        self.initial_classes = args.base_class
        self.increment_classes = args.way
        self.total_classes = len(self.classes)
        
        self.train_ds, self.cl_test_loaders, total_test = [], [], []
        self.full_test_loaders, current_train, current_test = [], [], []
        
        self.n_classes_per_task = self.initial_classes if self.args.n_tasks == 1 else self.increment_classes
        #self.n_classes_per_task = len(self.classes) // self.args.n_tasks
        for i, label in enumerate(self.classes):
            current_train.extend(np.where(self.train_dataset.targets == label)[0].tolist())
            current_test.extend(np.where(self.test_dataset.targets == label)[0].tolist())
            if (i == self.initial_classes-1) or (self.args.session > 1 and i >= self.initial_classes and (i - self.initial_classes + 1) % self.increment_classes == 0):
                self.train_ds += [current_train]
                total_test.extend(current_test)
                self.cl_test_loaders.append(DataLoader(Subset(self.test_dataset, current_test), batch_size=self.args.batch_size, shuffle=False))
                self.full_test_loaders.append(DataLoader(Subset(self.test_dataset, deepcopy(total_test)), batch_size=self.args.batch_size, shuffle=False))
                current_train, current_test = [], []
        self.groups = defaultdict(list)
        for task_id in range(args.session):
            train_indx = self.train_ds[task_id]
            targets = np.array(self.train_dataset.targets)[train_indx]
            if task_id==1 or task_id==3:
                args.n_clients+=1
            task_group =non_iid_dirichlet_sampling(task_id,targets,self.n_classes_per_task,self.args.non_iid_prob_class,args.n_clients,100,self.args.alpha_dirichlet)
            #task_group = self.get_task_group(task_id, args.num_users)
            for client in range(args.n_clients):
                data_i = task_group[client]
                self.groups[client].append(data_i)

    def get_task_group(self, task_id, num_users):

        train_indx = self.train_ds[task_id]
        targets = np.array(self.train_dataset.targets)[train_indx]

        groups, _ = create_lda_partitions(dataset=targets, num_partitions=num_users, concentration=self.args.alpha, accept_imbalanced=False)
        groups = [(np.array(train_indx)[groups[i][0]]).tolist() for i in range(num_users)]
        return groups

    def get_full_train(self, task_id):
        indexes = []
        for t in range(task_id + 1):
            indexes.extend(self.train_ds[t])
        return DataLoader(Subset(self.train_dataset, indexes), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    def get_full_test(self, t):
        return self.full_test_loaders[t]

    def get_cl_test(self, t):
        return self.cl_test_loaders[:t + 1]
