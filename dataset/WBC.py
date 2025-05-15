from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from . import data_base

class wbc(Dataset):
    def __init__(self, args, is_train=True, index=None,
                 base_session=True, old_index=None, netglobals=None, current_task=0, pim_model_list=None):
        self.is_train = is_train  # training set or test set
        self.transform = None
        self.args = args
        self.netglobals = netglobals
        self.pim_model_list = pim_model_list
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

            if base_session:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index, current_task)
            else:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index, old_index,
                                                                 current_task)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            if old_index is not None:
                index = np.append(index, old_index)
            have_see_class = index
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, have_see_class)

    def _pre_operate(self, args):
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

    def SelectfromClasses(self, data, targets, index, old_index=None, current_task=0):
        data_tmp = []
        targets_tmp = []
        print(f"current class index:{index}")
        print(f"current load target:{set(targets)}")
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        if (self.is_train == False and old_index is not None) or (
                self.args.use_old_class_index and old_index is not None):
            for i in old_index:
                ind_cl = np.where(i == targets)[0]
                for j in ind_cl[0:20]:  ###选择旧类的前20个样本加入到当前训练任务中
                    data_tmp.append(data[j])
                    targets_tmp.append(targets[j])
        elif self.is_train == True:
            #####使用pim方法,更新示例集
            if self.args.use_pim:
                data_base.update_client_pim(self, current_task=current_task)

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = Image.open(path).convert("RGB")

        # try:
        #     with Image.open(path) as img:
        #         # 尝试打开并处理 TIFF 图像
        #         img.thumbnail((2048, 2048), Image.ANTIALIAS)  # 限制最大尺寸
        #         img = img.convert('RGB')  # 转换为 RGB
        # except Exception as e:
        #     print(f"捕获到异常无法正确加载图片", e)
        #     print(f"异常图片路径为：", path)

        classify_image = self.transform(image)
        total_image = classify_image
        return total_image, targets
