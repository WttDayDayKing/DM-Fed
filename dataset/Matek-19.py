from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np

class matek19(Dataset):
    def __init__(self, root='./', is_train=True, index=None,
                 base_session=True, old_index=None):
        self.is_train = is_train  # training set or test set
        self.transform = None
        self._pre_operate(self.root)
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomAffine(degrees=10,translate=(0.02,0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

            if base_session:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index, old_index)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def _pre_operate(self, root):
        split_file_train =self.args.save_train_dir
        split_file_test=self.args.save_test_dir
        self.data = []
        self.targets = []
        if self.is_train:
            for file_name in os.listdir(split_file_train):
                images = os.listdir(os.path.join(split_file_train, file_name))
                for k in  range(len(images)):
                    image_path = os.path.join(split_file_train, file_name,images[k])
                    self.data.append(image_path)
                    self.targets.append(int(file_name))

        else:
            for file_name in os.listdir(split_file_test):
                images = os.listdir(os.path.join(split_file_test, file_name))
                for k in range(len(images)):
                    image_path = os.path.join(split_file_test, file_name,images[k])
                    self.data.append(image_path)
                    self.targets.append(int(file_name))

    def SelectfromClasses(self, data, targets, index, old_index=None):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        if old_index is not None:
            for i in old_index:
                ind_cl = np.where(i == targets)[0]
                for j in ind_cl[0:20]:
                    data_tmp.append(data[j])
                    targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = Image.open(path).convert('RGB')
        classify_image = self.transform(image)
        total_image = classify_image
        return total_image, targets