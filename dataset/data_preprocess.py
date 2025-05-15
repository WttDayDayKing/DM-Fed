import numpy as np
import os
import shutil
from utils.options import args_parser
#将3个数据集预处理划分为训练集和测试集
equivalent_classes = {

    # Acevedo-20 dataset
    'basophil': 'basophil',
    'eosinophil': 'eosinophil',
    'erythroblast': 'erythroblast',
    'ig': "unknown",  # immature granulocytes,
    'PMY': 'promyelocyte',  # immature granulocytes,
    'MY': 'myelocyte',  # immature granulocytes,
    'MMY': 'metamyelocyte',  # immature granulocytes,
    'lymphocyte': 'lymphocyte_typical',
    'monocyte': 'monocyte',
    'neutrophil': "unknown",
    'BNE': 'neutrophil_banded',
    'SNE': 'neutrophil_segmented',
    'platelet': "unknown",
    # Matek-19 dataset
    'BAS': 'basophil',
    'EBO': 'erythroblast',
    'EOS': 'eosinophil',
    'KSC': 'smudge_cell',
    'LYA': 'lymphocyte_atypical',
    'LYT': 'lymphocyte_typical',
    'MMZ': 'metamyelocyte',
    'MOB': 'monocyte',  # monoblast
    'MON': 'monocyte',
    'MYB': 'myelocyte',
    'MYO': 'myeloblast',
    'NGB': 'neutrophil_banded',
    'NGS': 'neutrophil_segmented',
    'PMB': "unknown",
    'PMO': 'promyelocyte',
     # WBC
    'Basophile': 'basophil',
    'Eosinophile': 'eosinophil',
    'Lymphoblast': 'lymphocyte_typical',
    'Lymphocyte': 'lymphocyte_typical',
    'Monocyte': 'monocyte',
    'Myeloblast': 'myeloblast',
    'Neutrophile Band': 'neutrophil_banded',
    'Neutrophile Segment': 'neutrophil_segmented',
    'Normoblast': 'erythroblast',
    # #  INT-20 dataset
    # '01-NORMO': 'erythroblast',
    # '04-LGL': "unknown",  # atypical
    # '05-MONO': 'monocyte',
    # '08-LYMPH-neo': 'lymphocyte_atypical',
    # '09-BASO': 'basophil',
    # '10-EOS': 'eosinophil',
    # '11-STAB': 'neutrophil_banded',
    # '12-LYMPH-reaktiv': 'lymphocyte_atypical',
    # '13-MYBL': 'myeloblast',
    # '14-LYMPH-typ': 'lymphocyte_typical',
    # '15-SEG': 'neutrophil_segmented',
    # '16-PLZ': "unknown",
    # '17-Kernschatten': 'smudge_cell',
    # '18-PMYEL': 'promyelocyte',
    # '19-MYEL': 'myelocyte',
    # '20-Meta': 'metamyelocyte',
    # '21-Haarzelle': "unknown",
    # '22-Atyp-PMYEL': "unknown",
}

# label_map = {
#     'basophil': 0,
#     'eosinophil': 1,
#     'erythroblast': 2,
#     'myeloblast': 3,
#     'promyelocyte': 4,
#     'myelocyte': 5,
#     'metamyelocyte': 6,
#     'neutrophil_banded': 7,
#     'neutrophil_segmented': 8,
#     'monocyte': 9,
#     'lymphocyte_typical': 10,
#     'lymphocyte_atypical': 11,
#     'smudge_cell': 12,
# }

label_map_matek19 = {
    'basophil': 0,
    'neutrophil_banded': 1,
    'promyelocyte': 2,
    'erythroblast': 3,
    'eosinophil': 4,
    'myeloblast': 5,
    'myelocyte': 6,
    'metamyelocyte': 7,
    'neutrophil_segmented': 8,
    'lymphocyte_atypical': 9,
    'monocyte': 10,
    'lymphocyte_typical': 11,
    'smudge_cell': 12,
}

label_map_acevedo = {
    'basophil': 0,
    'neutrophil_banded': 1,
    'promyelocyte': 2,
    'erythroblast': 3,
    'eosinophil': 4,
    # 'myeloblast': 5,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'neutrophil_segmented': 7,
    # 'lymphocyte_atypical': 9,
    'monocyte': 8,
    'lymphocyte_typical': 9,
    # 'smudge_cell': 12,
}

label_map_wbc={
    'basophil': 0,
    'neutrophil_banded': 1,
    'erythroblast': 2,
    'eosinophil': 3,
    'myeloblast': 4,
    'neutrophil_segmented': 5,
    'monocyte': 6,
    'lymphocyte_typical': 7,
}

# 对数据集进行预处理：1、包括去除未知类别样本 2、生成训练样本集和测试集
class data_preprocess():
    def __init__(self,args):
        self.args=args
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp'}
        datasets = {}
        images_count={}
        self.dataset_path=os.path.join(self.args.dataset_dir, args.dataset_name)
        keys = np.unique([x for x in os.listdir(self.dataset_path)])
        for key in keys:
            datasets[key] = key
        print("dataset.keys:",datasets.keys())
        samples2 = datasets.copy()
        #清除无效样本
        for s in samples2.keys():
            if not s in equivalent_classes.keys() or equivalent_classes[s] == "unknown":
                datasets.pop(s, None)

        print("清除无效样本后，datasets.keys:",datasets.keys())
        #计算每个类别样本个数
        for inx in datasets.keys():
            image_count=0
            for file in os.listdir(os.path.join(self.dataset_path,datasets[inx])):
                if file.lower().endswith(tuple(image_extensions)):
                    image_count+=1
            images_count[inx]=image_count
            print("类别：{},样本个数：{}",inx,images_count[inx])

        self.datasets=datasets
        self.images_count=images_count
        self.sample_train_test(self)

    def sample_train_test(self,args):
        select_index_train={}
        select_index_test={}
        png_files={}
        save_train_samples_dir=self.args.save_train_dir+self.args.dataset_name+'/train'
        save_test_samples_dir=self.args.save_test_dir+self.args.dataset_name+'/test'
        #每个类别按照8:2划分训练集和测试集
        for id in self.datasets:
            train_num=int(self.images_count[id]*self.args.train_ratio)
            for filename in os.listdir(os.path.join(self.dataset_path,self.datasets[id])):
                if id not in png_files:
                    png_files[id] = []
                png_files[id].append(filename)
            index_per_class=np.arange(len(png_files[id]))
            select_index_train[id]=set(
                    np.random.choice(index_per_class, train_num,
                                     replace=False))
            select_index_test[id]=set(index_per_class)-select_index_train[id]
            if self.args.dataset_name=='Acevedo-20':
                label_index=label_map_acevedo[equivalent_classes[id]]
            elif self.args.dataset_name=='Matek-19':
                label_index=label_map_matek19[equivalent_classes[id]]
            else:
                label_index=label_map_wbc[equivalent_classes[id]]
            #已选择的训练和测试样本数据存放至文件
            self.save_file_in_folder(self, save_train_samples_dir, png_files[id], select_index_train[id], label_index,id)
            self.save_file_in_folder(self,save_test_samples_dir,png_files[id],select_index_test[id],label_index,id)



    def save_file_in_folder(self,args,save_samples_dir,png_files,select_index,label_index,id):
        if not os.path.exists(save_samples_dir):
            os.mkdir(save_samples_dir)

        select_images_filename = [png_files[index] for index in select_index]
        save_class_train_dir = os.path.join(save_samples_dir, str(label_index))
        if not os.path.exists(save_class_train_dir):
            os.mkdir(save_class_train_dir)
        source_folder = os.path.join(self.dataset_path, self.datasets[id])
        for select_file in select_images_filename:
            source_png_file = os.path.join(source_folder, select_file)
            target_png_file = os.path.join(save_class_train_dir, select_file)
            shutil.copy(source_png_file, target_png_file)

if __name__ == '__main__':
    args = args_parser()
    data_preprocess(args)