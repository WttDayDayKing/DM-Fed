from torch.utils.data import Dataset
import os
import numpy as np

#数据集中各文件夹名对应类别
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
    'Basophile':'basophil',
    'Eosinophile':'eosinophil',
    'Lymphoblast':'lymphocyte_typical',
    'Lymphocyte':'lymphocyte_typical',
    'Monocyte':'monocyte',
    'Myeloblast':'myeloblast',
    'Neutrophile Band':'neutrophil_banded',
    'Neutrophile Segment':'neutrophil_segmented',
    'Normoblast':'erythroblast',
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

#各类别对应索引
label_map = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'myeloblast': 3,
    'promyelocyte': 4,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'neutrophil_banded': 7,
    'neutrophil_segmented': 8,
    'monocyte': 9,
    'lymphocyte_typical': 10,
    'lymphocyte_atypical': 11,
    'smudge_cell': 12,
}
class DataLoader(Dataset):
    def __init__(self,train=True):
        self.dataset_name=["Matek-19","Acevedo-20","Labelled"]
        datasets={}
        images={}
        keys = np.unique([x for x in os.listdir(self.args.dataset_dir)])
        for key in keys:
            datasets[key]=key
        for s in datasets.keys():
            if equivalent_classes[s]=="unknown":
                datasets.pop(s,None)
        for inx in datasets.keys():
            images_dir=os.path.join(self.args.dataset_dir,datasets[inx])
            image_files = os.listdir(images_dir)
            for image_file in image_files:
                with open(os.path.join(images_dir,image_file),"rb") as f:
                    file_images = pickle.load(f)
                images = {**images, **file_images}
        self.images=images
        self.datasets=datasets

    def __len__(self):
        return len(self.data)

