# 高级机器学习技术
Please read before executing the code; ensure a runnable environment and dataset are provided.

## Brief Introduction
This report presents several federated incremental learning methods on three datasets.

## Dataset 
Matek-19:A single-cell morphological dataset of leukocytes from aml patients and non-malignant controls (aml-
cytomorphology lmu).
Acevedo-20: A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.
HLwbc:A high-resolution large-scale dataset of pathological and normal white blood cells.
### datasets structure
The data structure is as follows(such as Matek-19)：
- Matek-19/
  - BAS/
    - BAS_0001.tiff
    - ...  
  - EOS/
    - EOS_0001.tiff
    - ...  
  - ...
Before running the algorithm, please execute: python data_preprocess.py --dataset_dir="Downloaded dataset path" --dataset_name="Matek-19" --save_train_dir="Training set path" --save_test_dir="Test set path"
### Access URL of the processed data 
To improve deployment efficiency, we provide preprocessed datasets for direct download and use.
Please load: 

## Task settings
- FCIL
  - Matek-19
    base_class:4
    way:3
    - session 0:0,1,2,3
    - session 1:4,5,6
    - session 2:7,8,9
    - session 3:10,11,12
  - Acevedo-20
    base_class:4
    way:3
    - session 0:0,1,2,3
    - session 1:4,5,6
    - session 2:7,8,9

  - HLwbc
    base_class:4
    way:2
    - session 0:0,1,2,3
    - session 1:4,5
    - session 2:6,7
    
- FDIL
  - session 0:Matek-19
  - session 1:Acevedo-20
  - session 2:HLwbc
## Methods
### FCIL-M (Defult setting)
We provide default configurations for FCIL on the Matek-19 dataset.    
Such as: python train_FedCL.py --save_train_dir="/data/FL_CL" --save_test_dir="/data/FL_CL"   
Please replace the “save_train_dir” and “save_test_dir” with your actual paths. Note that the directory structure should follow this format:  
- FL_CL/
  - Matek-19/
    - train/
    - test/
  - Acevedo-20/
    - train/
    - test/
### FCIL-A
1. Modify /utils/options.py:  
   save_train_dir  
   save_test_dir  
   dataset=Acevedo-20  
   session=3  
   task_num=3  
   n_classes=10  
   cl_step=3  
   dataset_name=Acevedo-20  
2. training  
   python train_FedCL.py  
### FCIL-H
1. Modify /utils/options.py:  
   save_train_dir=your train set path  
   save_test_dir=your test set path  
   dataset=Labelled  
   session=3  
   task_num=3  
   n_classes=8  
   cl_step=3  
   dataset_name=Labelled  
   way=2  
3. training  
   python train_FedCL.py  



