import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed_WTT', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='Matek-19', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--pretrained', type=int,  default=0)
    parser.add_argument('--local_bs',type=int,default=64,help="local train batch size")

    # for FL
    parser.add_argument('--n_clients', type=int,  default=5,
                        help='init number of users')
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid, fixed')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=1.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=40, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=20, help='rounds')

    parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a', type=float,  default=0.8, help='a')
    parser.add_argument('--warm', type=int,  default=1)
    parser.add_argument('--feddwa_topk',type=int,default=5,help="the number of combine in server")
    parser.add_argument('--combine',type=str,default="FedAvg",help="aggregation methods")

    # noise
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default="instance", help="type of noise")

    # fro CL
    parser.add_argument('--session',type=int,default=4,help='current task,only support 0-4')
    parser.add_argument('--dataset_path',type=str,default='/data/wtt/demo/own/FL_CL/data/example',help='save path of dataset')
    parser.add_argument('--task_num',type=int,default=4,help='CL task num')
    parser.add_argument('--ratio',type=float,default=0.3,help='example sample ratio')
    parser.add_argument('--n_classes',type=int,default=13,help='class number')
    parser.add_argument('--base_class',type=int,default=4,help='class numberexp of base session')
    parser.add_argument('--cl_step',type=int,default=4,help='the increase step of session')
    parser.add_argument('--il_type',type=str,default="CIL",help='Incremental task type')
    ####for pim
    parser.add_argument('--use_old_class_index',type=bool,default=False,help="the switch for random sample to deplay")
    parser.add_argument('--use_pim',type=bool,default=True,help="use pim for sample deplay")
    parser.add_argument('--ite',type=float,default=0.1,help="the rate to control the step size of the update pim model")
    parser.add_argument('--hyper',type=float,default=0.4,help="hyper-parameter adjusts the balance between the local and global information.")
    parser.add_argument('--local_pim_bs',type=int,default=10,help="batch size of update pim model")
    parser.add_argument('--cached_sample_size',type=int,default=200,help="cached sample size for each client")
    # for dtataset
    parser.add_argument('--save_train_dir',type=str,default='/data/wtt/data_set/FL_CL',help="load dataset path for train model")
    parser.add_argument('--save_test_dir',type=str,default='/data/wtt/data_set/FL_CL')
    parser.add_argument('--dataset_dir',type=str,default='/data/wtt/data_set/FL_CL',help="preprocess data output dir")
    parser.add_argument('--dataset_name',type=str,default='Matek-19')
    parser.add_argument('--train_ratio',type=float,default=0.8,help="dataset split traio")
    parser.add_argument('--way',type=int,default=3,help="incre class number per task,CIL-M and CIL-A=3,CIL-H=2")
    parser.add_argument('--client_max_cache_size',type=int,default=400,help="max cache size of each client")
    parser.add_argument('--server_all_cache_size',type=int,default=1200,help="all cache size of all client")
    parser.add_argument('--l1_weight',type=float,default=0.3,help="the factor of l1 norm to calculate memory size")
    parser.add_argument('--use_dynamic_memory',type=bool,default=True,help="if use dynamic memory size")
    parser.add_argument('--use_dynamic_class_memory',type=bool,default=True,help="is use dynamic class cache size per client")
    parser.add_argument('--use_carl',default=False,type=bool,help="is use icarl")
    parser.add_argument('--use_margin_regular',default=True,type=bool,help="is use local Margin Control,L_mg")
    parser.add_argument('--lambda_value',default=0.02,type=float,help="the weight of L_mg")
    parser.add_argument('--get_all_old_data',default=False,help="base line all data train,UP")
    parser.add_argument('--use_uacl',default=False,type=bool,help="is use uacl")
    parser.add_argument('--use_kd_loss',default=True,type=bool,help="loss for L_KL")
    parser.add_argument('--data_dist_weight',default=0.4,type=float,help="the weight of global data distribution")
    args = parser.parse_args()
    return args
