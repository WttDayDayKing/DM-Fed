import torch

def cal_client_memory_size(self,client_model,global_model,data_size,all_data_size):
    l1_norm = 0.0
    for param1, param2 in zip(client_model.parameters(), global_model.parameters()):
        l1_norm += torch.sum(torch.abs(param1.data - param2.data)).item()  # 计算 L1 范数
    n=self.args.l1_weight*l1_norm+(1-self.args.l1_weight)*(data_size/all_data_size)
    return n