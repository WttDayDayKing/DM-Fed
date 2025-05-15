import numpy as np
import torch
import math
import copy
class FedDWA(object):
    def __init__(self,args,w_locals,w_more_locals):
        self.args=args
        self.receive_client_next_models=[]
        self.w_locals=w_locals
        self.w_more_locals=w_more_locals
        self.send_client_models=None
        self.selected_clients_idx=[i for i in range(args.num_users)]
        self.num_users=self.args.num_users
        self.feddwa_topk=self.args.feddwa_topk

    def feddwa(self,rnd):
        self.receive_models()
        weight_matrix=self.cal_optimal_weight()
        self.aggregated(weight_matrix)
        self.cal_client_memory_size(rnd)

    def aggregated(self, optimal_matrix):
        """
                Aggregation method for FedDWA, use the optimal weight and the received model in
        the current round  to calculate the next round model which will be sent to each client
        the results are stored in self.send_client_models
        :param optimal_matrix:
        :type optimal_matrix: tensor matrix
        :return:
        :rtype:
        """
        # assert (len(self.selected_clients_idx) > 0)
        send_client_models = self.receive_client_next_models
        for idx, model_params in enumerate(send_client_models):
            weights = optimal_matrix[:, idx]
            for name in model_params:
                tmp = torch.stack(
                    [source[name].data.clone() * weights[row] for row, source in enumerate(self.receive_client_models)])
                tmp = torch.sum(tmp, dim=0).clone()
                model_params[name].data = tmp
        self.send_client_models = {idx: send_client_models[num] for num, idx in enumerate(self.selected_clients_idx)}

    def receive_models(self):
        self.receive_client_models = [{key: value for key, value in self.w_locals[idx].named_parameters()} for
                                      idx in range(self.num_users)]
        self.receive_client_next_models = [self.w_more_locals[idx] for idx in range(self.num_users)]

    def cal_optimal_weight(self):
        weight_matrix = np.zeros([self.num_users, self.num_users], dtype=np.float32)
        for col, s1_col_trainafter in enumerate(self.receive_client_next_models):
            for row, s2_row_trainafter in enumerate(self.receive_client_models):
                weight_matrix[row, col] = self.cal_norm(s1_col_trainafter, s2_row_trainafter)

        weight_matrix = self.column_normalization(weight_matrix)
        print(f"weight_matrix:{weight_matrix}")
        return weight_matrix

    def cal_norm(self, model1, model2):

        for idx, (k,v) in enumerate(model1.items()):
            if idx == 0:
                sum = torch.norm(v-model2[k])**2
            else:
                sum += torch.norm(v-model2[k])**2
        sum = 1 / sum
        return sum.detach().cpu().numpy()

    def column_normalization(self, matrix):
        """
        For a real matrix, sum over columns and then normalize each column to [0,1]
        :param matrix:
        :type matrix: numpy
        :return: normalized matrix
        :rtype:
        """
        if matrix.ndim == 1:
            column_sum = np.sum(matrix)
            return matrix / column_sum
        elif matrix.ndim == 2:
            result = np.zeros_like(matrix)
            M, N = matrix.shape
            for n in range(N):
                column_sum = np.sum(matrix[:, n])
                result[:, n] = matrix[:, n] / column_sum
            return result
        else:
            print("The input tensor array is not 1- or 2-dimensional.")
            return None

    def column_top_k(self, matrix, K=5):
        """inspect matrix and only store the top-K weight for each column"""
        total_num = matrix.shape[0]
        omit_num = (total_num - K) if (total_num > K) else 0
        for col in range(matrix.shape[1]):
            weights = matrix[:, col]
            mask = np.argpartition(weights, omit_num)[0:omit_num]
            weights[mask] = 0
            matrix[:, col] = weights
        return matrix

    def evaluate_acc(self, selected_all=False):
        """
        calclulate each client test acccuracy and then
        calclulate the weighted-mean accuracy
        """
        if selected_all == True:
            # test all clients
            acc_logs = []
            for idx in range(self.args.num_users):
                client_test_acc = self.w_locals[idx].test_accuracy()
                acc_logs.append(client_test_acc)

            client_mean_test_acc = 0.0
            receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in range(self.num_clients)])
            receive_client_weight = receive_client_datasize / receive_client_datasize.sum()
            for weight, acc in zip(receive_client_weight, acc_logs):
                client_mean_test_acc += weight * acc
        else:
            acc_logs = []
            for idx in self.selected_clients_idx:
                client_test_acc = self.clientsObj[idx].test_accuracy()
                acc_logs.append(client_test_acc)

            client_mean_test_acc = 0.0
            receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in self.selected_clients_idx])
            receive_client_weight = receive_client_datasize / receive_client_datasize.sum()
            for weight, acc in zip(receive_client_weight, acc_logs):
                client_mean_test_acc += weight * acc

        return acc_logs, client_mean_test_acc

    def test_accuracy(self,model):
        """
        Rewrite the method in clientBase, since in the method, for each client,
        they have their personalized model, and we use the personalized model
        to test the data.
        """
        correct = 0
        total = 0
        old_model = copy.deepcopy(model.state_dict())
        if self.next_step_model is not None:
            cur_model = model.state_dict()
            for k, v in self.next_step_model.items():
                cur_model[k] = v
            model.load_state_dict(cur_model)

        model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        acc = correct / total
        self.model.load_state_dict(old_model)
        return acc

    def cal_client_memory_size(self,rnd):
        task_server_all_cache_size = self.args.server_all_cache_size
        for idx in range(len(self.receive_client_models)):
            # l1_norm = 0.0
            # for param1, param2 in zip(self.receive_client_models[idx].parameters(), self.send_client_models[idx].parameters()):
            #     l1_norm += torch.sum(torch.abs(param1.data - param2.data)).item()  # 计算 L1 范数
            global_state_dict=self.receive_client_models[idx]
            local_state_dict=self.send_client_models[idx]
            l2_norm = compute_l2_norm(global_state_dict, local_state_dict)
            n = self.args.l1_weight * l2_norm + (1 - self.args.l1_weight) * (
                    self.args.client_data_size[idx] / self.args.all_data_size)
            #n = self.args.l1_weight * l1_norm + (1 - self.args.l1_weight) * (self.args.client_data_size[idx] / self.args.all_data_size)
            self.args.client_memeory_size[rnd].append(n.item())
        _sum=sum(self.args.client_memeory_size[rnd])
        self.args.client_memeory_size[rnd] = [(x / _sum) * task_server_all_cache_size for x in
                                         self.args.client_memeory_size[rnd]]
        self.args.client_memeory_size[rnd] = [math.floor(x) for x in self.args.client_memeory_size[rnd]]

        #print("client memory size:{}",self.args.client_memeory_size[rnd])
        ##排除已达到最大size的客户端，剩余的size重新加权
        if any(x > self.args.client_max_cache_size for x in self.args.client_memeory_size[rnd]):
            self.args.client_memeory_size[rnd] = [min(x, self.args.client_max_cache_size) for x in self.args.client_memeory_size[rnd]]
            all_indices = [index for index, value in enumerate(self.args.client_memeory_size[rnd])]
            max_indices = [index for index, value in enumerate(self.args.client_memeory_size[rnd]) if value == self.args.client_max_cache_size]
            task_server_all_cache_size = task_server_all_cache_size - len(
                max_indices) * self.args.client_max_cache_size
            while True:
                # 获取需要重新加权的索引
                need_reweight_indices = [index for index in all_indices if index not in max_indices]

                if not need_reweight_indices:
                    break  # 如果没有需要重新加权的索引，退出循环

                # 获取需要加权的值并计算总和
                values = [self.args.client_memeory_size[rnd][idx] for idx in need_reweight_indices]
                values_sum = sum(values)

                # 重新加权值
                values = [(x / values_sum) * task_server_all_cache_size for x in values]

                # 更新列表
                for index, new_value in zip(need_reweight_indices, values):
                    self.args.client_memeory_size[rnd][index] = new_value

                # 限制最大缓存大小
                self.args.client_memeory_size[rnd] = [min(x, self.args.client_max_cache_size) for x in
                                                 self.args.client_memeory_size[rnd]]

                # 更新 max_indices 和检查是否继续加权
                new_max_indices = [index for index, value in enumerate(self.args.client_memeory_size[rnd])
                                   if value == self.args.client_max_cache_size]
                is_go_on_reweight = [item for item in new_max_indices if item not in max_indices]

                # 如果没有需要重新加权的索引，退出循环
                if not is_go_on_reweight:
                    break

                # 更新服务器缓存大小
                task_server_all_cache_size -= len(new_max_indices) * self.args.client_max_cache_size
                max_indices = new_max_indices  # 更新 max_indices 以便下一次检查


def compute_l2_norm(model1_state_dict, model2_state_dict):
    # 初始化 L2 范数
    l2_norm = 0.0

    # 遍历两个模型的参数
    for key in model1_state_dict.keys():
        # 确保两个模型有相同的参数
        if key in model2_state_dict:
            # 获取参数
            param1 = model1_state_dict[key]
            param2 = model2_state_dict[key]

            # 计算参数的 L2 范数并累加
            l2_norm += torch.norm(param1 - param2, p=2).item() ** 2

    # 返回 L2 范数的平方根
    return torch.sqrt(torch.tensor(l2_norm))
