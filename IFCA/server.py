# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import enum
import os

from torch import device
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
# Now import FLamingo
from FLamingo.core.server import *
from models import create_model_instance_custom
import numpy as np
from torch.nn.init import uniform_, normal_


class IFCAClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        # IFCA
        self.cluster_id = None  # cluster id client belongs
        # self.cluster_stable = False  # cluster stable or not. If stable, only send cluster center to client
        # self.shared_params = None  # shared params for all clients
        # self.ifca_params = None
        self.cluster_loss = 0.0     # loss computed when selecting the cluster
        self.cluster_acc = 0.0      # acc computed when selecting the cluster
        # personalize
        self.acc_bf_train = 0.0
        self.loss_bf_train = 0.0
        self.acc_bf_train_opt = 0.0
        self.loss_bf_train_opt = 0.0
        self.acc_after_train = 0.0
        self.loss_after_train = 0.0
        self.acc_after_train_opt = 0.0
        self.loss_after_train_opt = 0.0
        self.train_sampels = 0
        self.train_loss = 0.0


class IFCAServer(Server):
    def init(self):
        self.model = create_model_instance_custom(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
    
    def print_results(self):
        acc_after_train = self.get_clients_attr_tolist('acc_after_train', self.selected_clients_idxes)
        loss_after_train = self.get_clients_attr_tolist('loss_after_train', self.selected_clients_idxes)
        train_samples = self.get_clients_attr_tolist('train_samples', self.selected_clients_idxes)
        train_loss = self.get_clients_attr_tolist('train_loss', self.selected_clients_idxes)
        cluster_loss = self.get_clients_attr_tolist('cluster_loss', self.selected_clients_idxes)
        cluster_acc = self.get_clients_attr_tolist('cluster_acc', self.selected_clients_idxes)
        self.log(f"cluster_loss: {np.mean(cluster_loss)}")
        self.log(f"cluster_acc: {np.mean(cluster_acc)}")
        self.log(f"acc_after_train: {np.mean(acc_after_train)}")
        self.log(f"loss_after_train: {np.mean(loss_after_train)}")
        self.log(f"train_samples: {np.mean(train_samples)}")
        self.log(f"train_loss: {np.mean(train_loss)}")
            
    # def check_cluster_stable(self):
    #     """Check if the cluster is stabled for clients
    #     """
    #     for id in self.selected_clients_idxes:
    #         cluster_list = self.client_cluster_tracker[id-1]
    #         # if the last five epochs, the cluster id of a client didn't change, then set client.cluster_stable = True
    #         if len(cluster_list) >= 5 and len(set(cluster_list[-5:])) == 1:
    #             self.get_client_by_rank(id).cluster_stable = True
            
    # def append_cluster_id(self):
    #     """After selecting clients, append the cluster id of selected client to self.client_cluster_tracker
    #     """
    #     for id in self.selected_clients_idxes:
    #         self.client_cluster_tracker[id-1].append(self.get_client_by_rank(id).cluster_id)

    def get_cur_cluster_status(self):
        cluster_clients = [[] for _ in range(self.K)]
        for rank in self.trainable_clients_idxes:
            client = self.get_client_by_rank(rank)
            # self.log(f"Client {rank} cluster id: {client.cluster_id}")
            cluster_clients[client.cluster_id].append(rank)
        for i in range(self.K):
            self.log(f"Cluster {i}: {cluster_clients[i]}")
        self.cluster_map = cluster_clients
            
    # def aggregate_shared(self):
    #     """Aggregate the shared weights
    #     """
    #     if self.share_weights_forall:
    #         updated_vec = torch.zeros_like(self.shared_params)
    #         total_samples = 0
    #         for client in self.selected_clients:
    #             total_samples += client.train_samples
    #         for client in self.selected_clients:
    #             updated_vec += client.train_samples/total_samples * client.shared_params
    #         self.shared_params = updated_vec
    
    def aggregate_ifca(self):
        """Aggregate the ifca weights for each cluster
        """
        for i in range(self.K):
            updated_vec = torch.zeros_like(self.cluster_params[i])
            total_samples = 0
            for client in self.selected_clients:
                if client.cluster_id == i:
                    total_samples += client.train_samples
            if total_samples == 0:
                continue
            for client in self.selected_clients:
                if client.cluster_id == i:
                    updated_vec += client.train_samples/total_samples * client.params
            self.cluster_params[i] = updated_vec

    def run(self):
        """
        Runs the IFCA server.
        Routine:
        1. init all clients
        2. randomly set K cluster centers. If share_weights_forall is True, then share some of the first layers globally.
        3. randomly assign clients to K clusters
        4. begin training
        - 4.1 select clients
        - 4.2 append cluster id to track cluster id changes
        - 4.3 check if the cluster is stable
        - 4.4 send models to clients
        - 4.5 listen to clients
        - 4.6 print results
        - 4.7 if weights are shared, then calculate the averaged shared parts
        - 4.8 finalize round
        """
        self.init_clients(clientObj=IFCAClientInfo)
        
        # if share weights:
        ## shared_params: shared parameter of 1 model
        ## ifca_params: a list of models for ifca_params
        # all_params = list(self.model.parameters())
        # all_params_vector = self.export_model_parameter()
        # last_two_layers_size = sum(p.numel() for p in all_params[-4:]) if self.share_weights_forall else 0
        # self.start_idx = all_params_vector.numel() - last_two_layers_size
        # self.end_idx = all_params_vector.numel()
                
        # randomly set different init weights for ifca trained params
        # model_params are lists of params of part for specific model
        # self.ifca_params_list = []
        # flatterned_two_layers = all_params_vector[self.start_idx:self.end_idx]
        # self.shared_params = all_params_vector[:self.start_idx].clone().detach()
        # for i in range(self.K):
        #     # 为每个元素生成新的随机初始化的向量
        #     new_params = torch.empty_like(flatterned_two_layers).uniform_(-0.5, 0.5).clone().detach()
        #     self.ifca_params_list.append(new_params)
    
        self.cluster_params = []
        for i in range(self.K):
            # K initial parameters
            temp_model = create_model_instance_custom(self.model_type, self.dataset_type)
            temp_model = temp_model.to(self.device)
            self.cluster_params.append(self.export_model_parameter(temp_model))
        del temp_model
        if self.dataset_type == 'cifar10':
            self.generate_global_test_set()  
        # randomly inital client cluster id
        # numbers = [i+1 for i in range(self.num_training_clients)]
        numbers = np.arange(1, self.num_trainable_clients+1)
        np.random.shuffle(numbers)
        self.cluster_map = np.array_split(numbers, self.K)
        for i, g in enumerate(self.cluster_map):
            for c in g:
                self.get_client_by_rank(c).cluster_id = i
        # self.get_cur_cluster_status()
        # keep track of all clients' cluster id in different epochs
        # self.client_cluster_tracker = [[] for _ in range(self.num_clients)]
        # begin
        while True:
            if self.dataset_type == 'cifar10':
                c_id = 0
                max_acc = 0.0
                min_loss = float('inf')
                for i in range(self.K):
                    self.set_model_parameter(self.cluster_params[i],self.model)
                    temp_test_dic = self.test(self.model, self.test_loader)
                    test_loss, test_acc = temp_test_dic['test_loss'], temp_test_dic['test_acc']
                    if test_loss <= min_loss:
                        max_acc = test_acc
                        min_loss = test_loss
                        c_id = i
                self.log(f"Global acc: {max_acc}, Global loss: {min_loss}, cluster: {c_id}")
            self.select_clients()
            # append cluster id
            # self.append_cluster_id()
            # send
            self.broadcast(data={
                'status': 'TRAINING',
                'cluster_params': self.cluster_params
            }
            )
            # for rank in self.selected_clients_idxes:
            #     client = self.get_client_by_rank(rank)
            #     # if need to save some bandwidth
            #     if self.send_only_cluster_after_stable is True and client.cluster_stable is True:
            #         self.network.send({
            #             'status':'TRAINING',
            #             'cluster_stable': True,
            #             'cluster_id': client.cluster_id, # 'cluster_id': 'cluster_center
            #             'shared_params': self.shared_params,
            #             'ifca_params': self.ifca_params_list[client.cluster_id],
            #             'global_round': self.global_round
            #         }, rank)
            #     # just send all the models
            #     else:
            #         self.network.send({
            #             'status':'TRAINING',
            #             'cluster_stable': False,
            #             'cluster_id': client.cluster_id, # 'cluster_id': 'cluster_center
            #             'shared_params': self.shared_params,
            #             'ifca_params_list': self.ifca_params_list,
            #             'global_round': self.global_round
            #         }, rank)
            self.listen()
            # for i in self.selected_clients:
            #     self.log(f"Received cluster ID for client {i.rank}: {i.cluster_id}")
            self.print_results()
            # self.log(self.client_cluster_tracker)
            # aggregate params for shared and ifca
            # self.aggregate_shared()
            # self.aggregate_ifca()
            self.aggregate_ifca()
            # send to eval
            self.broadcast(data={
                'status': 'EVAL',
                'cluster_params': self.cluster_params
            }, dest_ranks=self.eval_clients_idxes)
            # for rank in self.eval_clients_idxes:
            #     client = self.get_client_by_rank(rank)
            #     self.network.send({
            #         'status':'EVAL',
            #         'shared_params': self.shared_params,
            #         'ifca_params_list': self.ifca_params_list,
            #         'global_round': self.global_round
            #     }, rank)
            self.listen(self.eval_clients_idxes)
            self.average_client_info(self.eval_clients_idxes, attrs=['cluster_acc', 'cluster_loss', 'test_time'], dic_prefix='eval')
            # check cluster status
            self.get_cur_cluster_status()
            # check if the cluster is stable
            # self.check_cluster_stable()
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Server ends")
        self.stop_all()


if __name__ == "__main__":
    server = IFCAServer()
    server.run()
    