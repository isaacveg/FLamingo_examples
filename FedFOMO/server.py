# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os

from torch import device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
# Now import FLamingo
from FLamingo.core.server import *
from models import create_model_instance_custom
import numpy as np
import copy


class FomoClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        # Fomo
        self.send_ids = []
        self.weight_vector = None
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


class FomoServer(Server):
    def init(self):
        self.model = create_model_instance_custom(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        # FoMo setting
        self.M = min(self.M, self.num_training_clients)
        self.P_matrix = torch.diag(torch.ones(self.num_clients, device=self.device))
        # self.Eval_P_matrix

    
    def reduce_lr(self):
        self.alpha *= 0.993
        self.beta *= 0.993
    
    def print_results(self):
        acc_bf_train = self.get_clients_attr_tolist('acc_bf_train', self.selected_clients_idxes)
        loss_bf_train = self.get_clients_attr_tolist('loss_bf_train', self.selected_clients_idxes)
        # acc_bf_train_opt = self.get_clients_attr_tolist('acc_bf_train_opt', self.selected_clients_idxes)
        # loss_bf_train_opt = self.get_clients_attr_tolist('loss_bf_train_opt', self.selected_clients_idxes)
        acc_after_train = self.get_clients_attr_tolist('acc_after_train', self.selected_clients_idxes)
        loss_after_train = self.get_clients_attr_tolist('loss_after_train', self.selected_clients_idxes)
        # acc_after_train_opt = self.get_clients_attr_tolist('acc_after_train_opt', self.selected_clients_idxes)
        # loss_after_train_opt = self.get_clients_attr_tolist('loss_after_train_opt', self.selected_clients_idxes)
        train_samples = self.get_clients_attr_tolist('train_samples', self.selected_clients_idxes)
        train_loss = self.get_clients_attr_tolist('train_loss', self.selected_clients_idxes)
        self.log(f"acc_bf_train: {np.mean(acc_bf_train)}")
        self.log(f"loss_bf_train: {np.mean(loss_bf_train)}")
        # self.log(f"acc_bf_train_opt: {np.mean(acc_bf_train_opt)}")
        # self.log(f"loss_bf_train_opt: {np.mean(loss_bf_train_opt)}")
        self.log(f"acc_after_train: {np.mean(acc_after_train)}")
        self.log(f"loss_after_train: {np.mean(loss_after_train)}")
        # self.log(f"acc_after_train_opt: {np.mean(acc_after_train_opt)}")
        # self.log(f"loss_after_train_opt: {np.mean(loss_after_train_opt)}")
        self.log(f"train_samples: {np.mean(train_samples)}")
        self.log(f"train_loss: {np.mean(train_loss)}")

    def choose_send_models(self, clients_list=None):
        # send topk models to clients
        clients_list = clients_list or self.selected_clients_idxes
        for rank in clients_list:
            # rank 1~num_clients, need minus 1
            ## Choose top k models to send
            client = self.get_client_by_rank(rank)
            indices = torch.topk(self.P_matrix[rank-1], self.M).indices.tolist()
            client.send_ids = [i+1 for i in indices]
            self.log(f"Send model from {indices} to client {rank}")

    def run(self):
        """
        Runs the FedFomo server
        """
        self.init_clients(clientObj=FomoClientInfo)
        # set inital params
        for client in self.all_clients:
            client.params = self.export_model_parameter()
        # if self.dataset_type == 'cifar10':
        #     self.generate_global_test_set() 
        while True:
            # if self.dataset_type == 'cifar10':
            #     test_dic = self.test(self.model, self.test_loader, self.loss_func, self.device)
            #     self.log(f"Global acc: {test_dic['test_acc']}, global loss: {test_dic['test_loss']}")
            self.select_clients()
            # send
            self.choose_send_models(self.selected_clients_idxes)
            for rank in self.selected_clients_idxes:
                # send a list of clients params instead of single model param
                client = self.get_client_by_rank(rank)
                self.network.send({
                    'status':'TRAINING',
                    'send_ids': client.send_ids,
                    'params_list': [self.get_client_by_rank(i).params for i in client.send_ids],
                    'global_round': self.global_round,
                }, rank)
            self.listen()
            for i in self.selected_clients_idxes:
                # self.log(f"Received weight vec for client {i}: {self.get_client_by_rank(i).weight_vector}")
                self.P_matrix[i-1] += self.get_client_by_rank(i).weight_vector
            self.print_results()
            # self.log(self.P_matrix)
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Server ends")
        self.stop_all()


if __name__ == "__main__":
    server = FomoServer()
    server.run()
    