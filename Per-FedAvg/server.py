# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
# Now import FLamingo
from FLamingo.core.server import *
from models import create_model_instance_custom
import numpy as np


class PerClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
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


class PerServer(Server):
    def init(self):
        self.model = create_model_instance_custom(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
    
    def reduce_lr(self):
        self.alpha *= 0.993
        self.beta *= 0.993
        
    def finalize_round(self):
        self.reduce_lr()
        return super().finalize_round()
    
    def print_results(self):
        """
        Will print the average results of the following things in self.selected_clients:
        'acc_bf_train'
        'loss_bf_train'
        'acc_bf_train_opt'
        'loss_bf_train_opt'
        'acc_after_train'
        'loss_after_train'
        'acc_after_train_opt'
        'loss_after_train_opt'
        'train_samples'
        'train_loss'
        """
        acc_bf_train = self.get_clients_attr_tolist('acc_bf_train', self.selected_clients_idxes)
        loss_bf_train = self.get_clients_attr_tolist('loss_bf_train', self.selected_clients_idxes)
        acc_bf_train_opt = self.get_clients_attr_tolist('acc_bf_train_opt', self.selected_clients_idxes)
        loss_bf_train_opt = self.get_clients_attr_tolist('loss_bf_train_opt', self.selected_clients_idxes)
        acc_after_train = self.get_clients_attr_tolist('acc_after_train', self.selected_clients_idxes)
        loss_after_train = self.get_clients_attr_tolist('loss_after_train', self.selected_clients_idxes)
        acc_after_train_opt = self.get_clients_attr_tolist('acc_after_train_opt', self.selected_clients_idxes)
        loss_after_train_opt = self.get_clients_attr_tolist('loss_after_train_opt', self.selected_clients_idxes)
        train_samples = self.get_clients_attr_tolist('train_samples', self.selected_clients_idxes)
        train_loss = self.get_clients_attr_tolist('train_loss', self.selected_clients_idxes)
        self.log(f"acc_bf_train: {np.mean(acc_bf_train)}")
        self.log(f"loss_bf_train: {np.mean(loss_bf_train)}")
        self.log(f"acc_bf_train_opt: {np.mean(acc_bf_train_opt)}")
        self.log(f"loss_bf_train_opt: {np.mean(loss_bf_train_opt)}")
        self.log(f"acc_after_train: {np.mean(acc_after_train)}")
        self.log(f"loss_after_train: {np.mean(loss_after_train)}")
        self.log(f"acc_after_train_opt: {np.mean(acc_after_train_opt)}")
        self.log(f"loss_after_train_opt: {np.mean(loss_after_train_opt)}")
        self.log(f"train_samples: {np.mean(train_samples)}")
        self.log(f"train_loss: {np.mean(train_loss)}")

    def run(self):
        """
        Basically init client, 
        select, broadcast, listen, aggregate 
        and wait for next round
        """
        self.init_clients(clientObj=PerClientInfo)
        if self.dataset_type == 'cifar10':
            self.generate_global_test_set()
        while True:
            # self.round_start_time = time.time()
            self.select_clients()
            self.broadcast(data={'status':'TRAINING',
                                 'params': self.export_model_parameter(),
                                 'alpha':self.alpha,
                                 'beta':self.beta})
            self.listen()
            self.aggregate(weight_by_sample=True)
            if self.dataset_type == 'cifar10':
                test_dic = self.test(self.model, self.test_loader, self.loss_func)
                self.log(f"global_acc: {test_dic['test_acc']:.4f}, global_loss: {test_dic['test_loss']:.4f}")
            self.print_results()
            self.evaluate_on_new_clients()
            self.average_client_info(
                self.eval_clients_idxes, 
                attrs=['acc_bf_train', 'loss_bf_train', 'acc_bf_train_opt', 'loss_bf_train_opt'],
                dic_prefix='eval')
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Server Stopped.")
        self.stop_all()


if __name__ == "__main__":
    server = PerServer()
    server.run()
    