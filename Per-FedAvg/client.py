# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 8)
os.environ['CUDA_VISIBLE_DEVICES'] = str( 4 + (rank % 4) )

import sys
# sys.path.append(".")
# sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

# Now import FLamingo
from FLamingo.core.client import *
from FLamingo.core.utils.train_test_utils import infinite_dataloader
from models import create_model_instance_custom
from perfedavg_hf import train as pfa_train_hf
from perfedavg_fo import train as pfa_train_fo

class PerClient(Client):
    """
    Your own Client
    """
    def init(self):
        """
        Init model and network to enable customize these parts.   
        """
        self.network = NetworkHandler()
        # 魔改版建立模型，只导入需要的内容
        self.model = create_model_instance_custom(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def rand_send(self):
        if self.model_type == 'alexnet':
            return self.rand_time(self.communication, self.dynamics)*14.6
        elif self.model_type == 'fedavgcnn':
            return self.rand_time(self.communication, self.dynamics)*2.3
        else:
            return self.rand_time(self.communication, self.dynamics)*3.1

    def meta_train(self):
        """
        Training model use PerFedAvg-HF or FO, hessian free or first order.
        """
        # train_loader = infinite_dataloader(self.train_loader)
        train_loader = self.train_loader
        if self.method == 'hf':
            info_dic = pfa_train_hf(self.model, train_loader,
                                self.alpha, self.beta, self.local_iters, self.local_epochs,
                                self.device, self.model_type)
            train_time = info_dic['batches_num']*self.rand_comp() if self.USE_SIM_SYSHET else info_dic['train_time']
            return {'train_loss': info_dic['train_loss'], 
                    'grad_loss': info_dic['grad_loss'], 
                    'grad2_loss': info_dic['grad2_loss'], 
                    'train_time': train_time,
                    'batches_num': info_dic['batches_num'],
                    'params': info_dic['params'],
                    'train_samples': info_dic['samples_num']}
        else:
            info_dic = pfa_train_fo(self.model, train_loader,
                                self.alpha, self.beta, self.local_iters, self.local_epochs,
                                self.device, self.model_type)
            train_time = info_dic['batches_num']*self.rand_comp() if self.USE_SIM_SYSHET else info_dic['train_time']
            return {'train_loss': info_dic['train_loss'], 
                    'grad_loss': info_dic['grad_loss'], 
                    'train_time': info_dic['train_time'], 
                    'batches_num': info_dic['batches_num'],
                    'params': info_dic['params'],
                    'train_samples': info_dic['samples_num']}

    def local_optimization(self, model, dataloader, optim_steps=None):
        """
        Local optimization for meta-model.
        
        Args:
            - model: model to be optimized
            - dataloader: dataloader for training
            - optim_steps: number of steps to train
        
        Returns:
            - dict: containing optim_loss, optim_samples, optim_time
        """
        if optim_steps is None:
            optim_steps = len(dataloader)
        model.train()
        model = model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_func = torch.nn.CrossEntropyLoss()
        num, loss = 0, 0.0
        s_t = time.time()
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            batch_num, batch_loss = self._train_one_batch(model, data, target, optimizer, loss_func)
            num += batch_num
            loss += batch_loss * batch_num
            if idx+1 >= optim_steps:
                break
        optim_time = time.time() - s_t
        loss /= num
        return {'optim_loss':loss, 'optim_samples':num, 'optim_time':optim_time}
    
    def run(self):
        """
        Client jobs, usually have a loop
        """
        while True:
            # get from server
            data = self.network.get(0)
            if data['status'] == 'TRAINING':
                self.global_round = data['global_round']
                # self.alpha = data['alpha']
                # self.beta = data['beta']
                self.log('start training...')
                self.set_model_parameter(data['params'])
                # Test local acc
                params = self.export_model_parameter()
                test_from_server = self.test(self.model, self.test_loader)
                self.log(f"test_from_server: {test_from_server}")
                # Optim
                opt_from_server = self.local_optimization(self.model, self.train_loader, self.optim_steps)
                self.log(f"opt_from_server: {opt_from_server}")
                # Test local acc after optim
                test_from_server_opt = self.test(self.model, self.test_loader)
                self.log(f"test_from_server_opt: {test_from_server_opt}")
                ## Original model for meta-train
                self.set_model_parameter(params)
                info_dict = self.meta_train()
                ## Test local acc after meta-train
                params = self.export_model_parameter()
                test_acc_af_meta = self.test(self.model, self.test_loader)
                self.log(f"test_acc_af_meta: {test_acc_af_meta}")
                ## Test local acc after train and optim
                optim_af = self.local_optimization(self.model, self.train_loader, self.optim_steps)
                test_after_train_opt = self.test(self.model, self.test_loader)
                self.log(f"test_after_train_opt: {test_after_train_opt}")
                # send back to server
                if self.USE_SIM_SYSHET:
                    send_time = self.rand_send()
                send_data = {
                    'acc_bf_train': test_from_server['test_acc'],
                    'loss_bf_train': test_from_server['test_loss'],
                    'acc_bf_train_opt': test_from_server_opt['test_acc'],
                    'loss_bf_train_opt': test_from_server_opt['test_loss'],
                    'acc_after_train': test_acc_af_meta['test_acc'],
                    'loss_after_train': test_acc_af_meta['test_loss'],
                    'acc_after_train_opt': test_after_train_opt['test_acc'],
                    'loss_after_train_opt': test_after_train_opt['test_loss'],
                    'params': params,
                    'train_samples': info_dict['train_samples'],
                    'train_loss': info_dict['train_loss'],
                    'train_time': info_dict['train_time'],
                    'send_time': send_time
                }
                self.send(send_data, 0)
                self.alpha = max(self.alpha * self.lr_decay, 0.005)
                self.beta = max(self.beta * self.lr_decay, 0.005)
            elif data['status'] == 'EVAL':
                self.set_model_parameter(data['params'])
                test_from_server = self.test(self.model, self.test_loader)
                self.log(f"test_from_server: {test_from_server}")
                # Optim
                opt_from_server = self.local_optimization(self.model, self.train_loader, self.eval_optim_steps)
                self.log(f"opt_from_server: {opt_from_server}")
                # Test local acc after optim
                test_from_server_opt = self.test(self.model, self.test_loader)
                self.log(f"test_from_server_opt: {test_from_server_opt}")
                self.send(data={
                    'acc_bf_train': test_from_server['test_acc'],
                    'loss_bf_train': test_from_server['test_loss'],
                    'acc_bf_train_opt': test_from_server_opt['test_acc'],
                    'loss_bf_train_opt': test_from_server_opt['test_loss']
                })
            elif data['status'] == 'STOP':
                self.log('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        self.log("Stop and exit")


if __name__ == '__main__':
    client = PerClient()
    client.run()