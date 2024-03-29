# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

# Now import FLamingo
from FLamingo.core.client import *
import torch.nn.functional as F
import copy


class prox_loss(torch.nn.Module):
    """
    FedProx loss function
    """
    def __init__(self, mu):
        """
        :param mu: 正则化系数
        """
        super(prox_loss, self).__init__()
        self.mu = mu
    
    def forward(self, outputs, targets, local_model_params, global_model_params):
        """
        :param outputs: 模型输出
        :param targets: 目标值
        :param local_model_params: 当前模型参数
        :param global_model_params: 从服务器接收的全局模型参数
        """
        # 基本的交叉熵损失
        cross_entropy_loss = F.cross_entropy(outputs, targets)
        # FedProx 近似项
        proximal_term = 0.0
        for w, w_star in zip(local_model_params, global_model_params):
            proximal_term += (w - w_star).norm(2)
        proximal_loss = (self.mu / 2.0) * proximal_term
        # 总损失
        total_loss = cross_entropy_loss + proximal_loss
        return total_loss
    

class FedProxClient(Client):
    """
    FedProx Client.
    New defined:
        prox_loss: modified loss for FedProx, proximal term.
        capacity: save models not trained well.
    """
    def init(self):
        self.model = create_model_instance(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.loss_func = prox_loss(self.mu)
        self.test_loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        
    def train(self):
        """
        Use costumized loss func to train.
        """
        model, train_loader = self.model, self.train_loader
        global_model = copy.deepcopy(model)
        model.train()
        epoch_loss, num_samples = 0.0, 0
        for epoch in range(self.local_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = model(data)
                loss = self.loss_func(output, target, model.parameters(),global_model.parameters())
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return {'train_loss': epoch_loss / num_samples, 'train_samples': num_samples}
            
    def run(self):
        """
        Client jobs.
        """
        while True:
            # get from server
            # self.log(f"size: {self.size}")
            data = self.listen()
            if data['status'] == 'TRAINING':
                # self.log(f"size: {self.size}")
                self.set_model_parameter(data['params'], self.model)
                self.status = data['status']
                self.local_epochs = data['local_epochs']
                self.straggler = data['straggler']
                self.log(f"I'm {'straggler' if self.straggler else 'not straggler'}, local epochs {self.local_epochs}")
                bf_test_dic = self.test(
                    self.model, self.test_loader, self.test_loss_func, self.device)
                train_dic = self.train()
                # self.log("Training finish.")
                af_test_dic = bf_test_dic = self.test(
                    self.model, self.test_loader, self.test_loss_func, self.device)
                data_to_send = merge_several_dicts(
                    [af_test_dic, train_dic]
                )
                data_to_send.update({
                    'bf_acc': bf_test_dic['test_acc'],
                    'bf_loss': bf_test_dic['test_loss'],
                    'params': self.export_model_parameter()
                })
                self.send(data_to_send)
                # self.log("Send success")
            elif data['status'] == 'STOP':
                print('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        print('stopped')


if __name__ == '__main__':
    client = FedProxClient()
    client.run()