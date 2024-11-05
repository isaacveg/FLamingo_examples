# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 8)

import sys
# sys.path.append(".")
# sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

# Now import FLamingo
from FLamingo.core.client import *
from FLamingo.core.utils.train_test_utils import infinite_dataloader
from models import create_model_instance_custom


class IFCAClient(Client):
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
        lambda_lr = lambda step: max(0.005, self.lr_decay ** step)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        # ifca settings
        self.cluster_id = self.rank % self.K
        self.cluster_params = []
        # self.shared_params = None
        # self.ifca_params_list = []
        # self.ifca_params = None
    
    # def reduce_lr(self):
    #     self.lr = max(self.lr * 0.993, 0.005)
    def rand_send(self):
        if self.model_type == 'alexnet':
            return self.rand_time(self.communication, self.dynamics)*14.6
        elif self.model_type == 'fedavgcnn':
            return self.rand_time(self.communication, self.dynamics)*2.3
        else:
            return self.rand_time(self.communication, self.dynamics)*3.1
        
    def run(self):
        """
        Client jobs, usually have a loop
        """
        # all_params = list(self.model.parameters())
        # all_params_vector = self.export_model_parameter()
        # last_two_layers_size = sum(p.numel() for p in all_params[-4:]) if self.share_weights_forall else 0
        # self.start_idx = all_params_vector.numel() - last_two_layers_size
        # self.end_idx = all_params_vector.numel()
        # del all_params, all_params_vector, last_two_layers_size
        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                self.log('start training...')
                # self.cluster_id = data['cluster_id']
                self.cluster_params = data['cluster_params']
                # self.shared_params = data['shared_params']
                # self.cluster_stable = data['cluster_stable']
                # if self.cluster_stable:
                #     self.ifca_params = data['ifca_params']
                #     self.set_model_parameter(
                #         torch.cat((self.shared_params, self.ifca_params))
                #     )
                # else:
                # self.ifca_params_list = data['ifca_params_list']
                # self.ifca_params = self.ifca_params_list[self.cluster_id]
                # need to define which cluster to use, by testing loss using each cluster model
                # and choose the best one
                minimum_loss = float('inf')
                cluster_acc = 0.0
                self.test_time = time.time()
                for i in range(self.K):
                    # self.set_model_parameter(
                    #     torch.cat((self.shared_params, self.ifca_params_list[i]))
                    # )
                    self.set_model_parameter(self.cluster_params[i], self.model)
                    temp_test_dic = self.test(self.model, self.test_loader)
                    test_loss, test_acc = temp_test_dic['test_loss'], temp_test_dic['test_acc']
                    self.log(f"Cluster {i} test loss: {test_loss}, test acc: {test_acc}")
                    if test_loss <= minimum_loss:
                        minimum_loss = test_loss
                        self.cluster_id = i
                        # self.ifca_params = self.ifca_params_list[i]
                        cluster_acc = test_acc
                self.test_time = time.time() - self.test_time
                self.log(f"Choose: {self.cluster_id} loss: {minimum_loss} acc: {cluster_acc}, test time: {self.test_time}")
                self.set_model_parameter(self.cluster_params[self.cluster_id], self.model)
                # self.set_model_parameter(
                #     torch.cat((self.shared_params, self.ifca_params))
                # )
                # train
                # train_dict = self.train_iters(self.model, self.train_loader, self.local_epochs, self.loss_func, self.optimizer, self.lr_scheduler)
                train_dict = self.train_iters(self.model, self.train_loader, self.loss_func, self.optimizer, iters=self.local_iters)
                self.lr_scheduler.step()
                self.log(f"{train_dict}")
                # test after train
                test_after_train = self.test(self.model, self.test_loader)
                self.log(f"{test_after_train}")
                # send back to server
                # self.shared_params = self.export_model_parameter()[:self.start_idx].clone().detach()
                # self.ifca_params = self.export_model_parameter()[self.start_idx:self.end_idx].clone().detach()
                send_data = {
                    'cluster_id': self.cluster_id,
                    'cluster_loss': minimum_loss,
                    'cluster_acc': cluster_acc,
                    'acc_after_train': test_after_train['test_acc'],
                    'loss_after_train': test_after_train['test_loss'],
                    # 'shared_params': self.shared_params,
                    # 'ifca_params': self.ifca_params,
                    'params': self.export_model_parameter(),
                    'train_samples': train_dict['train_samples'],
                    'train_loss': train_dict['train_loss'],
                    'train_time': train_dict['train_time']+self.test_time,
                    'send_time': self.rand_send()*self.K # Need to send cluster models here 
                }
                self.send(send_data, 0)
                # self.reduce_lr()
                # self.lr = max(self.lr * 0.993, 0.005)
                # self.lr_scheduler
            elif data['status'] == 'EVAL':
                # test on all cluster models
                # self.shared_params = data['shared_params']
                # self.ifca_params_list = data['ifca_params_list']
                self.cluster_params = data['cluster_params']
                minimum_loss = float('inf')
                cluster_acc = 0.0
                self.test_time = time.time()
                for i in range(self.K):
                    # self.set_model_parameter(
                    #     torch.cat((self.shared_params, self.ifca_params_list[i]))
                    # )
                    self.set_model_parameter(self.cluster_params[i], self.model)
                    temp_test_dic = self.test(self.model, self.test_loader)
                    test_loss, test_acc = temp_test_dic['test_loss'], temp_test_dic['test_acc']
                    self.log(f"Cluster {i} test loss: {test_loss}, test acc: {test_acc}")
                    if test_loss <= minimum_loss:
                        minimum_loss = test_loss
                        self.cluster_id = i
                        cluster_acc = test_acc
                self.test_time = time.time() - self.test_time
                self.log(f"Choose: {self.cluster_id} loss: {minimum_loss} acc: {cluster_acc}, test time: {self.test_time}")
                self.send(data={
                    'cluster_id': self.cluster_id,
                    'cluster_loss': minimum_loss,
                    'cluster_acc': cluster_acc,
                    'test_time': self.test_time
                })
            elif data['status'] == 'STOP':
                self.log('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        self.log("Stop and exit")


if __name__ == '__main__':
    client = IFCAClient()
    client.run()