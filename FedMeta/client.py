import os
from venv import create
from mpi4py import MPI

# from FLamingo.core.utils.model_utils import create_model_instance
WRLD = MPI.COMM_WORLD
RANK = WRLD.Get_rank()
os.environ['CUDA_VISIBLE_DEVICES'] = str(RANK%4)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# import FLamingo after setting cuda information
import sys
sys.path.append('../FLamingo/')

from FLamingo.core.client import *
from FLamingo.core.utils.chores import merge_several_dicts
from model import *


class MetaClient(Client):
    """
    Your own Client
    """
    def init(self):
        # self.model = CNNMNIST() if self.dataset_type == 'mnist' else AlexNet()
        self.model = create_model_instance(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)

    def reptile_train(self, model, optimizer, loss_func, beta, steps, dataloader, local_epoch):
        """
        Reptile training using given functions
        """
        model.train()
        model.to(self.device)
        loss = 0.0
        samples = 0
        for ep in range(local_epoch):
            dataiter = iter(dataloader)
            for idx in range(len(dataloader)//steps):
                original_model_vector = self.export_model_parameter(model)
                for st in range(steps):
                    data, label = next(dataiter)
                    batch_num, batch_loss = self._train_one_batch(model, data, label, optimizer, loss_func)
                    samples += batch_num
                    loss += batch_loss * batch_num
                new_vec = self.export_model_parameter(model)
                original_model_vector = self.update_model(original_model_vector, new_vec, beta)
                self.set_model_parameter(original_model_vector, model)
        loss /= samples
        return {'train_loss':loss, 'train_samples':samples}

    def local_optimization(self, model, dataloader, steps=None):
        """
        Local optimization for meta-model
        """
        if steps is None:
            steps = len(dataloader)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_func = torch.nn.CrossEntropyLoss()
        num, loss = 0, 0.0
        for idx, (data, target) in enumerate(dataloader):
            batch_num, batch_loss = self._train_one_batch(model, data, target, optimizer, loss_func)
            num += batch_num
            loss += batch_loss * batch_num
        loss /= num
        return {'local_optimization_loss':loss, 'local_optimization_samples':num}
    
    def update_model(self, original, after, lr):
        """
        Update original model according to trained model
        """
        delta = after - original
        original += lr * delta
        return original

    def run(self):
        """
        Client jobs, usually have a loop
        """
        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                self.set_model_parameter(data['params'])
                # Using Reptile to train
                info_dict = self.reptile_train(
                    self.model, self.optimizer, self.loss_func, self.beta, self.steps, 
                    self.train_loader, self.local_epochs
                    )
                # Update beta to lower outer lr
                self.beta *= 0.993
                # Test before local optimization
                bf_test_dic = self.test(self.model, self.test_loader, self.loss_func, self.device)
                params = self.export_model_parameter()
                # Local optimization
                optim_dic = self.local_optimization(self.model, self.train_loader, self.optim_steps)
                # Test after local optimization
                test_dict = self.test(self.model, self.test_loader, self.loss_func, self.device)
                # self.log(f"train: {info_dict}\noptim: {optim_dic}\ntest: {test_dict}")
                send_dic = merge_several_dicts([info_dict, optim_dic, test_dict])
                send_dic.update({'before_test_acc':bf_test_dic['test_acc'],
                                 'before_test_loss':bf_test_dic['test_loss'],
                                 'params': params}
                                 )
                ## send_dic includes: 
                # train_loss, train_samples, local_optimization_loss, local_optimization_samples, 
                # test_loss, test_acc, before_test_loss, before_test_acc, params
                # log them (except params)
                self.log(f"train loss {send_dic['train_loss']}, \
                        \nlocal optimization loss {send_dic['local_optimization_loss']}, \
                        \ntest loss {send_dic['test_loss']}, \
                        \nbefore test loss {send_dic['before_test_loss']},\
                        \nbefore test acc {send_dic['before_test_acc']}, \
                        \nlocal optimization samples {send_dic['local_optimization_samples']}\
                        \ntrain samples {send_dic['train_samples']}\
                        \ntest acc {send_dic['test_acc']}")
                self.send(send_dic)
                
            elif data['status'] == 'STOP':
                self.log('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        self.log('stopped')


if __name__ == '__main__':
    client = MetaClient()
    client.run()