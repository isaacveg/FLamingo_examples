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


class FomoClient(Client):
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
        # FoMo setting
        self.M = min(self.M, self.num_training_clients)
        # self.P_matrix = torch.diag(torch.ones(self.num_training_clients, device=self.device))
        self.received_ids = []  # get models from these ranks
        self.params_list = []

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
  
    def weight_cal(self, val_loader):
        weight_list = []
        L = self.recalculate_loss(self.old_model, val_loader)
        old_params = self.export_model_parameter(self.old_model)
        for received_params in self.params_list:
            params_dif = received_params - old_params
            self.set_model_parameter(received_params, self.model)
            # small disturbance
            weight_list.append((L - self.recalculate_loss(self.model, val_loader)) / (torch.norm(params_dif) + 1e-5))
        self.weight_vector_update(weight_list)
        return torch.tensor(weight_list)
        
    def weight_vector_update(self, weight_list):
        self.weight_vector = np.zeros(self.num_clients)
        for w, id in zip(weight_list, self.received_ids):
            self.weight_vector[id-1] += w.item()
        self.weight_vector = torch.tensor(self.weight_vector, device=self.device)

    def recalculate_loss(self, new_model, val_loader):
        L = 0
        for x, y in val_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = new_model(x)
            loss = self.loss_func(output, y)
            L += loss.item()
        
        return L / len(val_loader)
  
    def weight_scale(self, weights):
        weights = torch.maximum(weights, torch.tensor(0))
        w_sum = torch.sum(weights)
        if w_sum > 0:
            weights = [w/w_sum for w in weights]
            return torch.tensor(weights)
        else:
            return torch.tensor([])
        
    def local_aggregation(self):
        weights = self.weight_scale(self.weight_cal(self.test_loader))
        if len(weights) > 0:
            self.aggregated_params = torch.zeros_like(self.export_model_parameter())
            for w, received_params in zip(weights, self.params_list):
                self.aggregated_params += w * received_params
            self.set_model_parameter(self.aggregated_params, self.model)
    
    def run(self):
        """
        Client jobs, usually have a loop
        """
        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                self.log('start training...')
                self.received_ids = data['send_ids']
                self.params_list = data['params_list']
                self.log(f"Received ids: {self.received_ids}")
                self.log(f"Received params: {len(self.params_list), self.params_list}")
                self.old_model = deepcopy(self.model)
                s_t = time.time()
                self.local_aggregation()
                f_t = time.time() - s_t
                self.log(f"Aggregation time: {f_t}")
                # self.model = self.set_model_parameter(self.local_aggregation)
                # test acc
                test_bf_train = self.test(self.old_model, self.test_loader)
                self.log(f"{test_bf_train}")
                # train
                s_t = time.time()
                train_dict = self.train_iters(
                    self.model, self.train_loader, self.loss_func, self.optimizer, self.lr_scheduler, self.local_iters)
                f_t = time.time() - s_t
                self.log(f"{train_dict}")
                self.log(f"train_time: {f_t}")
                # test after train
                test_after_train = self.test(self.model, self.test_loader)
                self.log(f"{test_after_train}")
                self.lr_scheduler.step()
                # send back to server
                send_data = {
                    'weight_vector': self.weight_vector,
                    'acc_bf_train': test_bf_train['test_acc'],
                    'loss_bf_train': test_bf_train['test_loss'],
                    # 'acc_bf_train_opt': test_bf_train_opt['test_acc'],
                    # 'loss_bf_train_opt': test_bf_train_opt['test_loss'],
                    'acc_after_train': test_after_train['test_acc'],
                    'loss_after_train': test_after_train['test_loss'],
                    # 'acc_after_train_opt': test_after_train_opt['test_acc'],
                    # 'loss_after_train_opt': test_after_train_opt['test_loss'],
                    'params': self.export_model_parameter(self.model),
                    'train_samples': train_dict['train_samples'],
                    'train_loss': train_dict['train_loss'],
                    'train_time': train_dict['train_time']*4,   # Fomo clients use approximately 3 training time to aggregate and total 4 times
                    'send_time': self.rand_time(self.communication, self.dynamics) * 10
                    # 'single_step_time': self.single_step_time,
                    # 'send_time': self.send_time,
                }
                self.send(send_data, 0)
                self.log(f"Weight vector: {self.weight_vector}")
            # elif data['status'] == 'EVAL':
            #     self.log(f"")
            elif data['status'] == 'STOP':
                self.log('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        self.log("Stop and exit")


if __name__ == '__main__':
    client = FomoClient()
    client.run()