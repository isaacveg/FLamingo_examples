# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
# Now import FLamingo
from FLamingo.core.server import *
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


class ProxClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        self.bf_acc = 0.0
        self.bf_loss = 0.0
        self.is_straggler = False
        self.local_epochs = 10

class ProxServer(Server):
    def generate_global_test_set(self):
        """
        Generate a global test set.
        """
        if self.dataset_type == 'mnist':
            self.test_set = MNIST(root='../datasets', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif self.dataset_type == 'cifar10':
            self.test_set = CIFAR10(root='../datasets', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.test_loader = DataLoader(
            dataset=self.test_set, batch_size=self.args.batch_size, shuffle=False)
        
    def test(self, model, test_loader):
        """
        Test the model.
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        self.log(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
        return {'test_loss': test_loss, 'accuracy': accuracy}
    
    def average_client_info(self, client_list):
        """
        Average client info and log it
        """
        length = len(client_list)
        clients = [self.get_client_by_rank(rank) for rank in client_list]
        avg_train_loss = 0.0
        avg_test_loss, avg_test_acc = 0.0, 0.0
        avg_bf_test_acc, avg_bf_test_loss = 0.0, 0.0
        for client in clients:
            avg_train_loss += client.train_loss
            avg_test_acc += client.test_acc
            avg_test_loss += client.test_loss
            avg_bf_test_acc += client.bf_acc 
            avg_bf_test_loss += client.bf_loss
        self.log(f"Avg global info:\n train loss {avg_train_loss/length}, \
                \ntest acc {avg_test_acc/length}, \
                \ntest loss {avg_test_loss/length},\
                \navg_bf_test_acc {avg_bf_test_acc/length}, \
                \navg_bf_test_loss {avg_bf_test_loss/length} ")
        
    def get_and_set_stragglers(self):
        """
        Generate a list of stagglers according to config.stragglers.
        """
        self.stragglers_num = int(self.stragglers * self.num_training_clients)
        # print(self.stragglers_num)
        self.stragglers_idxes = random.sample(self.selected_clients_idxes, k=self.stragglers_num)
        # print(self.stragglers_idxes, self.size)
        self.non_stragglers_idxes = list(set(self.selected_clients_idxes) - set(self.stragglers_idxes))
        for rank in self.stragglers_idxes:
            self.get_client_by_rank(rank).is_straggler = True
            self.get_client_by_rank(rank).local_epochs = random.choice(range(1, self.local_epochs+1))
            self.log(f"Client {rank} gets {self.get_client_by_rank(rank).local_epochs}")
        for rank in self.non_stragglers_idxes:
            self.get_client_by_rank(rank).is_straggler = False
            self.get_client_by_rank(rank).local_epochs = self.local_epochs
        self.log(f"stragglers: {self.stragglers_num}")
        self.log(f"stragglers_idxes: {self.stragglers_idxes}")
        self.log(f"non_stragglers_idxes: {self.non_stragglers_idxes}")
        return self.stragglers_idxes, self.non_stragglers_idxes
        

    def run(self):
        """
        Basically init client, 
        select, broadcast, listen, aggregate 
        and wait for next round
        """
        # print(self.size,flush=True)
        self.init_clients(clientObj=ProxClientInfo)
        self.generate_global_test_set()
        # print(os.environ)
        while True:
            self.select_clients()
            self.get_and_set_stragglers()
            # Non stragglers
            self.broadcast(
                data={
                    'status':'TRAINING',
                    'params':self.export_model_parameter(),
                    'local_epochs':self.local_epochs,
                    'straggler': False},
                dest_ranks=self.non_stragglers_idxes
                )
            # stragglers
            for rank in self.stragglers_idxes:
                self.network.send(
                    data={
                        'status':'TRAINING',
                        'params':self.export_model_parameter(),
                        'local_epochs':self.get_client_by_rank(rank).local_epochs,
                        'straggler': True
                    },
                    dest_rank=rank
                )
            self.listen()
            self.aggregate()
            self.test(self.model, self.test_loader)
            self.average_client_info(self.selected_clients_idxes)
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Yes, just end your job")
        self.stop_all()


if __name__ == "__main__":
    server = ProxServer()
    server.run()