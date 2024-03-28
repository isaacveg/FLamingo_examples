import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'


# Set cuda information before importing FLamingo
import sys
sys.path.append('../FLamingo/')
from FLamingo.core.server import *
from model import CNNMNIST, AlexNet
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class MetaClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        # define your own information
        self.local_optimization_loss = 0.0
        self.local_optimization_samples = 0
        self.before_test_acc = 0.0
        self.before_test_loss = 0.0
        

class MetaServer(Server):
    def init(self):
        """
        Defining model and related information
        """
        self.network = NetworkHandler()
        # self.model = CNNMNIST() if self.dataset_type == 'mnist' else AlexNet()
        # self.model_type = 'cnn' if self.dataset_type == 'mnist' else 'alexnet'
        self.model = create_model_instance(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        
    def generate_global_test_set(self):
        """
        Generate a global test set.
        """
        if self.dataset_type == 'mnist':
            self.test_set = MNIST(root='../datasets/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif self.dataset_type == 'cifar10':
            self.test_set = CIFAR10(root='../datasets/', train=False, download=True, transform=transforms.Compose([
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
        # clients = [self.get_client_by_rank(rank) for rank in client_list]
        clients = [self.get_client_by_rank(rank) for rank in client_list]
        # train_loss, train_samples, local_optimization_loss, local_optimization_samples, 
        # test_loss, test_acc, before_test_loss, before_test_acc, params
        avg_train_loss, avg_train_samples = 0.0, 0.0
        avg_test_loss, avg_test_acc = 0.0, 0.0
        avg_before_test_loss, avg_before_test_acc = 0.0, 0.0
        avg_local_optimization_loss, avg_local_optimization_samples = 0.0, 0
        for client in clients:
            avg_train_loss += client.train_loss
            avg_test_acc += client.test_acc
            avg_test_loss += client.test_loss
            avg_before_test_acc += client.before_test_acc
            avg_before_test_loss += client.before_test_loss
            avg_local_optimization_loss += client.local_optimization_loss
            avg_local_optimization_samples += client.local_optimization_samples
        self.log(f"Avg global info:\ntrain loss {avg_train_loss/length}, \
                 \ntest acc {avg_test_acc/length}, \
                 \ntest loss {avg_test_loss/length}, \
                 \nbefore test acc {avg_before_test_acc/length}, \
                 \nbefore test loss {avg_before_test_loss/length}, \
                 \nlocal optimization loss {avg_local_optimization_loss/length}, \
                 \nlocal optimization samples {avg_local_optimization_samples/length}")

    def run(self):
        self.print_model_info()
        self.init_clients(clientObj=MetaClientInfo)
        self.generate_global_test_set()
        while True:
            """
            Server acts the same as before
            """
            self.select_clients()
            self.broadcast(data={
                'status':'TRAINING',
                'params': self.export_model_parameter()
                })
            self.listen()
            self.aggregate()
            self.test(self.model, self.test_loader)
            self.average_client_info(self.selected_clients_idxes)
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.args.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        # out of loop
        self.log("Server stopped")
        self.stop_all()


if __name__ == "__main__": 
    server = MetaServer()
    server.run()