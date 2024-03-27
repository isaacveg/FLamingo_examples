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


class FedAvgServer(Server):
    """
    FedAvg Server, the original FLamingo Server
    """
    def generate_global_test_set(self):
        """
        Generate a global test set.
        """
        if self.dataset_type == 'mnist':
            self.test_set = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif self.dataset_type == 'cifar10':
            self.test_set = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
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
    
    def run(self):
        """
        FedAvg procedure:
        0. Initialize
        1. Select clients
        2. Send requests to clients
        3. Waiting for training results
        4. Aggregating results
        5. Evaluating and record
        """
        # self.init()
        self.print_model_info()
        self.init_clients(clientObj=ClientInfo)
        self.generate_global_test_set()
        while True:
            # Selecting and set params
            self.select_clients()
            # Sending models and parameters.
            self.broadcast(data={'status':'TRAINING',
                                 'params':self.export_model_parameter()}
                           )
            # Waiting for responses
            self.listen()
            # self.log(f'server param: {self.export_model_parameter()}')
            # stop_str = ''
            # for client in self.all_clients:
            #     if 'STOP' in client.status:
            #         stop_str += client.status.split('_')[1] + ','
            # self.log(stop_str)
            # # Aggregating model parameters
            self.aggregate()
            # self.log(f'server param after aggregation: {self.export_model_parameter()}')
            # Evaluate
            self.test(self.model, self.test_loader)
            # Average
            self.average_client_info(client_list=self.selected_clients_idxes)
            # break
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.args.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        self.stop_all()


if __name__ == "__main__":
    server = FedAvgServer()
    server.run()