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


class FedAvgClient(Client):
    """
    FedAvg Client, the original FLamingo Client
    """
    def run(self):
        """
        FedAvg procedure:
        1. Get request from server
        2. Training
        3. Send information back
        """
        # self.model = create_model_instance(self.model)
        # self.init()
        while True:
            # data = self.receive_data(self.MASTER_RANK)
            # data = self.network.get(self.MASTER_RANK)
            data = self.listen()
            # print([k for k,v in data.items()])
            if data['status'] == 'STOP':
                if self.verb: self.log('Stopped by server')
                break
            elif data['status'] == 'DEBUG':
                # debug mode, print everything client recieved.
                self.log(f"{data}")
                self.send(data={'status':f'STOP_{self.rank}'})
                break
            elif data['status'] == 'TRAINING':
                # print(f'{self.rank}, {self.verb}')
                # if self.verb: self.log('training')
                # self.log(f'client param after training: {self.export_model_parameter()}')
                self.set_model_parameter(data['params'])
                # self.log(f'client param after setting: {self.export_model_parameter()}')
                trained_info = self.train(
                    self.model, self.train_loader, self.args.local_epochs, self.loss_func, self.optimizer)
                tested_info = self.test(
                    self.model, self.test_loader, self.loss_func, self.device)
                # Construct data to send
                data_to_send = merge_several_dicts([trained_info, tested_info])
                data_to_send['params'] = self.export_model_parameter()
                # print(data_to_send)
                # self.network.send(data_to_send, self.MASTER_RANK)
                self.send(data_to_send)
                # if self.verb: self.log('training finished')

            elif data['status'] == 'TEST':
                if self.verb: self.log('testing')
                self.model = torch.load(data['model'])
                test_info = self.test()
                if self.verb: 
                    self.log(f'test info: {test_info.items()}')
                    # for k, v in test_info.items():
                    #     self.log(f'{k}: {v}')
                self.send(self.MASTER_RANK, test_info)
        
            else:
                raise Exception('Unknown status')
            
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        
        if self.verb: self.log(f'finished at round {self.global_round}')
       

if __name__ == '__main__':
    client = FedAvgClient()
    client.run()