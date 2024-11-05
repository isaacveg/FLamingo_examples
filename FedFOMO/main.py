import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
from FLamingo.core.runner import Runner
# from FLamingo.datasets import generate_cifar10


if __name__ == "__main__":
    runner = Runner(cfg_file='./config.yaml')
    # ================ Shakespeare
    # lr 0.1, bs 10
    # runner.update_cfg({
    #     'dataset_type': 'shakespeare',
    #     'data_dir': '../datasets/shakespeare/',
    #     'model_type': 'lstm',
    #     'lr': 0.5,
    #     # 'alpha': 1.0,
    #     # 'beta': 0.5,
    #     'batch_size': 16,
    #     'local_iters': 30
    #     # 'optim_steps': 10
    # })
    # runner.run()
    # runner.rename_last_run_dir('s_lr0.5_bs16')

    # ================ FEMNIST
    runner.update_cfg('dataset_type', 'femnist')
    runner.update_cfg('data_dir', '../datasets/femnist/')
    runner.update_cfg('model_type', 'fedavgcnn')
    # lr 0.01, bs 32 
    runner.update_cfg({
        'dataset_type': 'femnist',
        'data_dir': '../datasets/femnist/',
        'model_type': 'fedavgcnn',
        'lr': 0.03,
        # 'alpha': 0.1,
        # 'beta': 0.1,
        'batch_size': 32,
        'local_iters': 30
        # 'optim_steps': 10
    })
    runner.run()
    runner.rename_last_run_dir('f_lr0.03_bs32')

    # ================ Cifar10_nc30
    # iid, lr 0.05, bs 32 ,0.1, 0.1
    # runner.update_cfg({
    #     'dataset_type': 'cifar10',
    #     'data_dir': '../datasets/cifar10_nc50_distiid_blc1/',
    #     'model_type': 'alexnet',
    #     'lr': 0.05,
    #     # 'alpha': 0.5,
    #     # 'beta': 0.1,
    #     'batch_size': 32,
    #     'local_iters': 30,
    #     # 'optim_steps': 10
    # })
    # runner.run()
    # runner.rename_last_run_dir('c_i_lr0.05,iid_bs32')
    # # dir0.1, lr 0.05, bs 32
    # runner.update_cfg({
    #     'dataset_type': 'cifar10',
    #     'data_dir': '../datasets/cifar10_nc50_distdir0.1_blc0/',
    #     'model_type': 'alexnet',
    #     'lr': 0.05,
    #     # 'alpha': 0.5,
    #     # 'beta': 0.1,
    #     'batch_size': 32,
    #     'local_iters': 30,
    #     # 'optim_steps': 10
    # })
    # runner.run()
    # runner.rename_last_run_dir('c_n_lr0.05,dir0.1_bs32')
    # runner.update_cfg({
    #     'dataset_type': 'cifar10',
    #     'data_dir': '../datasets/cifar10_nc50_distdir0.5_blc0/',
    #     'model_type': 'alexnet',
    #     'lr': 0.05,
    #     # 'alpha': 0.5,
    #     # 'beta': 0.1,
    #     'batch_size': 32,
    #     'local_iters': 30,
    #     # 'optim_steps': 10
    # })
    # runner.run()
    # runner.rename_last_run_dir('c_n_lr0.05,dir0.5_bs32')
    # runner.update_cfg({
    #     'dataset_type': 'cifar10',
    #     'data_dir': '../datasets/cifar10_nc50_distdir1.0_blc0/',
    #     'model_type': 'alexnet',
    #     'lr': 0.05,
    #     # 'alpha': 0.5,
    #     # 'beta': 0.1,
    #     'batch_size': 32,
    #     'local_iters': 30,
    #     # 'optim_steps': 10
    # })
    # runner.run()
    # runner.rename_last_run_dir('c_n_lr0.05,dir1.0_bs32')
