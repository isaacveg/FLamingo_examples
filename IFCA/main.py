import sys

# from FLamingo.datasets.leaf_data.dataset import femnist_dataset
sys.path.append('../FLamingo/')

from FLamingo.core.runner import Runner

def quick_run(ds_type, niid=None, lr=0.05, alpha=0.1, beta=0.1, bs=32, local_iters=30, optim_step=10, max_e=500, abl_v=False, nc=10):
    if ds_type=='c':
        ds_type = 'cifar10'
        if niid=='iid':
            ds_path = '../datasets/cifar10_nc50_distiid_blc1'
        else:
            ds_path = f'../datasets/cifar10_nc50_distdir{niid}_blc0'
        md_t = 'alexnet'
    elif ds_type=='f':
        ds_type = 'femnist'
        ds_path = '../datasets/femnist'
        md_t = 'fedavgcnn'
    elif ds_type=='s':
        ds_type = 'shakespeare'
        ds_path = '../datasets/shakespeare'
        md_t = 'lstm'
    runner = Runner(cfg_file='./config.yaml')
    runner.update_cfg({
        'ABLATION_V': abl_v,
        'dataset_type': ds_type,
        'data_dir': ds_path,
        'model_type': md_t,
        'lr': lr,
        'alpha': alpha,
        'beta': beta,
        'batch_size': bs,
        'local_iters': local_iters,
        'optim_steps': optim_step,
        'max_epochs': max_e,
        'num_training_clients': nc
    })
    runner.run()
    runner.rename_last_run_dir(f'{ds_type}_{niid}_bs{bs}_{lr},{alpha},{beta},{nc},li{local_iters}')
    

if __name__ == "__main__":
    # init Runner
    runner = Runner(cfg_file='./config.yaml')
    runner.update_cfg('USE_SIM_SYSHET', True)
    # CIFAR-10
    # quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,10)
    # quick_run('c', '0.5', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,10)
    # quick_run('c', '1.0', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,10)
    # quick_run('c', 'iid', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,10)
    # # FEMNIST
    # quick_run('f', None, 0.05, 0.1, 0.1, 32, 30, 1, 500, False,10)
    # # SHAKESPEARE
    # quick_run('s', None, 1.4, 1.4, 0.5, 16, 30, 1, 300, False,10)
    # num_training
    # quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,20)
    # quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,30)
    # quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 30, 1, 500, False,40)
    # num_local_iters
    quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 20, 1, 500, False,10)
    quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 40, 1, 500, False,10)
    quick_run('c', '0.1', 0.05, 0.1, 0.1, 32, 50, 1, 500, False,10)