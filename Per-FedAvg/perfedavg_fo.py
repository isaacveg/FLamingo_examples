import sys
import time
import math
import re
import gc
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from FLamingo.core.utils.train_test_utils import infinite_dataloader


def train(model, train_loader, alpha, beta, local_iters=None, local_epochs=1, device=torch.device("cpu"), model_type=None):
    """Using first-order MAML, train the model on the local dataset.
    Returns the updated model and the train_loss(batch1), grad_loss(batch2) 
    """
    t_start = time.time()
    model.train()
    
    # First Order MAML train for local_ites
    if local_iters is None:
        # 2 batches 1 update
        local_iters = math.ceil(len(train_loader.dataset) / train_loader.batch_size / 2)
    else:
        local_iters = math.ceil(local_iters / 2)
    losses = [0.0, 0.0]
    sample_num = [0, 0]
    batches_num = 0
    
    train_loader = infinite_dataloader(train_loader)

    for epoch_idx in range(local_epochs):
        for epoch in range(local_iters):
            # ========================== FedAvg ==========================
            # NOTE: You can uncomment those codes for running FedAvg.
            #       When you're trying to run FedAvg, comment other codes in this branch.

            # data_batch = self.get_data_batch()
            # grads = self.compute_grad(self.model, data_batch)
            # for param, grad in zip(self.model.parameters(), grads):
            #     param.data.sub_(self.beta * grad)

            # ============================================================

            temp_model = copy.deepcopy(model)
            data_batch_1 = next(train_loader)
            grads, grad_loss1 = compute_grad(temp_model, data_batch_1, device)

            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(alpha * grad)

            data_batch_2 = next(train_loader)
            grads, grad_loss2 = compute_grad(temp_model, data_batch_2, device)

            for param, grad in zip(model.parameters(), grads):
                param.data.sub_(beta * grad)
        
            losses = [losses[0]+grad_loss1, losses[1]+grad_loss2]
            sample_num = [sample_num[0]+len(data_batch_1[0]), sample_num[1]+len(data_batch_2[0])]
            batches_num += 2

    return {'train_loss': losses[0]/sample_num[0] if sample_num[0] != 0 else losses[0], 
            'grad_loss': losses[1]/sample_num[1] if sample_num[1] != 0 else losses[1], 
            'train_time': time.time()-t_start,
            'samples_num': sum(sample_num),
            'batches_num': batches_num,
            'params': torch.nn.utils.parameters_to_vector(model.parameters()).detach()}

        # else:  # Per-FedAvg(FO)
        #     for _ in range(self.local_epochs):




def one_step(device, data, model, model_type, lr):
    """
    Performs one step of training for a given device, data, model, and learning rate.

    Args:
        device (torch.device): The device (CPU or GPU) on which to perform the calculations.
        data (tuple): A tuple containing the input sequence (seq) and corresponding label (label).
        model (torch.nn.Module): The model to train.
        lr (float): The learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the updated model and the loss value as a float.
    """
    seq, label = data

    if model_type == 'LR':
        seq = data.squeeze(1).view(-1, 28 * 28)

    seq = seq.to(device)
    label = label.to(device)
    y_pred = model(seq)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(device)
    loss = loss_function(y_pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, loss.item()


def compute_grad(model, data_batch, device, v=None, second_order_grads=False):
    """
    Compute the gradients of the model parameters with respect to the loss function.

    Parameters:
    - model: The model for which to compute the gradients.
    - data_batch: The input data batch consisting of features and labels.
    - device: The device on which to perform the computations.
    - v: Optional. The second-order gradients used for computation.
    - second_order_grads: Optional. Whether to compute second-order gradients.

    Returns:
    - grads: The gradients of the model parameters.
    - loss: The computed loss value.
    """
    x, y = data_batch
    x, y = x.to(device), y.to(device)
    loss_func = nn.CrossEntropyLoss().to(device)

    if second_order_grads:
        frz_model_params = copy.deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        logit_1 = model(x)
        # loss_func = nn.CrossEntropyLoss().to(device)
        loss_1 = loss_func(logit_1, y)

        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        logit_2 = model(x)
        # loss_func_2 = nn.CrossEntropyLoss()
        loss_2 = loss_func(logit_2, y)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
        return grads, (loss_1.item()+loss_2.item())/2

    else:
        logit = model(x)
        loss = loss_func(logit, y)
        grads = torch.autograd.grad(loss, model.parameters())
        return grads, loss.item()