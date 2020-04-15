import os
import copy
import time
import pickle
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from synthetic_dataloader import train_val_dataloader, SyntheticDataset
from gd import GD
from model_utils import MLP
from model_utils import inference, inference_personal, average_state_dicts

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

##########################################################################
##########################################################################
##########################################################################
def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
        
    device = torch.device(config['device'])
    
    """
    Setup models
    """
    global_model = MLP(dim_in=config['dimension'], dim_out=config['num_classes']).to(device)
    
    local_model_list = []
    local_optim_list = []
    for local_id in range(config['num_devices']):
        local_model = MLP(dim_in=config['dimension'], dim_out=config['num_classes']).to(device)
        local_optim = GD(local_model.parameters(), lr=config['lr'], weight_decay=1e-4)
        local_model_list.append(local_model)
        local_optim_list.append(local_optim)

    init_weight = copy.deepcopy(global_model.state_dict())

    criterion = nn.NLLLoss().to(device)
    
    """
    generate training data
    """
    synthetic_dataset = SyntheticDataset(num_classes=config['num_classes'], 
                                     num_tasks=config['num_devices'], 
                                     num_dim=config['dimension'],
                                     alpha=config['alpha'], beta=config['beta'])
    data = synthetic_dataset.get_all_tasks()
    num_samples = synthetic_dataset.get_num_samples()
    weight_per_user = num_samples/sum(num_samples)

    trainloader_list, validloader_list = [], []
    trainloader_iterator_list, validloader_iterator_list = [], []
    for local_id in range(config['num_devices']):
        trainloader, validloader = train_val_dataloader(data[local_id]['x'], data[local_id]['y'], batch_size=config['local_batch_size'])
        trainloader_list.append(trainloader)
        validloader_list.append(validloader)

        trainloader_iterator_list.append(iter(trainloader_list[local_id]))
        validloader_iterator_list.append(iter(validloader_list[local_id]))
        
    """
    load initial value
    """
    global_model.load_state_dict(init_weight)
    for local_id in range(config['num_devices']):
        local_model_list[local_id].load_state_dict(init_weight)

    """
    start training
    """
    global_acc = []
    global_loss = []

    local_loss = []
    local_acc = []
    train_loss = []

    # test global model
    list_acc, list_loss = [], [] 
    for local_id in range(config['num_devices']):
        acc, loss = inference(global_model, validloader_list[local_id], criterion, device)
        list_acc.append(acc)
        list_loss.append(loss)
    global_acc +=  [sum(list_acc)/len(list_acc)]
    global_loss += [sum(list_loss)/len(list_loss)]
    print('global %d, acc %f, loss %f'%(len(global_acc), global_acc[-1], global_loss[-1]))

    for global_iter in range(config['global_iters']): # T
        activate_devices = np.random.permutation(config['num_devices'])[:config['num_active_devices']] # np.arange(config['num_devices'])
        # get the local grad for each device
        
        for local_iter in range(config['local_iters']): # E
            cur_local_loss = []
            cur_local_acc = []
            cur_train_loss = []
            for local_id in activate_devices: # K
                # load single mini-batch
                try:
                    inputs, labels = next(trainloader_iterator_list[local_id])
                except StopIteration:
                    trainloader_iterator_list[local_id] = iter(trainloader_list[local_id])
                    inputs, labels = next(trainloader_iterator_list[local_id])

                # train local model
                inputs, labels = inputs.to(device), labels.to(device)
                local_model_list[local_id].train()
                local_model_list[local_id].zero_grad()
                log_probs = local_model_list[local_id](inputs)
                loss = criterion(log_probs, labels)
                loss.backward()
                cur_train_loss.append(loss.item())

                local_optim_list[local_id].step() # lr=config['lr']/(1+global_iter)

                # test local model
                acc, loss = inference(local_model_list[local_id], validloader_list[local_id], criterion, device)
                cur_local_loss.append(loss)
                cur_local_acc.append(acc)

            cur_local_loss = sum(cur_local_loss)/len(cur_local_loss)
            cur_local_acc = sum(cur_local_acc)/len(cur_local_acc)
            cur_train_loss = sum(cur_train_loss)/len(cur_train_loss)
            local_loss.append(cur_local_loss)
            local_acc.append(cur_local_acc)
            train_loss.append(cur_train_loss)

        # update learning rate
        for local_id in activate_devices:
            local_optim_list[local_id].inverse_prop_decay_learning_rate(global_iter) 

        # average local models 
        local_weight_list = [local_model.state_dict() for local_model in local_model_list]
        avg_local_weight = average_state_dicts(local_weight_list, weight_per_user)
        global_model.load_state_dict(avg_local_weight)
        for local_id in range(config['num_devices']):
            local_model_list[local_id].load_state_dict(avg_local_weight)

        # test global model
        list_acc, list_loss = [], [] 
        for local_id in range(config['num_devices']):
            acc, loss = inference(global_model, validloader_list[local_id], criterion, device)
            list_acc.append(acc)
            list_loss.append(loss)
        global_acc +=  [sum(list_acc)/len(list_acc)]
        global_loss += [sum(list_loss)/len(list_loss)]
        print('global %d, acc %f, loss %f'%(len(global_acc), global_acc[-1], global_loss[-1]))
        
    """
    save results
    """
    with open('fedavg_%.2f.pkl'%config['beta'], 'wb') as f:
        pickle.dump([global_acc, global_loss, local_acc, local_loss, train_loss], f)
        

if __name__ == '__main__':
    main(config_path='../config/synthetic_noniid_0_0.yaml')
    main(config_path='../config/synthetic_noniid_0.5_0.5.yaml')
    main(config_path='../config/synthetic_noniid_1_1.yaml')