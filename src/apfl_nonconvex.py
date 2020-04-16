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

from image_dataloader import get_dataset, train_val_dataloader
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
    global_model = MLP(dataset=config['dataset'], 
                       num_layers=config['num_layers'],
                       hidden_size=config['hidden_size'],
                       drop_rate=config['drop_rate']).to(device)
    print(global_model)

    local_model_list = []
    local_optim_list = []
    for local_id in range(config['num_devices']):
        local_model = MLP(dataset=config['dataset'], 
                       num_layers=config['num_layers'],
                       hidden_size=config['hidden_size'],
                       drop_rate=config['drop_rate']).to(device)
        local_optim = GD(local_model.parameters(), lr=config['lr'], weight_decay=1e-4)
        local_model_list.append(local_model)
        local_optim_list.append(local_optim)

    personal_model_list = []
    personal_optim_list = []
    for local_id in range(config['num_devices']):
        personal_model = MLP(dim_in=config['dimension'], dim_out=config['num_classes']).to(device)
        personal_optim = GD(personal_model.parameters(), lr=config['lr'], weight_decay=1e-4)
        personal_model_list.append(personal_model)
        personal_optim_list.append(personal_optim)

    init_weight = copy.deepcopy(global_model.state_dict())

    criterion = nn.NLLLoss().to(device)

    """
    load data and user group
    """
    train_dataset, test_dataset, user_groups = get_dataset(config)
    weight_per_user = [len(user_groups[local_id]) for local_id in range(config['num_devices'])]
    weight_per_user = np.array(weight_per_user)/sum(weight_per_user)

    trainloader_list, validloader_list = [], []
    trainloader_iterator_list, validloader_iterator_list = [], []
    for local_id in range(config['num_devices']):
        trainloader, validloader = train_val_dataloader(train_dataset, user_groups[local_id], 
                                                        batch_size=config['local_batch_size'])
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
                # cur_train_loss.append(loss.item())

                local_optim_list[local_id].step()
                
                # train personalize model
                personal_model_list[local_id].train()
                personal_model_list[local_id].zero_grad()
                log_probs_1 = local_model_list[local_id](inputs)
                log_probs_2 = personal_model_list[local_id](inputs)
                log_probs = (1-config['p_alpha']) * log_probs_1 + config['p_alpha'] * log_probs_2
                loss = criterion(log_probs, labels)
                loss.backward()
                cur_train_loss.append(loss.item())
                
                personal_optim_list[local_id].step()

                # test local model
                acc, loss = inference_personal(local_model_list[local_id], personal_model_list[local_id], 
                                               config['p_alpha'], validloader_list[local_id], criterion, device)
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
    with open('apfl_%s_noniid_%d.pkl'%(config['dataset'], config['noniid_level']), 'wb') as f:
        pickle.dump([global_acc, global_loss, local_acc, local_loss, train_loss], f)
                

if __name__ == '__main__':
    main(config_path='../config/mnist_iid.yaml')
    main(config_path='../config/mnist_noniid_2.yaml')
    main(config_path='../config/mnist_noniid_4.yaml')