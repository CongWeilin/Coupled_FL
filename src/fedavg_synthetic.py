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

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

##########################################################################
##########################################################################
##########################################################################

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_hidden = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)
    
def inference(model, dataloader, criterion, device):
    """ Returns the inference accuracy and loss.
    """

    model.eval()
    total, correct = 0.0, 0.0
    loss = list()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += [batch_loss.item()]

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = sum(loss)/len(loss)
    return accuracy, loss

def average_state_dicts(w, weight):
    """
    Returns the average of the weights or gradients.
    """
    weight = weight/sum(weight)
    
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] = w_avg[key] + w[i][key]*weight[i]
    return w_avg

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

    global_init_weight = copy.deepcopy(global_model.state_dict())
    local_model_init_weight_list = []
    for local_id in range(config['num_devices']):
        local_weight = copy.deepcopy(local_model_list[local_id].state_dict())
        local_model_init_weight_list.append(local_weight)

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
    global_model.load_state_dict(global_init_weight)
    for local_id in range(config['num_devices']):
        local_model_list[local_id].load_state_dict(local_model_init_weight_list[local_id])

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