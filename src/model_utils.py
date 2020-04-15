import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

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

def inference_personal(model, model_personal, alpha, dataloader, criterion, device):
    """ Returns the inference accuracy and loss.
    """

    model.eval()
    model_personal.eval()
    
    total, correct = 0.0, 0.0
    loss = list()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs_1 = model_personal(images)
        outputs_2 = model(images)
        outputs =  alpha * outputs_1 + (1-alpha) * outputs_2
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
