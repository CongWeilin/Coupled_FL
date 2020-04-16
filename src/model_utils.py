import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

##########################################################################
##########################################################################
##########################################################################

class Perceptron(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Perceptron, self).__init__()
        self.layer_hidden = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, dataset, num_layers, hidden_size, drop_rate):
        super(MLP, self).__init__()
        self.dataset = dataset

        # init
        self.num_layers = num_layers
        self.num_classes = self._decide_num_classes()
        input_size = self._decide_input_feature_size()

        # define layers.
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size

            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=drop_rate))
            setattr(self, 'layer{}'.format(i), layer)

        self.fc = nn.Linear(hidden_size, self.num_classes, bias=False)

    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100

    def _decide_input_feature_size(self):
        if 'cifar' in self.dataset:
            return 32 * 32 * 3
        elif 'mnist' in self.dataset:
            return 28 * 28
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for i in range(1, self.num_layers + 1):
            x = getattr(self, 'layer{}'.format(i))(x)
        x = self.fc(x)
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
        for i in range(len(weight)):
            w_ = w[i][key]*weight[i]
            w_avg[key] = w_avg[key] + w_
    return w_avg
