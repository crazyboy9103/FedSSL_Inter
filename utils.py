#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import os
import time
import GPUtil
from threading import Thread
from sklearn.neighbors import KDTree

def gradient_diversity(local_weights, global_weight):
    nabla = {}
    l2_norm = torch.tensor(0., device="cuda:0")
    for client_id, local_weight in local_weights.items():
        for k in local_weight:
            if "weight" in k or "bias" in k:
                nabla_k = local_weight[k].detach() - global_weight[k].detach()
                l2_norm += torch.pow(torch.norm(nabla_k), 2)
                if k in nabla:
                    nabla[k] += nabla_k
                
                else:
                    nabla[k] = nabla_k
        
    denom = torch.tensor(0., device="cuda:0")
    for k, nabla_k in nabla.items():
        denom += torch.pow(torch.norm(nabla_k), 2)

    grad_div = l2_norm / denom
    return grad_div.item()
    
def bn_divergence(local_weights, global_weight):
    div = {}
    l2_norms = {}
    for client_id, local_weight in local_weights.items():
        for k in local_weight:
            if "bn" in k and ("weight" in k or "bias" in k or "running_mean" in k or "running_var" in k):
                div_k = local_weight[k].detach() - global_weight[k].detach()
                if k in l2_norms:
                    l2_norms[k] += torch.pow(torch.norm(div_k), 2)
                else:
                    l2_norms[k] = torch.pow(torch.norm(div_k), 2)

                if k in div:
                    div[k] += div_k
                else:
                    div[k] = div_k
    
    denom = torch.tensor(0., device="cuda:0")
    divs = {}
    for k in div:
        div_k = div[k]
        l2_norm = l2_norms[k]
        
        result = l2_norm / torch.pow(torch.norm(div_k), 2)
        divs[k] = result
    return divs



def kNN_helpers(args, model, local_weights):
    random_input = torch.rand((1, 3, 32, 32), device="cuda:0" if torch.cuda.is_available() else "cpu", requires_grad=False)
    reps = None
    with torch.no_grad():
        model = model.eval()
        for client_id, local_weight in local_weights.items():
            model.load_state_dict(local_weight)
            out = model(random_input, return_embedding=True)
            if reps == None:
                reps = out
            
            else:
                reps = torch.cat((reps, out))
    
    helpers = {}

    reps = reps.detach().cpu().numpy()
    tree = KDTree(reps, leaf_size=6)
    for i in range(len(reps)):
        rep = reps[i]
        _, indices = tree.query(np.expand_dims(rep, 0), k=args.topH)
        indices = np.squeeze(indices, 0)

        if i not in helpers:
            helpers[i] = {}
        
        for index in indices:
            helpers[i][index] = copy.deepcopy(local_weights[index])
    
    # helpers = {client_id: {neighbor_id: state_dict}}
    return helpers






class EMAHelper(object):
    # Usage: 
    #   network
    #   ema_helper = EMAHelper()
    #   ema_helper.register(network)
    # For update:
    #   ema_helper.update(newly_trained_network)
    #   ema_helper.ema(network) # updates network in-place
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

class Timer():
    def __init__(self):
        self.last = None
    
    def set_timer(self):
        self.last = time.time()

    def see_timer(self):
        last = self.last
        self.last = None
        return time.time()-last

# https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py 
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class AverageMeter():
    def __init__(self, name):
        self.name = name
        self.values = []
    
    def update(self, value):
        self.values.append(value)
        
    def get_result(self):
        return sum(self.values)/len(self.values)
    
    def reset(self):
        self.values = []

def collect_weights(w):
    with torch.no_grad():
        w_avg = w[0]  
        for key in w_avg.keys():
            for i in range(1, len(w)): # 0th gpu resides at 0th index
                if 0 != w[i][key].get_device():
                    w[i][key] = w[i][key].to(w_avg[key].get_device())

def average_weights(w):
    """
    Returns the average of the weights.
    """
    with torch.no_grad():
        w_avg = copy.deepcopy(w[0]) 
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
                
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class CheckpointManager():
    def __init__(self, type):
        self.type = type
        if type == "loss":
            self.best_loss = 1E27 
        
        elif type == "top1":
            self.best_top1 = -1E27 

    def _check_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        
        return False
    
    def _check_top1(self, top1):
        if top1 > self.best_top1:
            self.best_top1 = top1
            return True
        
        return False


    def save(self, loss, top1, model_state_dict, checkpoint_path):
        save_dict = {
            "model_state_dict": model_state_dict, 
            # "optim_state_dict": optim_state_dict, 
            "loss": loss, 
            "top1": top1
        }
        if self.type == "loss" and self._check_loss(loss):
            torch.save(save_dict, checkpoint_path)

        elif self.type == "top1" and self._check_top1(top1):
            torch.save(save_dict, checkpoint_path)

        print(f"model saved at {checkpoint_path}")

class MonitorGPU(Thread):
    def __init__(self, delay):
        super(MonitorGPU, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()
    
    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)
    
    def stop(self):
        self.stopped = True
