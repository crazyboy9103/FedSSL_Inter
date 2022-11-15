import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy 
#import numpy as np
import os
from utils import EMAHelper

from torch.utils.data import DataLoader
#from fast_pytorch_kmeans import KMeans

def finetune_client_model(args, dataset, device, unsup_model, epochs=1, freeze=False):
    model = copy.deepcopy(unsup_model).to(device)
    
    model = model.train()
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model.classifier.reset_parameters()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr
    )

    loader = DataLoader(
        dataset,
        batch_size = args.finetune_bs, 
        shuffle = True,
        num_workers = args.num_workers,
        drop_last = True,
    )
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader): 
            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images, return_logits=True)
            loss = FL_criterion(device)(preds, labels)

            loss.backward()
            optimizer.step()
            
    return copy.deepcopy(model.state_dict())

def train_server_model(args, dataset, device, sup_model, epochs=1):
    model = copy.deepcopy(sup_model).to(device)
    
    model = model.train()
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr
    )

    loader = DataLoader(
        dataset,
        batch_size = args.server_bs, 
        shuffle = True,
        num_workers = args.num_workers,
        drop_last = True,
    )

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader): 
            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images, return_logits=True)
            loss = FL_criterion(device)(preds, labels)

            loss.backward()
            optimizer.step()
            
    return copy.deepcopy(model.state_dict())

    

def train_client_model(args, dataset, device, sup_model = None, unsup_model = None, epochs=1, q=None, done=None, helpers=None):
    client_model = copy.deepcopy(unsup_model).to(device)
    
    client_model = client_model.train()
    for param in client_model.parameters():
        param.requires_grad = True
    
    # this condition is same as main.py => should_send_server_model
    if args.agg == "FedSSL" or args.exp in ["FedBYOL", "FedMatch"]:
        assert sup_model != None, "sup model must be non null"
        ref_model = copy.deepcopy(sup_model).to(device).eval()

    if args.agg == "FedProx":
        glob_model = copy.deepcopy(unsup_model).to(device)

    if args.exp == "BYOL":
        ema_helper = EMAHelper()
        ema_helper.register(client_model)
        target_net = copy.deepcopy(client_model).to(device).eval() 

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, client_model.parameters()),
        args.lr
    )

    loader = DataLoader(
        dataset,
        batch_size = args.local_bs, 
        shuffle = True,
        num_workers = args.num_workers,
        drop_last = True,
    )

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            optimizer.zero_grad()
            if args.exp == "FLSL" or args.exp == "centralized":
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                preds = client_model(images, return_logits=True)
                loss = FL_criterion(device)(preds, labels)

            elif args.exp == "PseudoLabel":
                images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    pseudo_preds = client_model(images, return_logits=True)
                    pseudo_labels = pseudo_preds.argmax(dim=1)
                
                preds = client_model(images, return_logits=True)
                loss = FL_criterion(device)(preds, pseudo_labels)

            elif args.exp == "FedMatch":
                orig_images, views = images
                orig_images, views = orig_images.to(device, non_blocking=True), views.to(device, non_blocking=True)

                # Consistency regularization
                orig_state_dict = client_model.state_dict()
                with torch.no_grad():
                    z = client_model(orig_images, return_embedding=True)
                
                p =  client_model.projector(z)
                orig_logits = client_model.classifier(p)

                _, pseudo_labels = torch.max(orig_logits, 1) # indices are pseudo-labels from local model
                loss = torch.tensor(0., device=device)

                if helpers != None:
                    helper_labels = {}
                    for client_id, helper_state_dict in helpers.items():
                        with torch.no_grad():
                            client_model.load_state_dict(helper_state_dict)
                            _, p = client_model(orig_images)
                            helper_logits = client_model.classifier(p).detach()
                                            
                        loss += (nn.KLDivLoss(size_average="batchmean")(orig_logits, helper_logits)) / len(helpers)

                        values, indices = torch.max(nn.Softmax()(helper_logits), 1)
                        helper_label = []
                        for value, index in zip(values, indices):
                            if value > args.tau:
                                helper_label.append(index)
                            else:
                                helper_label.append(-1)

                        helper_labels[client_id] = helper_label
                    
                    

                for i, pseudo_label in enumerate(pseudo_labels):
                    counts = {pseudo_label: 1}
                    if helpers != None:
                        for client_id, helper_label in helper_labels.items():
                            if helper_label[i] != -1:
                                if helper_label[i] not in counts:
                                    counts[helper_label[i]] = 1
                                else:
                                    counts[helper_label[i]] += 1
                    
                    maj_vote_label = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                    pseudo_labels[i] = maj_vote_label
                
                client_model.load_state_dict(orig_state_dict)

                with torch.no_grad():
                    z = client_model(views, return_embedding=True)
                
                p =  client_model.projector(z)
                local_logits = client_model.classifier(p)

                loss += FL_criterion(device)(local_logits, pseudo_labels)
                loss *= args.iccs_lambda

            elif args.exp in ["SimCLR", "SimSiam", "FixMatch", "BYOL", "FedBYOL"]:
                orig_images, views1, views2 = images
                orig_images = orig_images.to(device, non_blocking=True)
                views1 = views1.to(device, non_blocking=True)
                views2 = views2.to(device, non_blocking=True)
                
                if args.exp == "SimCLR":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)
                    loss1 = NCE_loss(device, args.temperature, p1, p2) 
                    loss2 = NCE_loss(device, args.temperature, p2, p1) 
                    loss = (loss1 + loss2).mean()
                    
                elif args.exp == "SimSiam":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)
                    loss = SimSiam_loss(device, p1, p2, z1.detach(), z2.detach())

                elif args.exp == "FixMatch":
                    # view1 = weak aug
                    # view2 = strong aug
                    with torch.no_grad():
                        pseudo_preds = client_model(views1, return_logits=True).detach()
                        pseudo_probs, pseudo_labels = torch.max(nn.Softmax(dim=1)(pseudo_preds), 1)
                        above_thres = pseudo_probs > args.threshold
                    
                    if len(above_thres) > 0:
                        pseudo_labels = pseudo_labels[above_thres]
                        views2 = views2[above_thres]
                        preds = client_model(views2, return_logits=True)
                        loss = FL_criterion(device)(preds, pseudo_labels)
                        # pseudo_labels = pseudo_preds.argmax(dim=1)
                    else:
                        loss = torch.tensor(0., device=device)

                
                elif args.exp == "BYOL":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)
                    ema_helper.update(client_model)
                    ema_helper.ema(target_net)
                    with torch.no_grad():
                        # Use EMA target net
                        ref_z1 = target_net(views1, return_embedding=True)
                        ref_z2 = target_net(views2, return_embedding=True)
                    loss1 = BYOL_loss(device, p1, ref_z2.detach())
                    loss2 = BYOL_loss(device, p2, ref_z1.detach())
                    loss = (loss1 + loss2).mean()

                elif args.exp == "FedBYOL":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)   
                    with torch.no_grad():
                        # Use server model as target net
                        ref_z1 = ref_model(views1, return_embedding=True)
                        ref_z2 = ref_model(views2, return_embedding=True)
                    loss1 = BYOL_loss(device, p1, ref_z2.detach())
                    loss2 = BYOL_loss(device, p2, ref_z1.detach())
                    loss = (loss1 + loss2).mean()
            
            

            if epoch > 0:
                if args.agg == "FedProx":
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(client_model.backbone.parameters(), glob_model.backbone.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    
                    if client_model.projector != None: 
                        for w, w_t in zip(client_model.projector.parameters(), glob_model.projector.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += args.mu / 2. * w_diff

            
                elif args.agg == "FedSSL":
                    # MSE Loss
                    # with torch.no_grad():
                    #     ref_z = ref_model(orig_images, return_embedding=True)
                    
                    # orig_z, orig_p = client_model(orig_images)
                    # mse_loss = nn.MSELoss(reduction="mean").to(device)
                    # dis_loss = args.mse_ratio * mse_loss(ref_z.detach(), orig_p).mean()
                    # loss += dis_loss
                    # ===
                    #
                    
                    activation = {}
                    def get_activation(name):
                        def hook(model, input, output):
                            activation[name] = output # no detach here as we want to train client model
                        return hook
                    
                    teacher_activation = {}
                    def get_teacher_activation(name):
                        def hook(model, input, output):
                            teacher_activation[name] = output.detach()
                        return hook

                    teacher_handles = [
                        ref_model.backbone.layer1.register_forward_hook(get_teacher_activation('layer1')),
                        ref_model.backbone.layer2.register_forward_hook(get_teacher_activation('layer2')),
                        ref_model.backbone.layer3.register_forward_hook(get_teacher_activation('layer3')),
                        ref_model.backbone.layer4.register_forward_hook(get_teacher_activation('layer4'))
                    ]

                    client_handles = [
                        client_model.backbone.layer1.register_forward_hook(get_activation('layer1')),
                        client_model.backbone.layer2.register_forward_hook(get_activation('layer2')),
                        client_model.backbone.layer3.register_forward_hook(get_activation('layer3')),
                        client_model.backbone.layer4.register_forward_hook(get_activation('layer4'))
                    ]
                    
                    with torch.no_grad():
                        ref_z = ref_model(orig_images, return_embedding=True)
                    
                    orig_z = client_model(orig_images, return_embedding=True)

                    mse_fn = nn.MSELoss(reduction="mean").to(device)
                    dis_loss = mse_fn(ref_z.detach(), orig_z).mean()

                    for name in teacher_activation:
                        dis_loss +=  args.mse_ratio * mse_fn(teacher_activation[name], activation[name])
                    #KDLoss = FedSSL_loss(device, outputs = orig_z, teacher_outputs = ref_z.detach(), args = args)
                    loss += dis_loss
                    for th, ch in zip(teacher_handles, client_handles):
                        th.remove()
                        ch.remove()
                
                if args.exp == "FedMatch":
                    assert args.agg != 'FedProx', 'agg must be either FedAvg or FedSSL'
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(client_model.backbone.parameters(), ref_model.backbone.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    
                    if client_model.projector != None: 
                        for w, w_t in zip(client_model.projector.parameters(), ref_model.projector.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)

                    loss += args.l2_lamb * w_diff

                    w_diff_l1 = torch.tensor(0., device=device)
                    for w in client_model.backbone.parameters():
                        w_diff_l1 += torch.norm(w, 1)

                    if client_model.projector != None: 
                        for w in client_model.projector.parameters():
                            w_diff_l1 += torch.norm(w, 1)
                    
                    loss += args.l1_lamb * w_diff_l1

            loss.backward()
            optimizer.step()

            pid = os.getpid()
            print(f"{pid} [{batch_idx}/{len(loader)}] loss {loss.item()}")

    state_dict = copy.deepcopy(client_model.state_dict())

    
    # Multiprocessing Queue
    if q != None:
        q.put(state_dict)
        done.wait()

    return state_dict

def test_server_model(args, dataset, device, sup_model):
    model = copy.deepcopy(sup_model).to(device)
    model = model.eval()
    loader = DataLoader(
        dataset,
        batch_size = len(dataset), 
        shuffle = True,
        num_workers = args.num_workers,
        drop_last = True, 
    )

    images, labels = next(iter(loader))
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(images, return_logits=True)
        loss = FL_criterion(device)(preds, labels)
        loss_value = loss.item()

        _, top1_preds = torch.max(preds.data, -1)
        _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

        top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
        top5 = 0
        for label, pred in zip(labels, top5_preds):
            if label in pred:
                top5 += 1

        top5 /= labels.size(0)
        top5 *= 100

    return loss_value, top1, top5

def test_client_model(args, finetune_set, test_set, device, unsup_model, finetune=True, freeze=False, finetune_epochs = 5):
    # if finetune, finetune epochs must be > 0 
    assert (not finetune) or (finetune_epochs > 0 and finetune)
    
    if finetune:
        orig_state_dict = copy.deepcopy(unsup_model.state_dict())

        # load finetuned dict
        finetuned_state_dict = finetune_client_model(args, finetune_set, device, unsup_model, epochs=finetune_epochs, freeze=freeze)
        unsup_model.load_state_dict(finetuned_state_dict)

        # test
        loss, top1, top5 = test_server_model(args, test_set, device, unsup_model)

        # reload original state_dict
        unsup_model.load_state_dict(orig_state_dict)
        return loss, top1, top5
    
    # test using whole testset
    loss, top1, top5 = test_server_model(args, test_set, device, unsup_model)
    return loss, top1, top5 


def BYOL_loss(device, p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)

def FedSSL_loss(device, outputs, teacher_outputs, args):
    alpha = args.fsl_alpha
    T = args.fsl_temperature
    KD_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return KD_loss

def FL_criterion(device):
    return nn.CrossEntropyLoss().to(device)

def SimCLR_criterion(device):
    return nn.CrossEntropyLoss().to(device)

def SimSiam_criterion(device):
    return nn.CosineSimilarity(dim=1).to(device)

def SimSiam_loss(device, p1, p2, z1, z2):
    criterion = SimSiam_criterion(device)
    return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

def NCE_loss(device, temperature, feature1, feature2, labels=None):
    # features = (local batch size * 2, out_dim) shape 
    # feature1, feature2 = torch.tensor_split(features, 2, 0)
    # feature1, 2 = (local batch size, out_dim) shape
    feature1, feature2 = F.normalize(feature1, dim=1), F.normalize(feature2, dim=1)
    batch_size = feature1.shape[0]
    LARGE_NUM = 1e9
    
    # each example in feature1 (or 2) corresponds assigned to label in [0, batch_size)      
    masks = torch.eye(batch_size, device=device)
    
    
    logits_aa = torch.matmul(feature1, feature1.T) / temperature #similarity matrix 
    logits_aa = logits_aa - masks * LARGE_NUM
    
    logits_bb = torch.matmul(feature2, feature2.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    
    logits_ab = torch.matmul(feature1, feature2.T) / temperature
    logits_ba = torch.matmul(feature2, feature1.T) / temperature
    

    criterion = SimCLR_criterion(device)

    if labels == None:
        labels = torch.arange(0, batch_size, device=device, dtype=torch.int64)
        
    loss_a = criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
    
    loss = loss_a + loss_b
    return loss / 2
