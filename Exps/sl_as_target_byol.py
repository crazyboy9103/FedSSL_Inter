import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms as T
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

class SimCLRTransformWrapper(object):
    def __init__(self):
        self.no_transform = NoTransform()
        self.base_transform = SimCLRTransform()
        self.n_views = 2

    def __call__(self, x):
        return self.no_transform(x), [self.base_transform(x) for i in range(self.n_views)] # two views by default

def SimCLRTransform():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.GaussianBlur(kernel_size=int(0.1 * 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def NoTransform():
    return T.Compose([
        T.ToTensor(), 
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

class BYOLProjector(nn.Module):
    def __init__(self, in_features, hidden_dim, output_dim):
        super(BYOLProjector, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)

class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()
        self.backbone, self.projector, self.classifier = self.build_model()
        
    def build_model(self):
        backbone = models.resnet18(pretrained = False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        projector = BYOLProjector(in_features, 512, 512)
        classifier = nn.Linear(in_features, 10, bias=True)
        return backbone, projector, classifier
    
    def forward(self, x, return_embedding=False, return_logits=False):
        if return_embedding == False and return_logits == False:
            z = self.backbone(x)
            p = self.projector(z)
            return z, p
        
        if return_embedding == True:
            z = self.backbone(x)
            return z
    
        if return_logits == True:
            z = self.backbone(x)
            return self.classifier(z)
        
def BYOL_loss(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)


def FL_criterion():
    return nn.CrossEntropyLoss().cuda()

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    
import copy
batch_sizes = [32, 64, 128, 256, 512]

acc_list = {batch_size: [] for batch_size in batch_sizes}
train_loss_list = {batch_size: [] for batch_size in batch_sizes}

for batch_size in batch_sizes:
    epochs = 100

    online_net = nn.DataParallel(ResNet18Model(), output_device=0).cuda()
    target_net = nn.DataParallel(ResNet18Model(), output_device=0).cuda()


    cifar = datasets.CIFAR10(root="~/FedSSL_recent/data/cifar", train=True, download=True, transform=SimCLRTransformWrapper())
    cifar_test = datasets.CIFAR10(root="~/FedSSL_recent/data/cifar", train=False, download=True, transform=NoTransform())

    cifar_subset = Subset(cifar, list(range(10000)))
    cifar_sup_subset = Subset(cifar, list(range(10000, 20000)))


    loader = DataLoader(cifar_subset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(cifar_test, batch_size=32, num_workers=4)
    sup_loader = DataLoader(cifar_sup_subset, batch_size=batch_size, num_workers=4)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, online_net.parameters()), lr=0.001)
    linear_cl_optimizer = optim.Adam(filter(lambda p: p.requires_grad, online_net.module.classifier.parameters()), lr=0.001)
    sup_optimizer = optim.Adam(filter(lambda p: p.requires_grad, target_net.parameters()), lr=0.001)
    
    for epoch in range(epochs):
        target_net = target_net.train()
        for batch_idx, (images, labels) in enumerate(sup_loader):
            sup_optimizer.zero_grad()
            orig_images, (_, _) = images
            preds = target_net(orig_images, return_logits=True)
            loss = FL_criterion()(preds, labels.cuda())
            loss.backward()
            sup_optimizer.step()
            if batch_idx % 10 == 0:
                print("sup loss", loss.item())
            
        running_train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(loader):
            optimizer.zero_grad()

            orig_images, (views1, views2) = images
            z1, p1 = online_net(views1)
            z2, p2 = online_net(views2)
            with torch.no_grad():
                target_net = target_net.eval()
                target_z1, target_p1 = target_net(views1)
                target_z2, target_p2 = target_net(views2)

            loss = (BYOL_loss(p1, target_z2.detach()) + BYOL_loss(p2, target_z1.detach())).mean()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            if batch_idx % 10 == 0:
                print("byol loss", loss.item())
            
        running_train_loss /= len(loader)
        train_loss_list[batch_size].append(running_train_loss)

        online_net.module.backbone.requires_grad_(False)
        online_net.module.classifier.requires_grad_(True)
        online_net.module.classifier.reset_parameters()

        accs = []
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx < 0.5 * len(test_loader):
                linear_cl_optimizer.zero_grad()

                preds = online_net(images, return_logits=True)
                loss = FL_criterion()(preds, labels.cuda())
                loss.backward()
                linear_cl_optimizer.step()
            else:
                online_net = online_net.eval()
                with torch.no_grad():
                    preds = online_net(images, return_logits=True)
                    acc = (preds.argmax(dim=1) == labels.cuda()).sum().item() / labels.size(0)
                    accs.append(acc)

                    if batch_idx % 10 == 0:
                        print("byol acc", acc)
            
        whole_acc = sum(accs)/len(accs)
        
        acc_list[batch_size].append(whole_acc)

        online_net.module.backbone.requires_grad_(True)

import matplotlib.pyplot as plt
plt.figure(1)
plt.title("Variation of linear eval acc of BYOL")
plt.ylabel("Accuracy (%)")
plt.xlabel("Epoch")
for i, (batch_size, accs) in enumerate(acc_list.items()):
    plt.plot(accs, label=batch_size)
plt.legend()
plt.tight_layout()
plt.savefig("sl_as_target_byol.png")