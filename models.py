from torchvision import models
from torch import nn
import torch


class SimCLRProjector(nn.Module):
    def __init__(self, in_features, hidden_dim, output_dim):
        super(SimCLRProjector, self).__init__()
        
        self.projector =  nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
    
    def forward(self, x):
        return self.projector(x)        

class BYOLProjector(nn.Module):
    def __init__(self, in_features, hidden_dim, output_dim, gn=False, num_groups=4):
        super(BYOLProjector, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim) if gn == False else nn.GroupNorm(num_groups, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)

class SimSiamProjector(nn.Module):
    def __init__(self, hidden_dim, output_dim, gn=False, num_groups=4):
        super(SimSiamProjector, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim) if gn == False else nn.GroupNorm(num_groups, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)
        

class SimSiamEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, output_dim, gn=False, num_groups=4):
        super(SimSiamEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim) if gn == False else nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim) if gn == False else nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False) if gn == False else nn.GroupNorm(num_groups, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)
                
class ResNet18Model(nn.Module):
    def __init__(self, args):
        super(ResNet18Model, self).__init__()
        self.backbone, self.projector, self.classifier = self.build_model(args)
        
    def build_model(self, args):
        backbone = models.resnet18(pretrained = args.pretrained)
        in_features = backbone.fc.in_features
        
        if args.gn == True:
            backbone.bn1 = nn.GroupNorm(
                args.num_groups, 
                backbone.bn1.num_features
            )
            layers = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
            for i in range(len(layers)):
                if i != 0:
                    layers[i][0].downsample[1] = nn.GroupNorm(
                        args.num_groups, 
                        layers[i][0].downsample[1].num_features
                    )
                layers[i][0].bn1 = nn.GroupNorm(
                    args.num_groups, 
                    layers[i][0].bn1.num_features
                )
                layers[i][0].bn2 = nn.GroupNorm(
                    args.num_groups, 
                    layers[i][0].bn2.num_features
                )
                layers[i][1].bn1 = nn.GroupNorm(
                    args.num_groups, 
                    layers[i][1].bn1.num_features
                )
                layers[i][1].bn2 = nn.GroupNorm(
                    args.num_groups, 
                    layers[i][1].bn2.num_features
                )
         
        
        if args.exp in ["FLSL", "centralized", "FixMatch", "PseudoLabel", "FedRGD"]:
            backbone.fc = nn.Identity()
            projector = None
            classifier = nn.Linear(in_features, 10, bias=True)
        
        
        elif args.exp == "SimCLR":
            backbone.fc = nn.Identity()
            projector = SimCLRProjector(in_features, args.hidden_dim, args.output_dim)
            classifier = nn.Linear(in_features, 10, bias=True)
        
        elif args.exp == "SimSiam":
            backbone.fc = SimSiamEncoder(in_features, args.hidden_dim, args.output_dim, args.gn, args.num_groups)
            projector = SimSiamProjector(args.hidden_dim, args.output_dim, args.gn, args.num_groups)
            classifier = nn.Linear(args.output_dim, 10, bias=True)
        
        elif args.exp in ["BYOL", "FedBYOL"]:
            backbone.fc = nn.Identity()
            projector = BYOLProjector(in_features, args.hidden_dim, args.output_dim, args.gn, args.num_groups)
            classifier = nn.Linear(in_features, 10, bias=True)
        
        elif args.exp == "FedMatch":
            backbone.fc = nn.Identity()
            projector = SimSiamProjector(args.hidden_dim, in_features, args.gn, args.num_groups)
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
        