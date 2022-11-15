import numpy as np
from torchvision import transforms as T, datasets
from torch.utils.data import Subset, DataLoader
import os

class SimCLRTransform(object):
    def __init__(self):
        self.base = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.GaussianBlur(kernel_size=int(0.1 * 32)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.no = NoTransform()
    
    def __call__(self, x):
        return self.no(x), self.base(x), self.base(x)

class NoTransform(object):
    def __init__(self):
        self.base = T.Compose([
            T.ToTensor(), 
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def __call__(self, x):
        return self.base(x)
    
class FixMatchTransform(object):
    def __init__(self):
        self.weak = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.125, 0.125)), # No rotation, just translation
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.strong = T.Compose([
            T.RandAugment(num_ops=3), 
            T.ToTensor(), 
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.no = NoTransform()
        
    
    def __call__(self, x):
        return self.no(x), self.weak(x), self.strong(x)

class FedMatchTransform(object):
    def __init__(self):
        self.rand = T.Compose([
            T.RandAugment(num_ops=3), 
            T.ToTensor(), 
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.no = NoTransform()
    
    def __call__(self, x):
        return self.no(x), self.rand(x)

    
class CIFAR10_Test():
    def __init__(self, args):
        test_dataset = datasets.CIFAR10(
            root = os.path.join(args.data_path, "cifar"), 
            train=False, 
            download=True,
            transform=NoTransform()
        )
        finetune_idxs, test_idxs = self.get_partition(N = len(test_dataset))
        
        self.finetune_set, self.test_set = Subset(test_dataset, finetune_idxs), Subset(test_dataset, test_idxs) 
        
        self.finetune_bs = args.finetune_bs
        
        # test batch size = test length
        self.test_bs = len(test_idxs)
        
        self.num_workers = args.num_workers
        
    def get_partition(self, N):
        finetune_idxs = set(np.random.choice(range(N), N // 2, replace=False))
        test_idxs = set(range(N)) - finetune_idxs
        return list(finetune_idxs), list(test_idxs)

    def get_finetune_set(self):
        return self.finetune_set
    
    def get_test_set(self):
        return self.test_set

    
        
class CIFAR10_Train():
    def __init__(self, args):
        self.train_dataset = datasets.CIFAR10(
            root=os.path.join(args.data_path, "cifar"),
            train=True, 
            download=True, 
        )
        
        self.dict_users, self.server_data_idx = self.get_partition(args.num_users, args.num_items, args.alpha)
        
        self.server_transform = NoTransform()
        
        self.local_bs = args.local_bs
        self.server_bs = args.server_bs
        self.server_num_items = args.server_num_items
        
        transforms = {
            "FLSL": NoTransform(), 
            "FixMatch": FixMatchTransform(),
            "FedMatch": FedMatchTransform(),
            "PseudoLabel": NoTransform(),    #TODO
            "FedRGD": NoTransform(),         #TODO
            "SimCLR": SimCLRTransform(), 
            "SimSiam": SimCLRTransform(), 
            "BYOL": SimCLRTransform(), 
            "FedBYOL": SimCLRTransform(), 
        }
        self.client_transform = transforms[args.exp]
        self.num_workers = args.num_workers

    def get_partition(self, num_users, num_items, alpha):
        labels = self.train_dataset.targets
    
        # Collect idxs for each label
        idxs_labels = {i: set() for i in range(10)}
        for idx, label in enumerate(labels):
            idxs_labels[label].add(idx)


        # 10 labels
        class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
        class_dist = (class_dist * num_items).astype(int)

        if num_users == 1:
            for _class, class_num in enumerate(class_dist[0]):
                if class_num > len(idxs_labels[_class]):
                    class_dist[0][_class] = len(idxs_labels[_class])

        else:   
            for _class, class_num in enumerate(class_dist.T.sum(axis=1)):
                assert class_num < len(idxs_labels[_class]), "num_items must be smaller"


        dict_users = {i: set() for i in range(num_users)}
        dists = {i: [0 for _ in range(10)] for i in range(num_users)}

        for client_id, client_dist in enumerate(class_dist):
            for _class, num in enumerate(client_dist):
                sample_idxs = idxs_labels[_class]
                dists[client_id][_class] += num

                sampled_idxs = set(np.random.choice(list(sample_idxs), size=num, replace=False)) 
                # accumulate
                dict_users[client_id].update(sampled_idxs)

                # exclude assigned idxs
                idxs_labels[_class] = sample_idxs - sampled_idxs

        for i, data_idxs in dict_users.items():
            dict_users[i] = list(data_idxs)

        server_data_idx = {i: list(idxs) for i, idxs in idxs_labels.items()}

        return dict_users, server_data_idx

    def get_client_set(self, client_id):
        self.train_dataset.transform = self.client_transform
        subset = Subset(self.train_dataset, self.dict_users[client_id])
        return subset
    
    def get_server_set(self):
        idxs = []
        for _class, class_idxs in self.server_data_idx.items():
            selected_idxs = np.random.choice(class_idxs, self.server_num_items, replace=False)
            idxs.extend(selected_idxs.tolist())
        
        self.train_dataset.transform = self.server_transform
        subset = Subset(self.train_dataset, idxs)

        return subset
    