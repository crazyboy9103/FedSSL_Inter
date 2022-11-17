import numpy as np
import torch
from multiprocessing import Manager
import torch.multiprocessing as mp
import wandb
from datetime import datetime
import copy 


from options import args_parser
from trainers import test_server_model, test_client_model, train_client_model, train_server_model, finetune_client_model
from models import ResNet18Model
from utils import average_weights, set_seed, Timer, MonitorGPU, kNN_helpers, gradient_diversity, collect_weights, bn_divergence

from data_utils import CIFAR10_Train, CIFAR10_Test

if __name__ == '__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

    args = args_parser()

    wandb_name = now if args.wandb_tag == "" else args.wandb_tag
    wandb_writer = wandb.init(
        dir="./",
        name = wandb_name,
        project = "Fed", 
        resume = "never",
        id = now,
    )
    
    set_seed(args.seed)
    if args.parallel == True:
        mp.set_start_method('spawn', force=True)

    ser_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # Set the model to train and send it to device.
    server_model = ResNet18Model(args).to(ser_device)#.share_memory() # this is the supervised model
    client_model = ResNet18Model(args).to(ser_device)#.share_memory() # online model for FL
    
    train_set = CIFAR10_Train(args)
    test_set = CIFAR10_Test(args)

    # number of participating clients
    num_clients_part = int(args.frac * args.num_users)
    assert num_clients_part > 0
    
    # supervsied server model is required only for FedSSL or FedBYOL experiments
    should_send_server_model = False
    if args.agg == "FedSSL" or args.exp in ["FedBYOL", "FedMatch"]:
        should_send_server_model = True


    gpu_monitor = MonitorGPU(600)
    timer = Timer()

    # Client train
    local_weights = {}

    # Training 
    for epoch in range(args.epochs):
        timer.set_timer()
        if args.exp != "centralized":
            if should_send_server_model or args.exp == "FedRGD":          
                state_dict = train_server_model(
                    args = args,
                    dataset = train_set.get_server_set(),
                    device = ser_device,
                    sup_model = server_model,
                    epochs = args.server_epochs
                )

                server_model.load_state_dict(state_dict)

            # For rest SSL methods, must train client model with supervised set
            if args.exp in ["FixMatch", "PseudoLabel", "SimCLR", "SimSiam", "BYOL", "FedMatch", "FLSL"] and not should_send_server_model:
                state_dict = train_server_model(
                    args = args, 
                    dataset = train_set.get_server_set(), 
                    device = ser_device, 
                    sup_model = client_model,
                    epochs=args.server_epochs, 
                )
                client_model.load_state_dict(state_dict)   
            # --------------------------------------------------------------------------------------------
            
            # Select clients for training in this round
            part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
            
            # multiprocessing queue
            processes = []
            
            done = mp.Event()
            q = mp.Queue()
            
            for i, client_id in enumerate(part_users_ids):
                client_set = train_set.get_client_set(client_id)
                curr_device = torch.device(f"cuda:{i % torch.cuda.device_count()}")
                if args.parallel:
                    p = mp.Process(target = train_client_model, args=(
                        args, 
                        client_set, 
                        curr_device,
                        server_model if should_send_server_model else None,
                        client_model,
                        args.local_ep,
                        q,
                        done, # requires done event to wait until tensor is accessed from parent process
                        helpers[i] if args.exp == "FedMatch" and epoch > 0 else None, # https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
                    ))
                    
                    processes.append(p)
                    p.start()

                else:
                    client_state_dict = train_client_model(
                        args = args, 
                        dataset = client_set, 
                        device = curr_device, 
                        sup_model = server_model if should_send_server_model else None,
                        unsup_model = client_model, 
                        epochs = args.local_ep, 
                        q = None,
                        helpers = helpers[i] if args.exp == "FedMatch" and epoch > 0 else None
                    )

                    local_weights[i] = client_state_dict

            if args.parallel:
                # q: mp.Queue is blocking object, thus works as join
                for i in range(num_clients_part):
                    weight = q.get()        
                    local_weights[i] = weight
                    del weight
                
                # must retain the subprocesses until weights are fetched at main process
                done.set()
                
            collect_weights(local_weights) # bring weights to cuda:0 
            # FedMatch
            if epoch % 10 == 0 and args.exp == "FedMatch":
                helpers = kNN_helpers(args, client_model, local_weights)

            
            # Extract gradient diversity (FedRGD definition)
            grad_div = gradient_diversity(local_weights, client_model.state_dict())
            bn_div = bn_divergence(local_weights, client_model.state_dict())
            # --------------------------------------------------------------------------------------------
            # FedAvg
            if args.exp == "FedRGD":
                # Follow FedRGD EMNIST experiment's hyperparam
                groups = 2
                assert (num_clients_part % groups) == 0, "num clients must be divisible by num groups"

                idxs = set(range(num_clients_part))
                group_idxs = {}
                
                for group_idx in range(groups):
                    random_idxs = np.random.choice(tuple(idxs), num_clients_part // groups, replace=False)
                    idxs = idxs - set(random_idxs)
                    group_idxs[group_idx] = random_idxs

                
                grouped_averages = {}
                for group_id, idxs in group_idxs.items():
                    group_weights = {j: local_weights[idx] for j, idx in enumerate(idxs)}
                    group_weights[len(idxs)] = copy.deepcopy(server_model.state_dict())
                    
                    grouped_avg_weights = average_weights(group_weights)
                    grouped_averages[group_id] = grouped_avg_weights
                
                
                grouped_avg_weight = average_weights(grouped_averages)
            else:
                grouped_avg_weight = average_weights(local_weights)

            client_model.load_state_dict(copy.deepcopy(grouped_avg_weight))           
            # --------------------------------------------------------------------------------------------
            # Linear eval
            loss, top1, top5 = test_client_model(
                args = args,
                finetune_set = test_set.get_finetune_set(),
                test_set = test_set.get_test_set(),
                device = ser_device,
                unsup_model = client_model,
                finetune = True, # finetune trains the classifier only  
                freeze = args.freeze,   # freeze backbone
                finetune_epochs = args.finetune_epoch 
            )

        # if centralized 
        else:    
            state_dict = train_server_model(
                args = args,
                dataset = train_set.get_server_set(),
                device = ser_device,
                sup_model = server_model,
                epochs = args.server_epochs
            )

            server_model.load_state_dict(state_dict)
            
            loss, top1, top5 = test_server_model(
                args = args, 
                dataset = test_set.get_test_set(),
                device = ser_device,
                sup_model = server_model
            )

            grad_div = 0
            bn_div = {}

        print("#######################################################")
        print(f' \nAvg Validation Stats after {epoch+1} global rounds')
        print(f'Validation Loss     : {loss:.2f}')
        print(f'Validation Accuracy : top1/top5 {top1:.2f}%/{top5:.2f}%\n')
        print("#######################################################")
        print(f"Took {timer.see_timer()}s")
        wandb_log = {
            "test_loss_server": loss, 
            "top1_server": top1,
            "top5_server": top5,
            "epoch": epoch,
            "grad_div": grad_div 
        }
        wandb_log.update(bn_div) # bn div is recorded layer by layer
        wandb_writer.log(wandb_log)

    gpu_monitor.stop()

