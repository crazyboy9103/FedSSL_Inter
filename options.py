#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os
import time, datetime

def str2bool(v):
    #https://eehoeskrap.tistory.com/521
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getTimestamp():
    utc_timestamp = int(time.time())
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime("%Y_%m_%d_%H_%M_%S")
    return date

def args_parser():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model',            type=str,       default='resnet18',  help='resnet18|resnet50')
    parser.add_argument('--pretrained',       type=str2bool,  default=False,       help='pretrained backbone')
    parser.add_argument('--num_classes',      type=int,       default=10,          help="number of classes")
    
    # GN (BN used by default) 
    parser.add_argument('--gn',               type=str2bool,  default=False,       help="group normalization")
    parser.add_argument('--num_groups',       type=int,       default=4,       help="group normalization")

    # Experimental setup
    parser.add_argument("--exp",              type=str,       default="FLSL",    choices=["SimCLR", "SimSiam", "centralized", "FedBYOL", "FLSL", "BYOL", "FixMatch", "PseudoLabel", "FedMatch", "FedRGD"])
    parser.add_argument("--wandb_tag",        type=str,       default="",        help="optional tag for wandb logging")
    parser.add_argument("--alpha",            type=float,     default=0.5,       help="dirichlet param 0<alpha controls iidness (0:non iid)")
    
    # SimCLR
    parser.add_argument("--temperature",      type=float,     default=0.1,       help="softmax temperature")

    # SimCLR & SimSiam
    parser.add_argument("--output_dim",         type=int,       default=512,       help="output embedding dim")
    parser.add_argument("--hidden_dim",         type=int,       default=512)

    # Data setup
    # RandomCrop target size
    parser.add_argument('--target_size',      type=int,       default=32,        help="augmentation target width (=height)")
    parser.add_argument("--num_users",      type=int,      default=10,        help="num users")
    parser.add_argument("--num_items",      type=int,      default=32,        help="num data each client holds")
    parser.add_argument("--iid",            type=str2bool, default=True,      help="iid on clients")
    parser.add_argument('--dataset',        type=str,       default='cifar',  choices=["mnist", "cifar"],              help="mnist|cifar")

    # 
    parser.add_argument('--freeze',           type=str2bool,  default=False,      help='freeze feature extractor during linear eval')
    # Fixmatch
    parser.add_argument('--threshold',      type=float, default = 0.95)
    # FedSSL
    parser.add_argument('--fsl_alpha',      type=float, default=0.95)
    parser.add_argument('--fsl_temperature', type=float, default=6)
    parser.add_argument('--mse_ratio', type=float, default=1)    

    # FedProx
    parser.add_argument('--mu',             type=float,     default=0.01)

    # FedMatch
    parser.add_argument('--topH',          type=int,      default=2)
    parser.add_argument('--tau',           type=float,    default=0.85)
    parser.add_argument('--iccs_lambda',   type=float,    default=1e-2)
    parser.add_argument('--l2_lamb',       type=float,     default=10)
    parser.add_argument('--l1_lamb',       type=float,     default=1e-5)
    # FL
    parser.add_argument('--epochs',         type=int,      default=200,        help="number of rounds of training")
    parser.add_argument('--frac',           type=float,    default=1,       help='the fraction of clients: C')
    parser.add_argument('--local_ep',       type=int,      default=10,         help="the number of local epochs: E")
    parser.add_argument('--local_bs',       type=int,      default=32,         help="local batch size")
    parser.add_argument('--lr',             type=float,    default=0.001,      help='learning rate')
    
    
    # Server
    parser.add_argument('--server_epochs', type=int,  default=5)
    parser.add_argument('--server_num_items', type=int,  default=1000,  help="number of items per class used for training at server")
    parser.add_argument('--server_bs', type=int, default=32, help="server batch size for iid pre-training at server")
    
    
    parser.add_argument('--bn_stat_momentum', type=float, default=0.1, help="bn stat EMA momentum should be set < 0.1")

    # Train setting
    parser.add_argument("--parallel",       type=str2bool,  default=True,                help="parallel training with threads")
    parser.add_argument("--num_workers",    type=int,       default=8,                    help="num workers for dataloader")
    parser.add_argument('--seed',           type=int,       default=2022,                 help='random seed')
    parser.add_argument('--ckpt_path',      type=str,       default="./checkpoints/checkpoint.pth.tar", help="model ckpt save path")
    parser.add_argument('--data_path',      type=str,       default="./data",             help="path to dataset")
    
    # Finetune setting
    parser.add_argument('--finetune_epoch', type=int,       default=1,                   help='finetune epochs at server')
    parser.add_argument('--finetune_bs',    type=int,       default=32)
    parser.add_argument('--finetune',       type=str2bool,  default=True,                 help="finetune at the server")
    parser.add_argument("--agg",             type=str,     default="FedAvg", choices=["FedAvg", "FedProx", "FedSSL"])

    parser.add_argument('--sanity_check',  type=str2bool,  default=False)
    args = parser.parse_args() 

    get_opts(args)

    return args

def get_opts(args):
    if args.iid == True:
        args.alpha = 100000 # arbitrary large number for iid

    else:
        args.alpha = 0.5    # Non-i.i.d. 
    
    args.num_users = 100 if args.sanity_check != True else 10
    args.num_items = 300 if args.sanity_check != True else 100
    args.epochs  = 100   if args.sanity_check != True else 10
    args.frac = 0.05      if args.sanity_check != True else 0.5
    args.local_ep = 5   if args.sanity_check != True else 1
    args.local_bs = 128

    args.server_epochs = 5      if args.sanity_check != True else 1
    args.server_num_items = 300 if args.sanity_check != True else 100
    args.finetune_epoch = 5 if args.sanity_check != True else 1 
    args.finetune = True 
    args.freeze = True