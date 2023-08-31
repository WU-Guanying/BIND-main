import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
import torch.nn.functional as F
import torch.optim as optim
from model.BIND import *
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import *
from lib.metrics import *
from lib.TrainInits import print_model_parameters
import os
from os.path import join

#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'PEMSD8'      #PEMSD4 or PEMSD8
MODEL = 'BIND'

#get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

def none_or_other(value):
    if eval(value) is None:
        return None
    else:
        return value

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default='cuda:0', type=str, help='')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--comment', default='', type=str)


#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--period_in', type=int, default=config['data']['period_in'])
args.add_argument('--period_out', type=int, default=config['data']['period_out'])

#model
args.add_argument('--dim_in', type=int, default=config['model']['dim_in'],)#2 single step改为1
args.add_argument('--dim_mid', type=int, default=config['model']['dim_mid'],)#128
args.add_argument('--dim_out', type=int, default=config['model']['dim_out'],)#64
args.add_argument('--dim_output', type=int, default=config['model']['dim_output'],)#1
args.add_argument('--time_divides', type=list, default=config['model']['time_divides'],help='for low-frequency(long-term) signal')#[1]
args.add_argument('--time_divides_', type=list, default=config['model']['time_divides_'],help='for high-frequency(short-term) signal')#[1]
args.add_argument('--L_set', type=eval, default=config['model']['L_set'],help='kernel set for low-frequency signals')#6[8]
args.add_argument('--S_set', type=eval, default=config['model']['S_set'],help='kernel set for high-frequency signals')#2[8]
args.add_argument('--L_reso', type=int, default=config['model']['L_reso'],help='time resolution for low-frequency signals / None')#2[2]
args.add_argument('--S_reso', type=int, default=config['model']['S_reso'],help='time resolution for high-frequency signals / None')#4[2]

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--atol', default=config['train']['atol'], type=float)
args.add_argument('--rtol', default=config['train']['rtol'], type=float)
args.add_argument('--WEIGHTED', default=config['train']['WEIGHTED'], type=bool)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
args.add_argument('--Ablation', default=config['train']['Ablation'], type=none_or_other)
args.add_argument('--missing_test', default=False, type=bool)
args.add_argument('--missing_rate', default=0.1, type=float)

#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args.add_argument('--model_path', default='', type=str)
#log
args.add_argument('--log_dir', default='/home/wuguanying/Hard_nut/BIND-main/runs/', type=str)
args.add_argument('--data_path', default="/home/wuguanying/Hard_nut/data 3/", type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)


args = args.parse_args()
init_seed(args.seed)

# GPU_NUM = args.device
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
device = torch.device(args.device)
torch.cuda.set_device(device)
print(args)

#config log path
save_name = time.strftime("%m-%d-%Hh%Mm")+args.comment+"_"+ args.dataset+"_"+ args.model+"_"+"SIG(in-mid-out-output){"+str(args.dim_in)+str(args.dim_mid)+str(args.dim_out)+str(args.dim_output)+"}"+"lr{"+str(args.lr_init)+"}"+"wd{"+str(args.weight_decay)+"}"
path = '../runs'

log_dir = join(path, args.dataset, save_name)
args.log_dir = log_dir
if (os.path.exists(args.log_dir)):
        print('has model save path')
else:
    os.makedirs(args.log_dir)


data = load_dataset(val_r = args.val_ratio, test_r = args.test_ratio, filename = args.dataset, 
                        period_out = args.period_out, filepath = args.data_path, batch_size = args.batch_size,
                        period_in = args.period_in, device = device, valid_batch_size = None, 
                        test_batch_size = None, horizon = args.horizon)
Node_num = data['Node_num']
train_loader = data['train_loader']
val_loader = data['val_loader']
test_loader = data['test_loader']
scaler = data['scaler']

model = Graph_sampler(dim_in = args.dim_in, dim_out = args.dim_out, batch_num = args.batch_size, Node_num = Node_num, 
                              sequence_length = args.period_in, device = device, 
                              out_sequence_length = args.period_out, dim_mid = args.dim_mid, 
                              dim_output = 1, atol = args.atol, rtol = args.rtol, 
                              WEIGHTED = args.WEIGHTED, time_divides = args.time_divides,time_divides_ = args.time_divides_, 
                              L_set = args.L_set,S_set = args.S_set,L_reso = args.L_reso,S_reso = args.S_reso, Ablation = args.Ablation)

model = model.to(device)

# print(model)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'huber_loss':
    loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
else:
    raise ValueError

optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr_init,
                             weight_decay=args.weight_decay)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler, device)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
