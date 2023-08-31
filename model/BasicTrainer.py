import torch
import math
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.logger import get_logger
from lib.TrainInits import print_model_parameters
from lib.dataloader import *
from lib.metrics import *

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler, device):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(list(train_loader.get_iterator()))
        if val_loader != None:
            self.val_per_epoch = len(list(val_loader.get_iterator()))
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        total_param = print_model_parameters(model, only_num=False)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)
        self.logger.info(self.model)
        self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        val_loss = []
        val_mape = []
        val_rmse = []
        s1 = time.time()
        with torch.no_grad():
            for iter_, (Tr, label) in enumerate(val_dataloader.get_iterator()):
                Tr = Tr[...,[0,-1]]
                Tr[...,:1] = self.scaler.transform(Tr[...,:1])#B总,T,N,C C=1
                label = label[...,:1]
                assert len(Tr.shape) == 4 and len(label.shape) == 4
                self.optimizer.zero_grad()
                output = self.model([Tr,label])
                output = self.scaler.inverse_transform(output) 
                # almae, alrmse, almape,_,_=All_Metrics(output, label, self.args.mae_thresh, self.args.mape_thresh)
                loss = masked_mae(output,label,0.0).item()
                vmape = masked_mape(output,label,0.0).item()
                vrmse = masked_rmse(output,label,0.0).item()
                val_loss.append(loss)
                val_mape.append(vmape)
                val_rmse.append(vrmse)
        s2 = time.time()
        return np.mean(val_loss), np.mean(val_mape),np.mean(val_rmse),s2-s1 

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        self.train_loader.shuffle()
        for iter_, (Tr, label) in enumerate(self.train_loader.get_iterator()):
            Tr = Tr[...,[0,-1]]
            Tr[...,:1] = self.scaler.transform(Tr[...,:1])#B总,T,N,C C=1
            label = label[...,:1]
            assert len(Tr.shape) == 4 and len(label.shape) == 4
            self.optimizer.zero_grad()#BTNC C=2/1
      
            output = self.model([Tr,label])
            
            output = self.scaler.inverse_transform(output) 
            loss = masked_mae(output,label,0.0)
            loss.backward()
            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            loss = masked_mae(output,label,0.0).item()
            mape = masked_mape(output,label,0.0).item()
            rmse = masked_rmse(output,label,0.0).item()
            train_loss.append(loss)
            train_mape.append(mape)
            train_rmse.append(rmse)
            # total_loss += loss.item()
            if iter_ % self.args.log_step == 0:
                self.logger.info('Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'.format(
                    iter_, train_loss[-1], train_mape[-1], train_rmse[-1]))
        t2 = time.time()
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return np.mean(train_loss), np.mean(train_mape),np.mean(train_rmse),t2-t1

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        train_mape_list = []
        train_rmse_list = [] 
        val_loss_list = []
        val_mape_list = []
        val_rmse_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss, mape, rmse, ttime = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss, vmape, vrmse, stime = self.val_epoch(epoch, val_dataloader)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}, Infer Time: {:.4f}/epoch'
            print(log.format(epoch, train_epoch_loss, mape, rmse, val_epoch_loss, vmape, vrmse, ttime, stime,flush=True))
        
            train_loss_list.append(train_epoch_loss)
            train_mape_list.append(mape)
            train_rmse_list.append(rmse)

            val_loss_list.append(val_epoch_loss)
            val_mape_list.append(vmape)
            val_rmse_list.append(vrmse)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            
            
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, None)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        ttmaes = []
        ttmapes = []
        ttrmses = []
        with torch.no_grad():
            for iter_, (Tr, label) in enumerate(data_loader.get_iterator()):
                Tr = Tr[...,[0,-1]]
                Tr[...,:1] = scaler.transform(Tr[...,:1])#B总,T,N,C C=1
                label = label[...,:1]
                assert len(Tr.shape) == 4 and len(label.shape) == 4
                output = model([Tr,label])
                output = scaler.inverse_transform(output) 
                ttmae, ttrmse, ttmape, _, _=All_Metrics(output, label, args.mae_thresh, args.mape_thresh)

                ttmaes.append(ttmae.item())
                ttmapes.append(ttmape.item())
                ttrmses.append(ttrmse.item())

                y_true.append(label)
                y_pred.append(output)
                
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        print('MAE,MAPE.RMSE',np.mean(ttmaes),np.mean(ttmapes),np.mean(ttrmses))

        np.save(args.log_dir+'/{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save(args.log_dir+'/{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        # mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        # logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        #             mae, rmse, mape*100))
        loss = masked_mae(y_pred,y_true,0.0).item()
        tmape = masked_mape(y_pred,y_true,0.0).item()
        trmse = masked_rmse(y_pred,y_true,0.0).item()
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    loss, trmse, tmape*100))
        

    
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

def _add_weight_regularisation(total_loss, regularise_parameters, scaling=0.03):
    for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
    return total_loss