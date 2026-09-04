import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warmup_scheduler
import numpy as np
import time
import json

from utils import rand_bbox

class Trainer(object):
    def __init__(self, model, device):
        self.device = device
        self.clip_grad = 1.0
        self.cutmix_beta = 1.0
        self.cutmix_prob = 0.5
        self.model = model

        self.epochs = 300
        #self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
        #                  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.000001)
        WARMUP = 10
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1.,
                         total_epoch=WARMUP, after_scheduler=self.base_scheduler)

        self.scaler = torch.amp.GradScaler()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.

        #self.layer_grad_norm_df = []
        #self.layer_grad_norm_bp = []
        self.layer_grad_norm_bp = [[] for _ in range(16)]
        self.K = 15
        self.early_loss_weights = [0] * 16
        for i in range(0, 16, self.K):
            self.early_loss_weights[i] = 1

        self.layers_dict = [0 for _ in range(17)]

    def _train_one_step(self, batch, det, S_t):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        #################################################################################
        self.optimizer.zero_grad()
        r = np.random.rand(1)
        e_scale = 0.2
        e_l = [e_scale*sum(range(l + 1)) for l in range(15)]
        self.early_loss_weights = circular_shift_ones(self.early_loss_weights, 1)
        e_l.append(15+e_l[-1])
        weights = [c*e for c,e in zip(self.early_loss_weights, e_l)]
        norm_factor = sum(weights)
        weights_for_layers = [w/norm_factor for w in weights]
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            with torch.amp.autocast(device_type='cuda'):
                out, intermid = self.model(img, S_t, det=False)
                loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
                # intermid_loss = []
                # for idx, i in enumerate(intermid[1:]):
                #     l = self.criterion(i, target_a) * lam + self.criterion(i, target_b) * (1. - lam)
                #     #wei = 2*(idx+1)/(16*17)
                #     wei = weights_for_layers[idx]
                #     intermid_loss.append(wei*l)
                # loss = sum(intermid_loss) 
       
        else:
            # compute output
            with torch.amp.autocast(device_type='cuda'):
                out, intermid = self.model(img, S_t, det=False)
                loss = self.criterion(out, label)
                # intermid_loss = []
                # for idx, i in enumerate(intermid[1:]):
                #     l = self.criterion(i, label)
                #     #wei = 2*(idx+1)/(16*17)
                #     wei = weights_for_layers[idx]
                #     intermid_loss.append(wei*l)
                # loss = sum(intermid_loss) 
          
        
        self.scaler.scale(loss).backward()
        
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        
        for i, l in enumerate(self.model.mixer_layers):
            self.layer_grad_norm_bp_epoch[i].append(get_layer_grad(self.model, i))
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
    

        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)

        self.train_loss += loss * img.size(0)


    @torch.no_grad
    def _test_one_step(self, batch):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out, intermid = self.model(img, S_t=1, eval_=True)
            loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)
        
        #############
        # first_match_layer = -1 
        # final_pred = out.argmax(dim=-1)  
        # for i, inter_pred in enumerate(intermid):
        #     if inter_pred.argmax(dim=-1).eq(final_pred): 
        #         first_match_layer = i
        #         break 
        # self.layers_dict[first_match_layer] +=1
        #############
        for i in range(len(self.epoch_corr_list)):
            self.epoch_corr_list[i] = self.epoch_corr_list[i] + intermid[i].argmax(dim=-1).eq(label).sum(-1)

    def fit(self, train_dl, test_dl):
        train_loss = []
        val_loss = []
        best_accuracy = -1.0
        total_steps = len(train_dl) * self.epochs
        for epoch in range(1, self.epochs+1):
            print("Epoch: ", epoch)
            print("LR: ", self.scheduler.get_last_lr()[0])
            self.train_loss = 0.0
            num_imgs = 0.
            det = False
            #if epoch % 5 == 0: det = False
            self.layer_grad_norm_bp_epoch = [[] for _ in range(16)]
            for step, batch in enumerate(train_dl):
                t = epoch*(step+1)
                S_t = np.exp((t*np.log(2)/(total_steps-1)))-1 
                self._train_one_step(batch, det, S_t)
                num_imgs += batch[0].size(0)
            
            self.train_loss /= num_imgs
            print("Train loss: ", self.train_loss.item())
            
            for i in range(16):
                self.layer_grad_norm_bp[i].append(np.mean(self.layer_grad_norm_bp_epoch[i]))

            if epoch == 1298: # else save every x epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                },filename=f'checkpoints/lcn_300.pth.tar')
            if epoch > 1300 and epoch % 20 == 0: # else save every x epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                },filename=f'checkpoints/lcn_{epoch}.pth.tar')

            self.scheduler.step()
            
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            self.epoch_acc_list = []
            self.epoch_corr_list = [0. for _ in range(17)]
            for batch in test_dl:
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            for i in self.epoch_corr_list:
                self.epoch_acc_list.append((i / num_imgs).item())

            print("Val loss: ", self.epoch_loss.item())
            print("Val acc: ", self.epoch_acc.item())
            print("Intermediate Layers:")
            print(self.epoch_acc_list)
            print("")
            #print(self.layers_dict)

            # if self.epoch_acc.item() > best_accuracy:
            #     best_accuracy = self.epoch_acc.item()
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': self.model.state_dict(),
            #         'optimizer' : self.optimizer.state_dict(),
            #         'scheduler': self.scheduler.state_dict()
            #         },filename=f'checkpoints/checkpoint_best_model.pth.tar')
                
            #print("Avg. Inference Time: ", sum(self.inference_time_list) / len(self.inference_time_list))

            train_loss.append(self.train_loss.item())
            val_loss.append(self.epoch_loss.item())

        with open('workspace/grads/res_full_300ep.json', 'w') as file:
            json.dump(self.layer_grad_norm_bp, file)
        return train_loss, val_loss

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)             

def get_layer_grad(model, idx):
    fc1_grad_means = model.mixer_layers[idx].mlp2.fc1.weight.grad.norm().item()
    #fc2_grad_means = model.mixer_layers[idx].mlp2.fc2.weight.grad#.norm().item()
    return fc1_grad_means
    return (fc1_grad_means + fc2_grad_means) / 2


def circular_shift_ones(lst, K):
    n = len(lst)
    result = [0] * n  
    for i in range(n):
        if lst[i] == 1:
            new_position = (i + K) % n
            result[new_position] = 1
    return result
