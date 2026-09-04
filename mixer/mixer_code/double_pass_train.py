import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warmup_scheduler
import numpy as np
import time
import json

from utils import rand_bbox
from soap import SOAP
from layerwise_scheduler import get_mixer_param_groups, StaggeredRampScheduler
class Trainer(object):
    def __init__(self, model, device, args):
        self.device = device
        self.clip_grad = 1.0
        self.cutmix_beta = 1.0
        self.cutmix_prob = 0.5
        self.model = model
        self.args = args
        self.epochs = args.epochs
        print('NUM_EPOCHS: ', self.epochs)

        '''
        # layer-wise
        param_groups = get_mixer_param_groups(self.model, base_lr=0.001)
        self.optimizer = torch.optim.AdamW(param_groups) #, weight_decay=0.001)
        self.scheduler = StaggeredRampScheduler(
            self.optimizer, epochs=300, num_layers=16,
            base_lr=0.001,
            peak_start=10, peak_end=200
        )
        '''
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
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
        #self.layer_grad_norm_bp = [[] for _ in range(16)]
        self.K = 15
        self.early_loss_weights = [0] * 16
        for i in range(0, 16, self.K):
            self.early_loss_weights[i] = 1

        self.layers_dict = [0 for _ in range(17)]

    def _train_one_step(self, batch, det, S_t=0):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        #################################################################################
        self.optimizer.zero_grad()
        r = np.random.rand(1)
        '''
        e_scale = 0.2
        e_l = [e_scale*sum(range(l + 1)) for l in range(15)]
        self.early_loss_weights = circular_shift_ones(self.early_loss_weights, 1)
        e_l.append(15+e_l[-1])
        weights = [c*e for c,e in zip(self.early_loss_weights, e_l)]
        norm_factor = sum(weights)
        weights_for_layers = [w/norm_factor for w in weights]
        '''
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
                out, intermid = self.model(img,  det=False)
                loss_fg = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)      
                out, intermid = self.model(img,  det=True)
                loss_dg = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            with torch.amp.autocast(device_type='cuda'):
                out, intermid = self.model(img, det=False)
                loss_fg = self.criterion(out, label) 
                out, intermid = self.model(img, det=True)
                loss_dg = self.criterion(out, label)
        
        alpha = 0.2
        loss = (1 - alpha) * loss_fg + alpha * loss_dg
        self.scaler.scale(loss).backward()
        
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
         
        self.scaler.step(self.optimizer)
        ######
        if self.args.arch == 'clamped_hybrid' or self.args.arch == 'acn' or self.args.arch == 'long_short':
            with torch.no_grad():
                for alpha in self.model.alphas:
                    alpha.clamp_(0.0, 1.0)
        ######
        self.scaler.update()
    
        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        self.train_loss += loss * img.size(0)


    @torch.no_grad
    def _test_one_step(self, batch):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out, intermid = self.model(img, eval_=True)
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
            '''
            if epoch < 10: det = False
            elif (epoch % 4 == 0): det = False
            '''
            self.layer_grad_norm_bp_epoch = [[] for _ in range(16)]
            for step, batch in enumerate(train_dl):
                self._train_one_step(batch, det)
                num_imgs += batch[0].size(0)
            
            self.train_loss /= num_imgs
            print("Train loss: ", self.train_loss.item())
           
            '''
            if epoch > 250 and epoch < 500 and epoch % 20 == 0: # else save every x epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                },filename=(f'inherent_EE_checkpoints/acn_{epoch}.pth.tar' if self.args.arch == 'acn' else f'inherent_EE_checkpoints/res_{epoch}.pth.tar'))
            '''
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
            if self.args.arch == 'hybrid' or self.args.arch == 'clamped_hybrid' or self.args.arch == 'acn' or self.args.arch == 'long_short':
                print('~~~~~~~~~~~~~~~')
                print('Alphas:')
                print([p.item() for p in self.model.alphas])
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

        '''
        with open('workspace/grads/res_full_300ep.json', 'w') as file:
            json.dump(self.layer_grad_norm_bp, file)
        '''
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

def get_mlp_mixer_param_groups(model, base_lr=1e-3, lr_scale=1.1):
    """
    Assign layer-wise learning rates for MLPMixer.
    Shallow (patch_emb) starts at base_lr, deeper layers get progressively higher LRs.
    lr_scale > 1.0 makes deeper layers train faster.
    """
    param_groups = []

    # patch embedding = shallowest
    param_groups.append({"params": model.patch_emb.parameters(), "lr": base_lr})
    print(f"Patch embedding: lr={base_lr:.6f}")

    # mixer layers
    num_layers = len(model.mixer_layers)
    for i, layer in enumerate(model.mixer_layers):
        lr = base_lr * (lr_scale ** (i + 1))  # grows with depth
        param_groups.append({"params": layer.parameters(), "lr": lr})
        print(f"Mixer layer {i}: lr={lr:.6f}")

    # classifier / head if present
    if hasattr(model, "fc") or hasattr(model, "clf"):
        head = getattr(model, "fc", getattr(model, "clf"))
        lr = base_lr #*(lr_scale ** (num_layers + 1))
        param_groups.append({"params": head.parameters(), "lr": lr})
        print(f"Head: lr={lr:.6f}")

    return param_groups

