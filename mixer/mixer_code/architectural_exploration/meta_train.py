import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warmup_scheduler
import numpy as np
import time

from utils import rand_bbox

class Trainer(object):
    def __init__(self, model, device):
        self.device = device
        self.clip_grad = 1.0
        self.cutmix_beta = 1.0
        self.cutmix_prob = 0.5
        self.model = model

        self.epochs = 1
        #self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
        #                  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.000001)
        WARMUP = 2
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1.,
                         total_epoch=WARMUP, after_scheduler=self.base_scheduler)

        self.scaler = torch.amp.GradScaler()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.

        self.layer_grad_norm_df = []
        self.layer_grad_norm_bp = []

    def _train_one_step(self, batch, det):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        #################################################################################
        self.optimizer.zero_grad()
        r = np.random.rand(1)
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
                out, _ = self.model(img, det=det)
                loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            with torch.amp.autocast(device_type='cuda'):
                out, _ = self.model(img, det=det)
                loss = self.criterion(out, label)
        
        #start_time = time.time()
        self.scaler.scale(loss).backward()

        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
    
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
            out, intermid = self.model(img, eval_=True)
            loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)
        for i in range(len(self.epoch_corr_list)):
            self.epoch_corr_list[i] = self.epoch_corr_list[i] + intermid[i].argmax(dim=-1).eq(label).sum(-1)

    def fit(self, train_dl, test_dl):
        train_loss = []
        val_loss = []
        best_accuracy = -1.0
        for epoch in range(1, self.epochs+1):

            print("Epoch: ", epoch)
            print("LR: ", self.scheduler.get_last_lr()[0])
            print('Layer Weights: ', [param.item() for param in self.model.weights])
            
            # self.train_loss = 0.0
            # num_imgs = 0.
            # det = False
            # for batch in train_dl:
            #     self._train_one_step(batch, det)
            #     num_imgs += batch[0].size(0)
            
            # self.train_loss /= num_imgs
            # print("Train loss: ", self.train_loss.item())
    
            # self.scheduler.step()
            
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

            #train_loss.append(self.train_loss.item())
            #val_loss.append(self.epoch_loss.item())
        
        return train_loss, val_loss

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)             
