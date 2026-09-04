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
        self.cutmix_beta = 0 #1.0
        self.cutmix_prob = 0.5
        self.model = model

        self.epochs = 10
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

        self.layer_grad_norm_df =  [[] for _ in range(16)]
        self.layer_grad_norm_bp =  [[] for _ in range(16)]
        self.layer_grad_angle =  [[] for _ in range(16)]
        self.current_dict = {}
        self.current_angle = []

    def _train_one_step(self, batch, det):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)
        r = np.random.rand(1)
        rand_index = torch.randperm(img.size(0)).to(self.device)
        #################################################################################
        #self.optimizer.zero_grad()
        #with torch.amp.autocast(device_type='cuda'):
        #    out, _ = self.model(img, det=True)
        #    loss = self.criterion(out, label)
        
        #start_time = time.time()
        #self.scaler.scale(loss).backward()
        
        #for i, l in enumerate(self.model.mixer_layers):
        #    self.layer_grad_norm_df_epoch[i].append(get_layer_grad(self.model, i))
        #################################################################################
        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            out, _ = self.model(img, det=False)
            loss = self.criterion(out, label)

            logits = out
            logit_difference = logits[:, 0] - logits[:, 1]
            distance_from_boundary = torch.abs(logit_difference).detach().cpu().numpy()
            predicted_class = predicted_class = torch.argmax(logits, dim=-1)   
            if (predicted_class == label): correct = True
            else: correct = False
            self.current_dict = {'correct': correct, 'distance': distance_from_boundary}
        
        #start_time = time.time()
        self.scaler.scale(loss).backward()
        
        for i, l in enumerate(self.model.mixer_layers):
            self.layer_grad_norm_bp_epoch[i].append(get_layer_grad(self.model, i))
        #####################################################################################
        #######
        for i in range(16):
            grad1 = self.layer_grad_norm_df_epoch[i][0]
            grad2 = self.layer_grad_norm_bp_epoch[i][0]
            self.layer_grad_angle_epoch[i].append( (torch.sum(grad1*grad2) / (torch.norm(grad1)*torch.norm(grad2))).item())
            self.current_angle.append( (torch.sum(grad1*grad2) / (torch.norm(grad1)*torch.norm(grad2))).item() )
        self.layer_grad_norm_df_epoch = [[] for _ in range(16)]
        self.layer_grad_norm_bp_epoch = [[] for _ in range(16)]
        #######
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
            self.train_loss = 0.0
            num_imgs = 0.
            det = False
            self.layer_grad_norm_df_epoch = [[] for _ in range(16)]
            self.layer_grad_norm_bp_epoch = [[] for _ in range(16)]
            self.layer_grad_angle_epoch = [[] for _ in range(16)]
            batch_dict = {}
            for idx, batch in enumerate(train_dl):
                self._train_one_step(batch, det)
                num_imgs += batch[0].size(0)

                ######
                ratio = self.current_angle
                batch_dict[idx] = {'correct': self.current_dict['correct'], 'distance': self.current_dict['distance'].tolist(), 'ratio': ratio}
                self.current_angle = []
                ######
            
            self.train_loss /= num_imgs
            print("Train loss: ", self.train_loss.item())
            for i in range(16):
                #self.layer_grad_norm_df[i].append(np.mean(self.layer_grad_norm_df_epoch[i]))
                #self.layer_grad_norm_bp[i].append(np.mean(self.layer_grad_norm_bp_epoch[i]))
                self.layer_grad_angle[i].append(np.mean(self.layer_grad_angle_epoch[i]))
            #print(self.layer_grad_norm_df_epoch[0])
            self.scheduler.step()

            if epoch == 3:
                with open('grads/lcn/samples_ep3.json', 'w') as file:
                    json.dump(batch_dict, file)
            if epoch == 5:
                with open('grads/lcn/samples_ep5.json', 'w') as file:
                    json.dump(batch_dict, file)
            if epoch == 9:
                with open('grads/lcn/samples_ep9.json', 'w') as file:
                    json.dump(batch_dict, file)
            
            # num_imgs = 0.
            # self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            # self.epoch_acc_list = []
            # self.epoch_corr_list = [0. for _ in range(17)]
            # for batch in test_dl:
            #     self._test_one_step(batch)
            #     num_imgs += batch[0].size(0)
            # self.epoch_loss /= num_imgs
            # self.epoch_acc = self.epoch_corr / num_imgs
            # for i in self.epoch_corr_list:
            #     self.epoch_acc_list.append((i / num_imgs).item())

            # print("Val loss: ", self.epoch_loss.item())
            # print("Val acc: ", self.epoch_acc.item())
            # print("Intermediate Layers:")
            # print(self.epoch_acc_list)
            # print("")
            self.epoch_loss = torch.tensor(0)

            train_loss.append(self.train_loss.item())
            val_loss.append(self.epoch_loss.item())
        
        # with open('grads/res/df.json', 'w') as file:
        #     json.dump(self.layer_grad_norm_df, file)
        # with open('grads/res/bp.json', 'w') as file:
        #     json.dump(self.layer_grad_norm_bp, file)
        with open('grads/lcnn/angle.json', 'w') as file:
            json.dump(self.layer_grad_angle, file)


        return train_loss, val_loss

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)  

def get_layer_grad(model, idx):
    fc1_grad_means = model.mixer_layers[idx].mlp2.fc1.weight.grad#.norm().item()
    #fc2_grad_means = model.mixer_layers[idx].mlp2.fc2.weight.grad#.norm().item()
    return fc1_grad_means
    return (fc1_grad_means + fc2_grad_means) / 2    

# def get_layer_grad(model, idx):
#     fc1_grad_means = model.mixer_layers[idx][0].weight.grad.norm().item()
#     fc2_grad_means = model.mixer_layers[idx][2].weight.grad.norm().item()
#     return (fc1_grad_means + fc2_grad_means) / 2    

def get_layer_norm(model, idx):
    fc1_grad_means = model.mixer_layers[idx].mlp2.fc1.weight.norm().item()
    fc2_grad_means = model.mixer_layers[idx].mlp2.fc2.weight.norm().item()
    return (fc1_grad_means + fc2_grad_means) / 2   
