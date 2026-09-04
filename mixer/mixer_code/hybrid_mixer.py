'''
code: https://github.com/omihub777/MLP-Mixer-CIFAR/blob/main/mlp_mixer.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import copy
import numpy as np

class DepthLayerNorm(nn.Module):
    """ LayerNorm with learnable depth-adaptive strength """

    def __init__(self, ndim, depth, bias=True, strength_factor=0.05):
        super().__init__()
        # Base LayerNorm parameters
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        # Learnable strength parameter initialized with depth bias
        init_strength = 1.0 + depth * strength_factor
        self.strength = nn.Parameter(torch.tensor(init_strength))

    def forward(self, input):
        # Standard layer normalization
        normalized = F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        # Apply learnable strength
        return self.strength * normalized

class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256,
                hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False, vanilla=True):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token
        self.vanilla = vanilla

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        self.mixer_layers = nn.ModuleList([MixerLayer(num_patches, hidden_size, hidden_s,
                            hidden_c, drop_p, off_act, vanilla=self.vanilla, layer_idx=layer_idx) for layer_idx in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_size)
        
        alpha_values = [np.random.normal(loc=0.25, scale=0.005) for _ in range(16)]
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(w), requires_grad=True) for w in alpha_values])

        self.clf = nn.Linear(hidden_size, num_classes)

    def forward(self, x, eval_= False, det=False):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)

        intermid = [out]
        current = out
        for layer,a in zip(self.mixer_layers, self.alphas): 
            mlp2, mlp1 = layer(out)
            current = current + mlp2 + a*mlp1
            out = mlp2 + a*(mlp1+out)
            intermid.append(current)
        out = current 
        intermid_preds = []
        if eval_: intermid_preds = [self.clf(self.ln(i[:, 0])) for i in intermid]

        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)

        return out, intermid_preds

class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, vanilla, layer_idx):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act, layer_idx)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act, layer_idx)

        self.vanilla = vanilla
        
        self.relu =  nn.ReLU()
        self.linear = nn.Linear(hidden_size, hidden_size)
        #self.ln = nn.LayerNorm(hidden_size)
        self.ln =  DepthLayerNorm(hidden_size, layer_idx)

    def forward(self, x):
        mlp1_out = self.mlp1(x)
        intermid = mlp1_out+x
        mlp2_out = self.mlp2(intermid)
        ########
        mlp2_out = self.ln(self.relu(self.linear(mlp2_out)+x))
        ########
        return mlp2_out, mlp1_out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act, layer_idx):
        super(MLP1, self).__init__()
        #self.ln = nn.LayerNorm(hidden_size)
        self.ln =  DepthLayerNorm(hidden_size, layer_idx)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out
    
class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act, layer_idx):
        super(MLP2, self).__init__()
        #self.ln = nn.LayerNorm(hidden_size)
        self.ln =  DepthLayerNorm(hidden_size, layer_idx)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        
        return out
    


if __name__ == '__main__':
    net = MLPMixer(
        in_channels=3,
        img_size=32,
        patch_size=4,
        hidden_size=128,
        hidden_s=512,
        hidden_c=64,
        num_layers=8,
        num_classes=10,
        drop_p=0.,
        off_act=False,
        is_cls_token=True
        )
    

