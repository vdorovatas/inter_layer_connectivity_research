import torch
from dataloader import get_dataloaders
from dataloader import get_filtered_dataloaders
from analyze_train import Trainer
import sys 
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='res', choices=['res', 'acn', 'hybrid', 'clamped_hybrid'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=300, help='Num Epochs')
    args = parser.parse_args()
    NUM_CLASSES=10
    #train_dl, test_dl = get_filtered_dataloaders(NUM_CLASSES)
    train_dl, test_dl = get_dataloaders(32)
    if True:
        print('Res-Mixer')
        from res_mixer import MLPMixer
        model_res = MLPMixer(
            in_channels=3,
            img_size=32,
            hidden_size=128, #128
            patch_size=4,
            hidden_c=512,
            hidden_s=64,
            num_layers=16, 
            num_classes=NUM_CLASSES,
            drop_p=0.,
            off_act=False,
            is_cls_token=True,
            vanilla=True
            )
    if True:
        print('AC-Mixer')
        from ac_mixer import MLPMixer
        model_acn = MLPMixer(
            in_channels=3,
            img_size=32,
            hidden_size=128, #128
            patch_size=4,
            hidden_c=512,
            hidden_s=64,
            num_layers=16,
            num_classes=NUM_CLASSES,
            drop_p=0.,
            off_act=False,
            is_cls_token=True,
            vanilla=False
            )
    checkpoint = torch.load('inherent_EE_checkpoints/acn_480.pth.tar', weights_only=False)
    model_acn.load_state_dict(checkpoint['state_dict'], strict=False) 
    checkpoint = torch.load('inherent_EE_checkpoints/res_400.pth.tar', weights_only=False)
    model_res.load_state_dict(checkpoint['state_dict'], strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_res.to(device)
    model_acn.to(device)
    trainer = Trainer(model_res, model_acn, device, args)
    train_loss, test_loss = trainer.eval_analyze(test_dl)
