import torch
from dataloader import get_dataloaders
from dataloader import get_filtered_dataloaders
from train import Trainer
from long_mixer import MLPMixerL
import sys 
if __name__=='__main__':

        NUM_CLASSES=10
        #train_dl, test_dl = get_filtered_dataloaders(NUM_CLASSES)
        train_dl, test_dl = get_dataloaders()
        sys.exit()
        model = MLPMixerL(
            in_channels=3,
            img_size=32,
            hidden_size=256, #128
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
        ####
        #checkpoint = torch.load('checkpoints/lcn_500.pth.tar')
        #model.load_state_dict(checkpoint['state_dict'], strict=False)
        ####
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device: ', device)
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')
        trainer = Trainer(model, device)
        train_loss, test_loss = trainer.fit(train_dl, test_dl)
