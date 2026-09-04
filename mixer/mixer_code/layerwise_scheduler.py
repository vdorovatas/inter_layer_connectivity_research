import math


def get_mixer_param_groups(model, base_lr=0.001):
    """
    Assigns a layer_idx to each param group so the scheduler
    knows when to ramp each one up.
      patch_emb -> layer_idx = -1  (starts ramping immediately)
      mixer_layers[i] -> layer_idx = i
      ln, clf -> layer_idx = N     (same as last block)
    """
    N = len(model.mixer_layers)
    param_groups = []
    param_groups.append({'params': list(model.patch_emb.parameters()), 'lr': 0.0, 'layer_idx': -1})
    for i, block in enumerate(model.mixer_layers):
        param_groups.append({'params': list(block.parameters()), 'lr': 0.0, 'layer_idx': i})
    param_groups.append({'params': list(model.ln.parameters()), 'lr': 0.0, 'layer_idx': N - 1})
    param_groups.append({'params': list(model.clf.parameters()), 'lr': 0.0, 'layer_idx': N - 1})
    return param_groups


class StaggeredRampScheduler:
    """
    Each layer ramps up linearly to base_lr at its peak epoch, then holds.
    Earlier layers (lower layer_idx) peak sooner, deeper layers peak later.

    Args:
        optimizer:    AdamW or any optimizer whose param_groups have 'layer_idx'
        epochs:       total training epochs
        num_layers:   number of mixer layers (e.g. 16)
        base_lr:      target lr all layers converge to after their peak
        eta_min:      starting lr (before ramp), default 0
        peak_start:   epoch at which layer 0 (earliest) reaches base_lr
        peak_end:     epoch at which layer N-1 (deepest) reaches base_lr

    Mild example:      peak_start=50,  peak_end=200
    Aggressive example: peak_start=20, peak_end=250
    """
    def __init__(self, optimizer, epochs, num_layers,
                 base_lr=0.001, eta_min=0.0,
                 peak_start=50, peak_end=200):
        self.optimizer = optimizer
        self.epochs = epochs
        self.num_layers = num_layers
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.current_epoch = 0

        # initialise all lrs to eta_min
        for group in self.optimizer.param_groups:
            group['lr'] = self.eta_min

    def get_peak_epoch(self, layer_idx):
        # clamp layer_idx to [0, N-1] so patch_emb (-1) behaves like layer 0
        idx = max(layer_idx, 0)
        t = idx / max(self.num_layers - 1, 1)
        return int(self.peak_start + t * (self.peak_end - self.peak_start))

    def get_lr(self, layer_idx):
        peak = self.get_peak_epoch(layer_idx)
        e = self.current_epoch
        if peak == 0 or e >= peak:
            return self.base_lr
        # linear ramp from eta_min to base_lr
        return self.eta_min + (self.base_lr - self.eta_min) * (e / peak)

    def step(self):
        self.current_epoch += 1
        for group in self.optimizer.param_groups:
            group['lr'] = self.get_lr(group['layer_idx'])

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'peak_start': self.peak_start,
            'peak_end': self.peak_end,
            'base_lr': self.base_lr,
            'eta_min': self.eta_min,
        }

    def load_state_dict(self, d):
        self.current_epoch = d['current_epoch']
        self.peak_start = d['peak_start']
        self.peak_end = d['peak_end']
        self.base_lr = d['base_lr']
        self.eta_min = d['eta_min']
