import torch.nn as nn
import torch
import torch.nn.functional as F
import util.util as util
import types
class InnerCos(nn.Module):
    def __init__(self, crit='MSE', strength=1, skip=0, layer_to_last=3):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()
        self.strength = strength
        # To define whether this layer is skipped.
        self.skip = skip
        self.layer_to_last = layer_to_last
        # Init a dummy value is fine.
        self.target = torch.tensor(1.0)
        def identity(self):
            return self
        self.loss = torch.cuda.FloatTensor(1)
        self.loss.float = types.MethodType(identity, self.loss)
        self.register_buffer('cos_loss', self.loss)

    def set_mask(self, mask_global):
        mask = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask = mask.squeeze()
        if torch.cuda.is_available:
            self.mask = self.mask.float().cuda()

    def forward(self, in_data):
        self.bs, self.c, _, _ = in_data.size()
        if torch.cuda.is_available:
            self.mask = self.mask.cuda()
        if not self.skip:
            self.former = in_data.narrow(1, 0, self.c//2)
            self.former_in_mask = torch.mul(self.former, self.mask)
            if len(self.target.size()) == 0: # For the first iteration.
                self.target = self.target.expand_as(self.former_in_mask).type_as(self.former_in_mask)
            
            if self.former_in_mask.size() != self.target.size():  # For the last iteration of one epoch
                self.target = self.target.narrow(0, 0, 1).expand_as(self.former_in_mask).type_as(self.former_in_mask)
            
            # self.loss shuold put before self.target.
            # For each iteration, we input GT, then I. That means we get the self.target in the first forward. And in this forward, self.loss is dummy!
            # In the second forward, we input the corresponding I, then self.loss is working as expected. The self.target is the corresponding GT.
            self.loss = self.criterion(self.former_in_mask, self.target)
            self.target = in_data.narrow(1, self.c // 2, self.c // 2).detach() # the latter part
            self.target = self.target * self.strength
        else:
            self.loss = 0
        self.output = in_data
        return self.output

    def __repr__(self):
        skip_str = 'True' if not self.skip else 'False'
        return self.__class__.__name__+ '(' \
              + 'skip: ' + skip_str \
              + 'layer ' + str(self.layer_to_last) + ' to last' \
              + ' ,strength: ' + str(self.strength) + ')'
