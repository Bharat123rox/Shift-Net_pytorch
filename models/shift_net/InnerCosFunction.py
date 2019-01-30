import torch
import torch.nn as nn

class InnerCosFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, criterion, strength, target, mask):
        ctx.c = input.size(1)
        ctx.strength = strength
        ctx.criterion = criterion

        ctx.save_for_backward(input, target, mask)
        return input


    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            input, target, mask = ctx.saved_tensors
            former = input.narrow(1, 0, ctx.c//2)
            former_in_mask = torch.mul(former, mask)
            former_in_mask_clone = former_in_mask.clone().detach().requires_grad_(True)
            ctx.loss = ctx.criterion(former_in_mask_clone, target) * ctx.strength
            ctx.loss.backward()

        grad_output[:,0:ctx.c//2, :,:] += former_in_mask_clone.grad  
            
        return grad_output, None, None, None, None
