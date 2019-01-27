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
        print('original former grad mean ', grad_output[:, 0:ctx.c//2, :, :].mean())
        with torch.enable_grad():
            input, target, mask = ctx.saved_tensors
            former = input.narrow(1, 0, ctx.c//2)
            former_in_mask = torch.mul(former, mask)
            print(former_in_mask.requires_grad)
            print(target.requires_grad)
            former_in_mask_clone = former_in_mask.clone().detach().requires_grad_(True)
            ctx.loss = ctx.criterion(former_in_mask_clone, target) * ctx.strength
            ctx.loss.backward()
            print('now former grad mean ',former_in_mask_clone.grad.mean())
            
        print('now former grad mean ', grad_output[:, 0:ctx.c//2, :, :].mean())
            
        print()
        assert 1==2

        grad_output[:,0:ctx.c//2, :,:] += former_in_mask.grad
        return grad_output, None, None, None, None
