import torch
import torch.nn as nn

class InnerCosFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, criterion, strength, target, mask):
        ctx.bs, ctx.c, _, _ = input.size()
        ctx.target = target
        ctx.strength = strength
        if torch.cuda.is_available:
            ctx.mask = mask.cuda()
        ctx.former = input.narrow(1, 0, ctx.c//2)
        ctx.former_in_mask = torch.mul(ctx.former, ctx.mask)
        ctx.criterion = criterion

        return input


    @staticmethod
    def backward(ctx, grad_output):
        ctx.fmask = ctx.former_in_mask.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            ctx.loss = ctx.criterion(ctx.fmask * ctx.strength, ctx.target)
            ctx.loss.backward()

        grad_output[:,0:ctx.c//2, :,:] += ctx.fmask.grad
        return grad_output, None, None, None, None
