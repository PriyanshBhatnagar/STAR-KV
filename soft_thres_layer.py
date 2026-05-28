import torch
import torch.nn as nn
import torch.nn.functional as F


class soft_thres_layer(nn.Module):
    def __init__(self, s, c, alpha=0.75):
        '''s: controls sharpness'''
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]), requires_grad=True)
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))
        self.register_buffer("c", torch.tensor(c, dtype=torch.float32))

    def forward(self, x):
        return soft_thres_func.apply(x, self.alpha, self.s, self.c)


class soft_thres_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, s, c):
        ctx.save_for_backward(x, alpha, s, c)
        tanh = torch.tanh(s * (x - alpha))
        coef = torch.where(x > alpha, x, -c)
        output = coef * tanh
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, s, c = ctx.saved_tensors
        tanh = torch.tanh(s * (x - alpha))
        grt_grad_x = tanh + s * x * (1 - torch.pow(tanh, 2))
        les_grad_x = -c * s * (1 - torch.pow(tanh, 2))
        grad_x = torch.where(x > alpha, grt_grad_x, les_grad_x)

        grt_grad_alpha = x * (-s) * (1 - torch.pow(tanh, 2))
        les_grad_alpha = c * s * (1 - torch.pow(tanh, 2))
        grad_alpha = torch.where(x > alpha, grt_grad_alpha, les_grad_alpha)

        grad_x_out = grad_x * grad_output
        grad_alpha_out = grad_alpha * grad_output
        grad_alpha_sum = torch.tensor([torch.sum(grad_alpha_out)]).to('cuda')

        return grad_x_out, grad_alpha_sum, None, None
