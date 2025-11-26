import math

import torch


class MyAdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=0.5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        beta1, beta2 = betas

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)

        super(MyAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                beta1, beta2 = group['beta1'], group['beta2']
                eps = group['eps']

                if t == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['beta1_t'] = 1.
                    state['beta2_t'] = 1.

                beta1_t, beta2_t = state['beta1_t'] * beta1, state['beta2_t'] * beta2
                m, v = state['m'], state['v']

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                adjusted_lr = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                p.data = p.data - adjusted_lr * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * group['weight_decay'] * p.data

                state['t'] = t + 1
                state['m'], state['v'] = m, v
                state['beta1_t'], state['beta2_t'] = beta1_t, beta2_t

        return loss
