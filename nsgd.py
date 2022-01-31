import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import numpy as np

required=True

class NSGD(Optimizer):

    def __init__(self, params, irho=required, col=-1):
        if irho is not required and irho < 0.0:
            raise ValueError("Invalid learning rate: {}".format(irho))

        defaults = dict(irho=irho, col=col)
        super(NSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NSGD, self).__setstate__(state)

    def nyscurve(self, gradloader, model, criterion, device):
        """Nystrom-Approximated Curvature Information"""
        for group in self.param_groups:
            col = group['col']
            p = torch.cat([pi.view(-1) for pi in group['params']])
            if col < 0:
                col = np.int32(np.ceil(np.log2(p.shape[0])))
            h = torch.zeros(col, p.shape[0]).to(device)
            idx = torch.randperm(p.shape[0])[:col]
            for batch_idx, (inputs, targets) in enumerate(gradloader):
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                #optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
                g = torch.cat([gi.view(-1) for gi in g])
                for j in range(col):
                    if j == col-1:
                        h[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[idx[j]], group['params'], retain_graph=False)])
                    else:
                        h[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[idx[j]], group['params'], retain_graph=True)])
            h = h/len(gradloader)
            M = h[:,idx]
            rnk = torch.matrix_rank(M)
            U, S, V = torch.svd(M)
            ix = range(0, rnk)
            U = U[:, ix]
            S = torch.sqrt(torch.diag(1./S[ix]))
            self.Z = torch.mm(h.t(), torch.mm(U, S))
            self.Q = group['irho']**2 * torch.mm(self.Z, torch.inverse(torch.eye(rnk).to(device) + group['irho'] * torch.mm(self.Z.t(), self.Z)))

    def step(self):
        """Compute the scaled gradient
        """
        for group in self.param_groups:
            g=torch.cat([p.grad.view(-1) for p in group['params']])
            v_new = group['irho']*g.view(-1,1)-torch.mm(self.Q, torch.mm(self.Z.t(), g.view(-1,1)))
            ls=0
            for p in group['params']:
                vp=v_new[ls:ls+torch.numel(p)].view(p.shape)
                ls += torch.numel(p)
                p.grad = vp

