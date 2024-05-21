import random

import numpy as np
import torch
from .. import basic_balancer
from .. import balancers


@balancers.register("gradvac")
class GradVacBalancer(basic_balancer.BasicBalancer):
    """
    Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models
    Arxiv: https://arxiv.org/abs/2010.05874

    Modification of:
        https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/GradVac.py
    """
    def __init__(self, beta=0.5, scale_decoder_grad=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.beta = beta
        self.rho_T = None

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        if self.rho_T is None:
            task_num = len(losses)
            self.rho_T = torch.zeros(task_num, task_num)

        grads = self.get_G_wrt_shared(losses, shared_params)

        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(len(losses)):
            task_index = list(range(len(losses)))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm() * grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = (
                        pc_grads[tn_i].norm() * (self.rho_T[tn_i, tn_j] * (1 - rho_ij ** 2).sqrt() - rho_ij * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt())
                        / (grads[tn_j].norm() * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt()))
                    pc_grads[tn_i] += grads[tn_j] * w
                    batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j] = (1 - self.beta) * self.rho_T[
                        tn_i, tn_j
                    ] + self.beta * rho_ij

        grad = pc_grads.sum(0)
        self.set_shared_grad(shared_params, grad)
        self.set_losses(losses)
