import torch
import numpy as np

from .. import basic_balancer
from .. import balancers


@balancers.register("dwa")
class DynamicWeightAveraging(basic_balancer.BasicBalancer):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Arxiv: https://arxiv.org/abs/1803.10704
    
    Modification of:
        https://github.com/lorenmt/mtan/blob/c36c30baa18968dec74fe9039abcfd4f132edfa1/im2im_pred/utils.py#L139
    """

    def __init__(self, iteration_window: int = 25, temp=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = None
        self.weights = None

    def reset(self, n_tasks):
        self.costs = np.ones((self.iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)
        self.running_iterations = 0

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        n_tasks = len(losses)
        if self.costs is None:
            self.reset(n_tasks)

        losses_vals = [losses[task_id].item() for task_id in losses]
        cur_cost = np.asarray(losses_vals)
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cur_cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[:self.iteration_window, :].mean(0)
            self.weights = (n_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        total_loss = sum([self.weights[i] * losses[task_id] for i, task_id in enumerate(losses)]) / n_tasks

        self.running_iterations += 1
        total_loss.backward()
        self.set_losses({task_id: losses[task_id] * self.weights[i] for i, task_id in enumerate(losses)})
