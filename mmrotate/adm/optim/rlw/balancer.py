import torch
import torch.nn.functional as F

from .. import basic_balancer
from .. import balancers


@balancers.register("rlw")
class RandomLossWeighting(basic_balancer.BasicBalancer):
    """
    Random loss weighting with normal distribution
    "Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning"
    Arxiv: https://arxiv.org/abs/2111.10603
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        losses, hrepr = self.compute_losses(data, targets, model, criteria)
        n_tasks = len(criteria)
        weight = (F.softmax(torch.randn(n_tasks), dim=-1)).to(data.device)

        losses = {task_id: losses[task_id] * weight[i] for i, task_id in enumerate(criteria)}
        total_loss = sum(losses.values())

        if self.compute_stats:
            G = self.get_G_wrt_shared(losses, list(model.encoder.parameters()))
            self.compute_metrics(G)

        total_loss.backward()
        self.set_losses(losses)
