import torch
from .. import basic_balancer
from .. import balancers


@balancers.register("uncertainty")
class HomoscedasticUncertaintyBalancer(basic_balancer.BasicBalancer):
    """
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    Arxiv: https://arxiv.org/abs/1705.07115
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_model_parameters(self, model):
        for task_id in model.decoders:
            model.decoders[task_id].log_var = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        def loss_fn(task_id):
            precision = (-model.decoders[task_id].log_var).exp()
            task_loss = criteria[task_id](model.decoders[task_id](hrepr), targets[task_id])
            return precision * task_loss + 0.5 * model.decoders[task_id].log_var

        self.zero_grad_model(model)
        hrepr = model.encoder(data)
        losses = {task_id: loss_fn(task_id) for task_id in criteria}

        if self.compute_stats:
            G = self.get_G_wrt_shared(losses, list(model.encoder.parameters()))
            self.compute_metrics(G)

        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)
