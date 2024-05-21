import torch
from .. import basic_balancer
from .. import balancers


@balancers.register("si")
class ScaleInvariantLinearScalarization(basic_balancer.BasicBalancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        def loss_fn(id):
            return criteria[id](model.decoders[id](hrepr), targets[id]).log1p()

        self.zero_grad_model(model)
        hrepr = model.encoder(data)

        losses = {task_id: loss_fn(task_id) for task_id in criteria}
        total_loss = sum(losses.values())

        if self.compute_stats:
            G = self.get_G_wrt_shared(losses, list(model.encoder.parameters()))
            self.compute_metrics(G)

        total_loss.backward()
        self.set_losses(losses)
