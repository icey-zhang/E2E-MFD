import torch

from .. import basic_balancer
from .. import balancers


@balancers.register("gradnorm")
class GradNormBalancer(basic_balancer.BasicBalancer):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    Arxiv: https://arxiv.org/pdf/1711.02257.pdf
    """
    def __init__(self, alpha=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.initial_losses = None

    def add_model_parameters(self, model):
        for task_id in model.decoders:
            model.decoders[task_id].weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        losses, hrepr = self.compute_losses(data, targets, model, criteria)

        G = self.get_G_wrt_shared(losses, list(model.last_shared_layer.parameters()), update_decoder_grads=True)

        weights = torch.stack([d.weight for d in model.decoders.values()])
        weights = weights * len(criteria) / torch.sum(weights)
        grads = G.T * weights.view(1, -1)

        if self.initial_losses is None:
            self.initial_losses = {task_id: losses[task_id].clone().detach() for task_id in losses}

        # inverse training rates
        itrates = torch.stack([losses[task_id].clone().detach() / self.initial_losses[task_id] for task_id in losses])
        # mean inverse training rate
        mean_itrate = torch.mean(itrates)
        # relative inverse training rates
        itrates = itrates / mean_itrate
        # apply restoring force
        itrates = itrates.pow(self.alpha)

        norms = torch.norm(grads, dim=0, p=2)
        mean_norm = torch.mean(norms).clone().detach()

        grad_loss = torch.sum(torch.abs(norms - itrates * mean_norm))
        grad_loss.backward()

        self.set_losses({task_id: losses[task_id] * model.decoders[task_id].weight for task_id in losses})

        loss = 0.0
        for id, decoder in model.decoders.items():
            weight = decoder.weight.clone().detach()
            loss = loss + weight * criteria[id](decoder(hrepr), targets[id])

        loss.backward()
