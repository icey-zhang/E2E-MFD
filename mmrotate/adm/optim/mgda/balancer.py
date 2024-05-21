import torch

from .. import balancers
from .. import basic_balancer
from .solver import MinNormSolver


@balancers.register("mgda")
class MGDABalancer(basic_balancer.BasicBalancer):
    """
        Multi-Task Learning as Multi-Objective Optimization
        Arxiv: https://arxiv.org/abs/1810.04650

        Modification of:
            https://github.com/isl-org/MultiObjectiveOptimization
    """
    def __init__(self, scale_decoder_grad=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_decoder_grad = scale_decoder_grad

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        grads = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        scales, _ = MinNormSolver.apply(grads)
        grads = grads * scales.view(-1, 1)

        if self.compute_stats:
            self.compute_metrics(grads)

        grad = torch.sum(grads, dim=0)
        self.set_shared_grad(shared_params, grad)
        self.set_losses(losses)


@balancers.register("mgdaub")
class MGDAUBBalancer(basic_balancer.BasicBalancer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        self.zero_grad_model(model)
        hrepr = model.encoder(data)

        grads, losses = self.get_model_G_wrt_hrepr(hrepr, targets, model, criteria,
                                                   update_decoder_grads=True, return_losses=True)
        shape = grads.shape
        grads = grads.reshape(shape[0], -1)

        scales, _ = MinNormSolver.apply(grads)
        grads = grads * scales.view(-1, 1)
        self.set_losses({task_id: losses[task_id] * scales[i] for i, task_id in enumerate(losses)})

        if self.compute_stats:
            wgrads = list()
            for t in range(grads.shape[0]):
                hrepr.backward(grads[t].view_as(hrepr), retain_graph=True)

                wgrads.append(
                    torch.cat(
                        [
                            p.grad.flatten().detach().data.clone()
                            for p in model.encoder.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                model.encoder.zero_grad()
            wgrads = torch.stack(wgrads, dim=-1)
            self.compute_metrics(wgrads)

        grad = torch.sum(grads, dim=0)
        hrepr.backward(grad.view_as(hrepr))
