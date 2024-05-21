import torch

from .. import basic_balancer
from .. import balancers


@balancers.register("graddrop")
class GradDropBalancer(basic_balancer.BasicBalancer):
    """
    Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout
    Arxiv: https://arxiv.org/abs/2010.06808

    """
    def __init__(self, leak=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leak = leak

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        self.zero_grad_model(model)
        hrepr = model.encoder(data)
        grads, losses = self.get_model_G_wrt_hrepr(hrepr, targets, model, criteria,
                                                   update_decoder_grads=True, return_losses=True)
        grads = grads.T
        G = torch.sign(hrepr.view(-1, 1)) * grads
        G = torch.sum(G, dim=0, keepdim=True)
        P = torch.sum(G, dim=-1, keepdim=True) / (torch.sum(G.abs(), dim=-1, keepdim=True) + 1e-8)
        P = 0.5 * (1 + P)
        U = torch.rand(P.shape, device=P.device)
        Mi = ((P > U).int() * (G > 0).int()) + ((P < U).int() * (G < 0).int())
        grads = grads * (Mi * self.leak + (1 - self.leak))

        grad = torch.sum(grads, dim=-1)
        hrepr.backward(grad.view_as(hrepr))
        self.set_losses(losses)

