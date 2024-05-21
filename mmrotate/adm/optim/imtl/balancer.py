import torch
from .. import basic_balancer
from .. import balancers


@balancers.register("imtl")
class IMTLG(basic_balancer.BasicBalancer):
    """
    Towards Impartial Multi-task Learning
    Paper: https://openreview.net/forum?id=IMPnRXEWpvr

    Modification of:
    https://github.com/AvivNavon/nash-mtl/blob/7cc1694a276ca6f2f9426ab18b8698c786bff4f0/methods/weight_methods.py#L671
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        n_tasks = len(losses)
        G = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        U = G / torch.linalg.norm(G, dim=1, keepdim=True)

        D = G[0, ] - G[1:, ]
        U = U[0, ] - U[1:, ]

        first_element = torch.matmul(G[0, ], U.t())
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(n_tasks - 1, device=first_element.device) * 1e-8 + torch.matmul(D, U.t()))

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat((torch.tensor(1 - alpha_.sum(), device=first_element.device).unsqueeze(-1), alpha_))

        if self.compute_stats:
            self.compute_metrics(G * alpha.view(-1, 1))

        self.set_shared_grad(shared_params, (G * alpha.view(-1, 1)).sum(dim=0))
        self.set_losses(losses)
