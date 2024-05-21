import numpy as np
import torch
from scipy.optimize import minimize
from .. import basic_balancer
from .. import balancers


@balancers.register("cagrad")
class CAGradBalancer(basic_balancer.BasicBalancer):
    """
    Conflict-Averse Gradient Descent for Multitask Learning (CAGrad)
    Arxiv: https://arxiv.org/abs/2110.14048

    Modification of:
        https://github.com/Cranial-XIX/CAGrad/blob/e1de075dc6bbb038d564a99cf8ffa3f6f0edfef8/cityscapes/utils.py#L358
    """
    def __init__(self, calpha=0.5, rescale=0, scale_decoder_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.calpha = calpha
        self.rescale = rescale
        self.scale_decoder_grad = scale_decoder_grad

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        grads = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        GG = torch.matmul(grads, grads.T).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(len(losses), dtype=np.float32) / len(losses)
        bnds = tuple((0, 1) for _ in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (self.calpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
                + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)

        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw
        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1 + self.calpha ** 2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError("No support rescale type {}".format(self.rescale))

        if self.compute_stats:
            self.compute_metrics(grads * (ww.view(-1, 1) * lmbda * ww.numel() + 1))

        self.set_shared_grad(shared_params, new_grads)
        self.set_losses(losses)
