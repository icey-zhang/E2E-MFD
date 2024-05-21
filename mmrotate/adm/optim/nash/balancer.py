import torch
import numpy as np
import cvxpy as cp
from .. import balancers
from .. import basic_balancer


@balancers.register("nash")
class NashMTL(basic_balancer.BasicBalancer):
    """
        Multi-Task Learning as a Bargaining Game
        Arxiv: https://arxiv.org/abs/2202.01017
        Modification of: https://github.com/AvivNavon/nash-mtl
    """
    def __init__(
        self,
        n_tasks: int = 3,
        device: torch.device = torch.device('cuda'),
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
        *args, **kwargs
    ):
        super(NashMTL, self).__init__(*args, **kwargs)
        self.n_tasks = n_tasks
        self.device = device

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)
        self._init_optim_problem()

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        G = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        GTG = torch.mm(G, G.t())
        self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
        GTG = GTG / self.normalization_factor.item()
        alpha = self.solve_optimization(GTG.cpu().detach().numpy())
        alpha = torch.from_numpy(alpha).view(-1, 1).cuda().float()

        shared_grad = (G * alpha).sum(dim=0)
        self.set_shared_grad(shared_params, shared_grad)

        if self.compute_stats:
            self.compute_metrics(G * alpha)

        self.set_losses({task_id: losses[task_id] * alpha[i] for i, task_id in enumerate(losses)})
