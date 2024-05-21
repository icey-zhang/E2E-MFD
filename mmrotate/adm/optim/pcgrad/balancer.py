from .solver import RandomProjectionSolver
from .. import basic_balancer
from .. import balancers


@balancers.register("pcgrad")
class PCGradBalancer(basic_balancer.BasicBalancer):
    """
    Gradient Surgery for Multi-Task Learning
    Arxiv: https://arxiv.org/pdf/2001.06782.pdf
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:

        G = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        proj_grads = RandomProjectionSolver.apply(G)
        self.set_shared_grad(shared_params, proj_grads.sum(0))
        self.set_losses(losses)

        if self.compute_stats:
            self.compute_metrics(proj_grads)
