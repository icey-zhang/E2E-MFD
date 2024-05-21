import torch
from .. import basic_balancer
from .. import balancers


@balancers.register("ls")
class LinearScalarization(basic_balancer.BasicBalancer):
    """
    Uniform task weighting
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None):

        if self.compute_stats:
            G = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=False)
            self.compute_metrics(G)

        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)
