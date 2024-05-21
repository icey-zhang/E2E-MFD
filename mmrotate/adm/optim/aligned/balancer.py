import torch

from .solver import ProcrustesSolver
from .. import basic_balancer
from .. import balancers
import matplotlib.pylab as plt

@balancers.register("amtl")
class AlignedMTLBalancer(basic_balancer.BasicBalancer):
    def __init__(self, scale_mode='min', scale_decoder_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode
        print('AMGDA balancer scale mode:', self.scale_mode)

    def step(self, losses,
             shared_params,
             task_specific_params,
             shared_representation=None,
             last_shared_layer_params=None,
             iter=None,
    ):
        grads = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=False)
        # plt.figure()
        # plt.plot(grads[0].cpu().detach().numpy())
        # plt.plot(grads[1].cpu().detach().numpy())
        # plt.savefig("./grad/grad_before_{}.png".format(iter))
        grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), self.scale_mode)
        # plt.figure()
        # plt.plot(grads[0][:,0].cpu().detach().numpy())
        # plt.plot(grads[0][:,1].cpu().detach().numpy())
        # plt.savefig("./grad/grad_after_{}.png".format(iter))
        grad, weights = grads[0].sum(-1), weights.sum(-1)
        # plt.figure()
        # plt.plot(grad.cpu().detach().numpy())
        # #65536个数值
        # plt.savefig("./grad/grad_sum_{}.png".format(iter))
        
        if self.compute_stats: #false不走
            self.compute_metrics(grads[0])

        self.set_shared_grad(shared_params, grad)
        if self.scale_decoder_grad is True:
            self.apply_decoder_scaling(task_specific_params, weights)

        self.set_losses({task_id: losses[task_id] * weights[i] for i, task_id in enumerate(losses)})


@balancers.register("amtlub")
class AlignedMTLUBBalancer(basic_balancer.BasicBalancer):
    def __init__(self, scale_decoder_grad=False, scale_mode='min', **kwargs):
        super().__init__(**kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode

    def step_with_model(self, data, targets, model, criteria, **kwargs):
        self.zero_grad_model(model)
        hrepr = model.encoder(data)

        grads, losses = self.get_model_G_wrt_hrepr(hrepr, targets, model, criteria,
                                                   update_decoder_grads=True, return_losses=True)
        grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), self.scale_mode)
        grad, weights = grads[0].sum(-1), weights.sum(-1)

        if self.compute_stats:
            # Computationally expensive, use it for demonstration only.
            # This code is not supposed to be executed within real training pipelines.
            wgrads = list()
            for t in range(grads.shape[-1]):
                hrepr.backward(grads[:, :, t].view_as(hrepr), retain_graph=True)

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

        grad, weights = grads.sum(-1).view_as(hrepr), weights.sum(-1)
        hrepr.backward(grad)

        self.set_losses(losses)
        if self.scale_decoder_grad is True:
            self.apply_decoder_scaling(model.decoders, weights)
