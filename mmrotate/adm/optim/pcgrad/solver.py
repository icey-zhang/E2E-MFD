import torch


class RandomProjectionSolver:
    @staticmethod
    def apply(grads):
        assert (
            len(grads.shape) == 2
        ), f"Invalid shape of 'grads': {grads.shape}. Only 2D tensors are applicable"

        with torch.no_grad():
            order = torch.randperm(grads.shape[0])
            grads = grads[order]
            grads_task = grads

            def proj_grad(grad_task):
                for k in range(grads_task.shape[0]):
                    inner_product = torch.sum(grad_task * grads_task[k])
                    proj_direction = inner_product / (
                        torch.sum(grads_task[k] * grads_task[k]) + 1e-5
                    )
                    grad_task = (
                        grad_task
                        - torch.minimum(
                            proj_direction, torch.tensor(0.0).type_as(proj_direction)
                        )
                        * grads_task[k]
                    )
                return grad_task

            proj_grads = torch.stack(list(map(proj_grad, grads)), dim=0)

        return proj_grads
