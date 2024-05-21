import numpy as np
import torch


class MMnistEvaluator:
    @staticmethod
    def _compute_hist(model, data_loader, device):
        hist_left = np.zeros((10, 10))
        hist_right = np.zeros((10, 10))

        for input, t1, t2 in data_loader:
            hrepr = model["encoder"](input.to(device))
            p1 = model["left"](hrepr).argmax(dim=1).cpu().numpy()
            p2 = model["right"](hrepr).argmax(dim=1).cpu().numpy()

            t1, t2 = t1.numpy(), t2.numpy()

            hist_left += np.bincount(10 * p1 + t1, minlength=100).reshape(10, 10)
            hist_right += np.bincount(10 * p2 + t2, minlength=100).reshape(10, 10)
        return hist_left, hist_right

    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            hist_left, hist_right = MMnistEvaluator._compute_hist(
                model, data_loader, device
            )
            acc_per_class_left = np.diag(hist_left) / hist_left.sum(1)
            acc_per_class_right = np.diag(hist_right) / hist_right.sum(1)
            acc_left = np.diag(hist_left).sum() / hist_left.sum()
            acc_right = np.diag(hist_right).sum() / hist_right.sum()

        print("Acc: left = {:.4f}, right ={:.4f}".format(acc_left, acc_right))
        return acc_left, acc_right
