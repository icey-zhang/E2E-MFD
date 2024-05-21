import numpy as np
import torch
from tqdm import tqdm


# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (
        torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item(), (
        torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item()


def normal_error(x_pred, x_output):
    binary_mask = torch.sum(x_output, dim=1) != 0
    error = (
        torch.acos(
            torch.clamp(
                torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    error = np.degrees(error)
    return (
        np.mean(error).item(),
        np.median(error).item(),
        np.mean(error < 11.25).item(),
        np.mean(error < 22.5).item(),
        np.mean(error < 30).item(),
    )


class NYUv2Evaluator:
    @staticmethod
    def _compute_stats(model, data_loader, device):
        conf_mat = ConfMatrix()
        arr_pred_depth, arr_gt_depth = [], []
        arr_pred_normal, arr_gt_normal = [], []
        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            gt_semantic, gt_depth, gt_normal = (
                data[1],
                data[2],
                data[3],
            )

            hrepr = model.encoder(input.to(device))

            # depths
            p_depth = model.decoders["DE"](hrepr).to(gt_depth.device)
            arr_pred_depth.append(p_depth)
            arr_gt_depth.append(gt_depth)

            # normals
            p_normal = model.decoders["NE"](hrepr).cpu().to(gt_depth.device)
            arr_pred_normal.append(p_normal)
            arr_gt_normal.append(gt_normal)

            # semantic segmentation
            conf_mat.update(
                model.decoders["SS"](hrepr).argmax(1).flatten().cpu(),
                gt_semantic.flatten(),
            )
            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar
        m_segm_miou, m_segm_pix_acc = conf_mat.get_metrics()

        arr_pred_normal, arr_gt_normal = torch.cat(arr_pred_normal), torch.cat(
            arr_gt_normal
        )
        arr_pred_depth, arr_gt_depth = torch.cat(arr_pred_depth), torch.cat(
            arr_gt_depth
        )

        normal_errs = normal_error(arr_pred_normal, arr_gt_normal)
        depth_errs = depth_error(arr_pred_depth, arr_gt_depth)

        return m_segm_miou, m_segm_pix_acc, normal_errs, depth_errs

    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                m_segm_miou,
                m_segm_pix_acc,
                normal_errs,
                depth_errs,
            ) = NYUv2Evaluator._compute_stats(model, data_loader, device)
        print(
            f"mIOU: {m_segm_miou}, Pix Acc: {m_segm_pix_acc}; "
            f"Abs.: {depth_errs[0]}, Rel.: {depth_errs[1]};"
            f" Angle Dist (mean): {normal_errs[0]}, Angle Dist (median): {normal_errs[1]},"
            f" Within (11.25): {normal_errs[2]}, Within (22.5): {normal_errs[3]}, Within (30): {normal_errs[4]}"
        )
        return {'ss_miou': m_segm_miou, 'ss_pixacc': m_segm_pix_acc,
                'de_abs': depth_errs[0], 'de_rel': depth_errs[1],
                'ne_mean': normal_errs[0], 'ne_median': normal_errs[1],
                'ne_11.25': normal_errs[2], 'ne_22.5': normal_errs[3], 'ne_30': normal_errs[4]}
