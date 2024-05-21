import numpy as np
import torch


class CityScapesEvaluator:
    @staticmethod
    def _compute_stats(model, data_loader, device):
        ss_hist = np.zeros((19, 19))
        is_enum = 0.0
        is_denum = 0.0

        de_enum = 0.0
        de_denum = 0.0

        for data in data_loader:
            input = data[0]
            tss, tis, tde = data[1].numpy(), data[2].numpy(), data[3].numpy()

            hrepr = model.encoder(input.to(device))
            pis = model.decoders["IS"](hrepr).cpu().numpy()
            pss = model.decoders["SS"](hrepr).argmax(dim=1).cpu().numpy()
            pde = model.decoders["DE"](hrepr).cpu().numpy()

            # update semantic metric
            mask = (tss >= 0) & (tss < 19)
            ss_hist += np.bincount(
                19 * tss[mask] + pss[mask], minlength=19 ** 2
            ).reshape(19, 19)

            # update instance metric
            _tis = tis.astype(np.int32)
            _pis = pis.astype(np.int32)
            mask = _tis != 250
            if np.sum(mask) >= 1:
                # L1 pixel loss
                is_enum += np.sum(np.abs(_pis[mask] - _tis[mask]))
                is_denum += np.sum(mask)

            # update depth metric
            # _tde = tde.astype(np.int32)
            mask = tde != 0.0
            if np.sum(mask) >= 1:
                # Up to shift and scale MSE
                de_enum += np.sum((pde[mask] - tde[mask]) ** 2)
                de_denum += np.sum(mask)

        return ss_hist, is_enum, is_denum, de_enum, de_denum

    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            (
                ss_hist,
                is_enum,
                is_denum,
                de_enum,
                de_denum,
            ) = CityScapesEvaluator._compute_stats(model, data_loader, device)
            iou = np.diag(ss_hist) / (
                ss_hist.sum(1) + ss_hist.sum(0) - np.diag(ss_hist)
            )
            miou = np.nanmean(iou)

            is_mae = is_enum / is_denum
            de_mse = de_enum / de_denum

        print(
            "mIoU: {:.4f}, Instance L1[pxl]: {:.4f}, Depth UTSS MSE: {:.4f}".format(
                miou, is_mae, de_mse
            )
        )

        return {'ss_miou': miou, 'is_mae': is_mae, 'de_mse': de_mse}
