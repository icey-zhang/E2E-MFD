from adm.data.datasets.posenet import cal_quat_angle_error

import numpy as np
import torch
from tqdm import tqdm


class SevenScenesEvaluator:
    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            q_err_all, t_err_all = [], []
            encoder = model["encoder"]
            t_decoder, q_decoder = model["left"], model["right"]
            for data in tqdm(data_loader, total=len(data_loader)):
                t_gt, q_gt = data["t_gt"].to(device), data["q_gt"].to(device)
                img = data["img"].to(device)

                enc_out = encoder(img)
                t_est, q_est = t_decoder(enc_out), q_decoder(enc_out)

                # compute errors for translation and orientation
                t_err = torch.norm(t_gt - t_est, p=2, dim=1)
                q_err = torch.FloatTensor(
                    cal_quat_angle_error(q_gt.cpu().numpy(), q_est.cpu().numpy())
                ).to(device)

                q_err_all.append(q_err)
                t_err_all.append(t_err)

            q_err_all = torch.cat(q_err_all).cpu().numpy()
            t_err_all = torch.cat(t_err_all).cpu().numpy()

        t_err_med, q_err_med = np.median(t_err_all), np.median(q_err_all)
        print(f"Acc (median error): pos: {t_err_med}, ornt: {q_err_med}")
        return t_err_med, q_err_med
