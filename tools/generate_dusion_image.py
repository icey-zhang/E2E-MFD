import math
import os

# from prettytable import PrettyTable
import tqdm
import cv2
import scipy
import numpy as np
import torch
import copy
# from Metrics.Metric import Evaluator
from mmrotate.models.detectors.oriented_rcnn_m import Oriented_rcnn_m
from mmdet.datasets import (build_dataloader, build_dataset)
from mmrotate.utils import collect_env, get_root_logger, setup_multi_processes
from mmcv import Config, DictAction


import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat, savemat
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if __name__ == '__main__':
    device = 'cuda'
    
    cfg = Config.fromfile('/home/zjq/LSKcmx/tools/cfg/lsk_s_fpn_1x_dota_le90.py')
    setup_multi_processes(cfg)

    model_my = Oriented_rcnn_m(cfg.model.backbone, cfg.model.neck, cfg.model.rpn_head, cfg.model.roi_head, cfg.model.train_cfg, cfg.model.test_cfg).to(device)
    model_data = torch.load('/home/zjq/LSKcmx/output/s_maxhalf_ir_bs4/epoch_12.pth')
    # key = model.load_state_dict(model_data['model'], strict=True)
    key = model_my.load_state_dict(model_data['state_dict'], strict=False)
    model_my.eval()
    iter_name = 's_maxhalf_ir_DV_RGB/'
    iter_name_V = 's_maxhalf_ir_DV_V/'
    path_out = '/home/data3/zjq/DroneVehicle_mm/generate_DV/test/'

    os.makedirs(path_out + iter_name, exist_ok=True)
    os.makedirs(path_out + iter_name_V, exist_ok=True)
    val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=False,
            shuffle=False,
            persistent_workers=False)

    val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
    

    time_sum = 0
    # for name in tqdm.tqdm(names):
    for i, data_batch in enumerate(val_dataloader):
        name = val_dataset.data_infos[i]['filename'].split('.')[0]
        with torch.no_grad():
            # image = data_batch['img'].to(device)
            V_img, output, time_per_img, vi = model_my.forward_fusion(data_batch['img'][0])
            time_sum += time_per_img
            ############ 在这里保存V空间的图片 ###########
            bri = V_img.detach().cpu().numpy() * 255
            bri = bri.reshape([V_img.size()[2], V_img.size()[3]])
            bri = bri[:712, :840]
            ###############################################
            # bri = np.where(bri < 0, 0, bri)
            # bri = np.where(bri > 255, 255, bri)              #########  是否应该按比例放缩，而不是硬放缩到0-1  如下代码所示
            ###############################################
            min_value = bri.min()
            max_value = bri.max() 
            scale = 255 / (max_value - min_value) 
            bri = (bri - min_value) * scale
            bri = np.clip(bri, 0, 255)
            ###############################################
            im1 = Image.fromarray(bri.astype(np.uint8))

            im1.save(path_out +iter_name_V + name + '.png')
            ###############################################

        ############ 在这里保存RGB2HSV空间的图片 ###########
        vi = vi.cpu().numpy().transpose(1, 2, 0)
        if output.shape[:2] != vi.shape[:2]:
            output = cv2.resize(output, vi.shape[:2][::-1])
        output = output[..., ::-1]
        output = output[:712, :840, :]
        # vi = vi[..., ::-1]
        # cv2.imwrite(path_out + '/vi/' + name + '.png', vi)
        # cv2.imwrite(path_out + '/ir/' + name + '.png', ir)
        cv2.imwrite(path_out + iter_name + name + '.png', output)
        ###############################################
        print(i)
    print('generate images done')
    print(time_sum)