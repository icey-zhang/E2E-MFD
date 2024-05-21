# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results

@ROTATED_PIPELINES.register_module()
class LoadImagePairFromFile(LoadImageFromFile):
    """Load dualspectral image pair from two files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "v/t_img_norm_cfg" (means=0 and stds=1).

    Args:
        spectrals (tuple/list): Names of folders denoting different spectrals.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self,
                 spectrals=('rgb', 'ir'),
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.spectrals = spectrals
        
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename1 = osp.join(results['img_prefix'], results['img_info']['filename'])
            filename2 = osp.join(results['img_prefix'].replace(self.spectrals[0], self.spectrals[1], 1), 
                                 results['img_info']['filename'])
        else:
            filename1 = results['img_info']['filename']
            filename2 = results['img_info']['filename'].replace(self.spectrals[0], self.spectrals[1], 1)

        img1_bytes = self.file_client.get(filename1)
        img2_bytes = self.file_client.get(filename2)
        img1 = mmcv.imfrombytes(img1_bytes, flag=self.color_type, channel_order=self.channel_order)
        img2 = mmcv.imfrombytes(img2_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename'] = (filename1, filename2)
        results['ori_filename'] = results['img_info']['filename']
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        
        results['img_fields'] = ['img1', 'img2']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'spectrals={self.spectrals},'
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
