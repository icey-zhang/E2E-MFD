# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math
import time
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .fusion import FusionNet
from .loss import DetcropPixelLoss
import matplotlib.pyplot as plt


class Add_fusion(nn.Module):
    def __init__(self, in_chs):
        super().__init__()
        self.n_levels = len(in_chs)

    def forward(self, x_v, x_t):
        x = []
        for i in range(self.n_levels):
            x.append(x_t[i] + x_v[i])
        return x

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, h, w = x.shape
        not_mask = torch.ones(bs, h, w)
        y_embed = not_mask.cumsum(1, dtype=torch.float32).to(x.device)
        x_embed = not_mask.cumsum(2, dtype=torch.float32).to(x.device)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0

def ms_deform_attn_core_pytorch_key_aware(query, 
                                          value, 
                                          key, 
                                          value_spatial_shapes, 
                                          sampling_locations,
                                          query_proj):
    # for debug and test only,
    # need to use cuda version instead
    # N: batch szie; S_: total value num;   M_: head num 8; mD: 256/M (32)
    # Lq_: len q;  L_: num levels (4); P_: sample point per-level (4)
    # N bs, S 所有token的数量, M head的数量, D 256/8
    N_, S_, M_, D_ = value.shape
    # Lq query的数量
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    # 按照各个特征层分开
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    key_list = key.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    sampling_key_list = []

    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # 多了一个对key的处理
        key_l_ = key_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1).to(value_l_.dtype)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', 
                                          padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

        # 与上面value的处理类似
        # N_*M_, D_, Lq_, P_
        sampling_key_l__ = F.grid_sample(key_l_, sampling_grid_l_, mode='bilinear', 
                                         padding_mode='zeros', align_corners=False)
        sampling_key_list.append(sampling_key_l__)

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    key = torch.stack(sampling_key_list, dim=-2).flatten(-2)
    value = torch.stack(sampling_value_list, dim=-2).flatten(-2)

    # N_*M_, D_, Lq_, L_*P_ -> N*M, Lq, L*P, D -> N*M*Lq, L*P, D
    key = key.permute(0, 2, 3, 1).flatten(0, 1)

    N_, Lq, DD_ = query.shape
    query = query_proj(query)
    query = query.view(N_, Lq, M_, DD_ // M_)
    query = query.permute(0, 2, 1, 3).flatten(0, 2)  # N, Lq, M, D -> N, M, Lq, D -> N*M*Lq, D
    query = query.unsqueeze(-2)  # N*M*Lq, D-> N*M*Lq, 1, D
    dk = query.size()[-1]

    # QK的运算，self attention的内容
    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    attention_weights = F.softmax(attention_weights, -1)
    value = value.permute(0, 2, 3, 1).flatten(0, 1)  # N*M*Lq, L*P, D
    # QKV得到最后的结果
    output = attention_weights.matmul(value)  # N*M, Lq, 1,  L*P x N*M*Lq, L*P, D -> N*M, Lq, 1,  D
    output = output.squeeze(-2).view(N_, M_, Lq_, D_).permute(0, 2, 1, 3)  # N*M, Lq, 1,  D -> N, Lq, M,  D
    output = output.flatten(2)
    return output.contiguous()

def multi_scale_deformable_attn_pytorch(value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()

class MSDeformAttnKDA(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4,\
                 same_loc=False, key_aware=True, proj_key=True):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.same_loc = same_loc
        self.key_aware = key_aware
        self.proj_key = proj_key

        if not same_loc:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        if not key_aware:
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        if proj_key:
            self.key_proj = nn.Linear(d_model, d_model)
        else:
            self.key_proj = None
        self.query_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.same_loc:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(
                1, self.n_points, 1)
            for i in range(self.n_points):
                grid_init[:, i, :] *= i + 1
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
                1, self.n_levels, self.n_points, 1)
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if not self.key_aware:
            constant_(self.attention_weights.weight.data, 0.)
            constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        if self.key_proj:
            xavier_uniform_(self.key_proj.weight.data)
            constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, input_flatten, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = input_flatten.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(input_flatten)
        key = input_flatten
        if self.proj_key:
            key = self.key_proj(key)
        else:
            key = value

        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
            key = key.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        key = key.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)

        if not self.same_loc:
            sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_points, 2)
            sampling_offsets = sampling_offsets[:, :, :, None].repeat(1, 1, 1, self.n_levels, 1, 1)
        
        attention_weights = None
        if not self.key_aware:
            attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        
        if self.key_aware:
            output = ms_deform_attn_core_pytorch_key_aware(query, value, key, value_shapes, sampling_locations, self.query_proj)
        else:
            output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)

class DeformableCrossattKDAInter(nn.Module):
    def __init__(self, c1, d_model=256, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.n_level = 1
        self.d_model = d_model
        self.posembed = PositionEmbeddingSine(d_model//2)
        # self.posembed = PositionEmbeddingLearned(d_model//2)
        # self.level_embed = nn.Parameter(torch.Tensor(self.n_level, d_model))

        # input_proj_list = []
        # for cin in c1:
        #     input_proj_list.append(nn.Sequential(
        #         nn.Conv2d(cin, d_model, kernel_size=1),
        #         nn.GroupNorm(32, d_model),
        #     ))
        # self.input_proj = nn.ModuleList(input_proj_list)
        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)

        self.msdeformattn = MSDeformAttnKDA(d_model=d_model, n_levels=self.n_level, n_heads=8, n_points=4)
        # self.msdeformattn = MSDeformAttnKDA(d_model=d_model, n_levels=self.n_level, n_heads=8, n_points=4, same_loc=True)
        # self.msdeformattn = MSDeformAttnKDA(d_model=d_model, n_levels=self.n_level, n_heads=d_model//32, n_points=4)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # weight
        # self.alph = nn.Parameter(torch.ones(1))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        """
        生成参考点   reference points  为什么参考点是中心点？  为什么要归一化？
        spatial_shapes: 4个特征图的shape [4, 2]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        device: cuda:0
        """
        reference_points_list = []
        # 遍历4个特征图的shape  比如 H_=100  W_=150
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 0.5 -> 99.5 取100个点  0.5 1.5 2.5 ... 99.5
            # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
            # ref_y: [100, 150]  第一行：150个0.5  第二行：150个1.5 ... 第100行：150个99.5
            # ref_x: [100, 150]  第一行：0.5 1.5...149.5   100行全部相同
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # [100, 150] -> [bs, 15000]  150个0.5 + 150个1.5 + ... + 150个99.5 -> 除以100 归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [100, 150] -> [bs, 15000]  100个: 0.5 1.5 ... 149.5  -> 除以150 归一化
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # [bs, 15000, 2] 每一项都是xy
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
        # [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
        # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
        # 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 4个flatten后特征图的归一化参考点坐标
        return reference_points

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
    def forward(self, input):
        x, y = input
        bs, c, h, w = x.shape

        spatial_shapes = [x.shape[-2:]]
        valid_ratios = torch.ones(bs, self.n_level, 2).to(x.device)
        refer_points = self.get_reference_points(spatial_shapes, valid_ratios, x.device)
        pos_emed = self.posembed(x).to(x.dtype).flatten(2).transpose(1, 2)

        # lvl_pos_embed = []
        # pos_emed = []
        x = x.flatten(2).transpose(1, 2)
        # lvl_pos_embed.append(pos_emed[-1] + self.level_embed[lvl].view(1, 1, -1))
        y = y.flatten(2).transpose(1, 2)

        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device)
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        y1 = self.msdeformattn(self.with_pos_embed(y, pos_emed), refer_points, x, spatial_shapes, value_mask=None)
        y = y + self.dropout1(y1)
        # y = y + self.dropout1(y1) * self.alph
        y = self.norm1(y)

        output = self.forward_ffn(y)
        # output = y

        return output.transpose(1, 2).reshape(bs , self.d_model, h, w)

class DeformCrossatt_Fusion(nn.Module):
    def __init__(self, in_chs):
        super().__init__()
        self.n_levels = len(in_chs)
        self.DCKDA = nn.ModuleList(DeformableCrossattKDAInter(in_chs[i], in_chs[i], in_chs[i]*2)\
                                   for i in range(self.n_levels))

    def forward(self, x_v, x_t):
        x = []
        for i in range(self.n_levels):
            x.append(x_t[i] + self.DCKDA[i]((x_v[i], x_t[i])))
        return x

def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))
def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8)))
def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    # coordinates = list(get_best_begin_point_single(coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates
def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    polys = get_best_begin_point(polys)
    return polys

@ROTATED_DETECTORS.register_module()
class Oriented_rcnn_m(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Oriented_rcnn_m, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        # self.backbone_v = build_backbone(backbone)
        # self.backbone_t = build_backbone(backbone)

        # chs = neck['in_channels']
        # self.fusion_model = Add_fusion(chs)
        # self.fusion_model = DeformCrossatt_Fusion(chs)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        self.fusion = FusionNet(block_num=3, feature_out=False)
        self.criterion_fuse = DetcropPixelLoss()

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = img
        # inputs = torch.cat([F_ir[:,0:1,:,:], F_rgb], dim=1)
        # vis_weight = None
        # inf_weight = None
        x_rgb = self.backbone(OD_RGB)
        x_ir = self.backbone(OD_IR)

        if self.with_neck:
            x_rgb = self.neck(x_rgb)
            x_ir = self.neck(x_ir)
        features = list()
        for i in range(5):
            feature = x_rgb[i]+x_ir[i]
            features.append(feature)
        return features

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat((img, img))
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def change_hsv2rgb(self, fus_img, visimage_clr):
        bri = fus_img.detach().cpu().numpy() * 255
        bri = bri.reshape([fus_img.size()[2], fus_img.size()[3]])
        # bri = np.where(bri < 0, 0, bri)
        # bri = np.where(bri > 255, 255, bri)
        min_value = bri.min()
        max_value = bri.max() 
        scale = 255 / (max_value - min_value) 
        bri = (bri - min_value) * scale
        bri = np.clip(bri, 0, 255)
        im1 = Image.fromarray(bri.astype(np.uint8))

        clr = visimage_clr.cpu().numpy().squeeze().transpose(1, 2, 0)
        clr = np.concatenate((clr, bri.reshape(fus_img.size()[2], fus_img.size()[3], 1)), axis=2)
        clr[:, :, 2] = im1
        clr = cv2.cvtColor(clr.astype(np.uint8), cv2.COLOR_HSV2RGB) #zjq
        
        return clr

    def forward_fusion(self, img):
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = img
        device = 'cuda'
        vi_rgb = OD_RGB.data[0].to(device)
        visimage_bri = RGB_bri.data[0].to(device)
        visimage_clr = RGB_clr.data[0].to(device)
        ir_image = F_ir.data[0][:,0:1,:,:].to(device)
        vi_image = F_rgb.data[0].to(device)
        ir_image = ir_image.to(torch.float32)
        vi_image = vi_image.to(torch.float32)
        visimage_bri = visimage_bri.to(torch.float32)
        ir_rgb = OD_IR.data[0].to(device)
        inputs = torch.cat([ir_image, vi_image], dim=1)
        start = time.time()

        x_rgb = self.backbone(vi_rgb)          #vi_image是/255的，vi_rgb是用的normalizer1
        x_ir = self.backbone(ir_rgb) #3个通道     ir_rgb是用的normalizer1   ir_image是/255的

        if self.with_neck:
            x_rgb = self.neck(x_rgb)
            x_ir = self.neck(x_ir)

        features = list()
        for i in range(4):
            feature = x_rgb[i]+x_ir[i]
            features.append(feature)
        _, res_weight = self.fusion(features,inputs) #这个测试需要修改
        end = time.time()
        time_per_img = end-start
        # cmx
        # greater = torch.gt(res_weight[:, 0:1, :, :], res_weight[:, 1:, :, :])  #ir权重大于bri的话在对应位置返回true
        # num_greater = torch.sum(greater).item()
        # total = res_weight[:, 1:, :, :].numel()

        # if num_greater > int(total * 0.8):
        #     fus_img = ir_image.to(torch.float32) + res_weight[:, 1:, :, :] * visimage_bri.to(torch.float32)#/ 255.
        # elif (total - num_greater) > int(total * 0.8):
        #     fus_img = res_weight[:, 0:1, :, :] * ir_image.to(torch.float32) + visimage_bri.to(torch.float32)#/ 255.
        # else:
        #     fus_img = res_weight[:, 0:1, :, :] * ir_image.to(torch.float32) + res_weight[:, 1:, :, :] * visimage_bri.to(torch.float32)
        fus_img = res_weight[:, 0:1, :, :] * ir_image + res_weight[:, 1:, :, :] * visimage_bri

        fusion_img = self.change_hsv2rgb(fus_img, visimage_clr)
      
        return fus_img, fusion_img, time_per_img, vi_image.squeeze(0)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        fusion_feature = [x1 for x1 in x[:4]]
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = img
        F_ir = F_ir.to(torch.float32)
        F_rgb = F_rgb.to(torch.float32)
        RGB_bri = RGB_bri.to(torch.float32)
        inputs = torch.cat([F_ir[:,0:1,:,:], F_rgb], dim=1)
        vis_weight = None
        inf_weight = None
        _, res_weight = self.fusion(fusion_feature, inputs)
        
        # greater = []
        # num_greater = []
        # total = []
        # for i in range(2):   #2是batch size数
        #     greater.append(torch.gt(res_weight[i:i+1, 0:1, :, :], res_weight[i:i+1, 1:, :, :]))  #ir权重大于bri的话在对应位置返回true
        #     num_greater.append(torch.sum(greater[i]).item())
        #     total.append(res_weight[i:i+1, 1:, :, :].numel())
        num_greater = None
        total = None
        
        # fus_img_temp = []
        # for i in range(2):
        #     if num_greater[i] > int(total[i] * 0.8):
        #         fus_img_temp.append(F_ir[i:i+1, 0:1, :, :] + res_weight[i:i+1, 1:, :, :] * RGB_bri[i:i+1, :, :, :])#/ 255.
        #     elif (total[i] - num_greater[i]) > int(total[i] * 0.8):
        #         fus_img_temp.append(res_weight[i:i+1, 0:1, :, :] * F_ir[i:i+1, 0:1, :, :] + RGB_bri[i:i+1, :, :, :])#/ 255.
        #     else:
        #         fus_img_temp.append(res_weight[i:i+1, 0:1, :, :] * F_ir[i:i+1, 0:1, :, :] + res_weight[i:i+1, 1:, :, :] * RGB_bri[i:i+1, :, :, :])
        # fus_img = torch.cat([fus_img_temp[0], fus_img_temp[1]], dim=0)
        fus_img = res_weight[:, 0:1, :, :] * F_ir[:,0:1,:,:] + res_weight[:, 1:, :, :] * RGB_bri  #origin
        mask_list = []
        for per_image in gt_bboxes:
            mask = torch.zeros(712,840)
            n_cx_cy_w_h_a = per_image.cpu().numpy()
            points = obb2poly_np_le90(n_cx_cy_w_h_a)
            points = torch.tensor(points).to(img[0].device).to(torch.float32)
            for k in range(points.shape[0]):
            # for k in range(per_image.shape[0]):
                #horizontal
                # mask[int(per_image[k][1] - 0.5 * per_image[k][3]):int(per_image[k][1] + 0.5*per_image[k][3]),\
                #      int(per_image[k][0] - 0.5 * per_image[k][2]):int(per_image[k][0] + 0.5*per_image[k][2])]=1
                #horizontal_full_wrap
                max_x, _ = torch.max(torch.stack([points[k][0], points[k][2], points[k][4], points[k][6]]), dim=0)
                min_x, _ = torch.min(torch.stack([points[k][0], points[k][2], points[k][4], points[k][6]]), dim=0)
                max_y, _ = torch.max(torch.stack([points[k][1], points[k][3], points[k][5], points[k][7]]), dim=0)
                min_y, _ = torch.min(torch.stack([points[k][1], points[k][3], points[k][5], points[k][7]]), dim=0)
                mask[int(min_y) : int(max_y), int(min_x) : int(max_x)]=1
            mask_list.append(mask)
       
        mask = torch.stack(mask_list).unsqueeze(1).to(img[0].device)
        pad1 = int((fus_img.shape[-2]-mask.shape[-2]))
        pad2 = int((fus_img.shape[-1]-mask.shape[-1]))
        mask = F.pad(mask, (0,pad2,0,pad1))

        # ### 做mask的可视化
        # plt.imshow(mask.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("mask.png")
        # plt.imshow((fus_img*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("fus_img.png")
        # plt.imshow((F_ir*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("ir_image.png")
        # plt.imshow((F_ir[:,0:1,:,:]*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("ir_image_dan.png")
        # plt.imshow((RGB_bri*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("RGB_bri.png")
        # # 将叠加图像叠加到背景图像上
        # background_image = (F_ir*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
        # overlay_image = (mask*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
        # alpha = 0.5  # 设置叠加图像的透明度
        # x_offset = 0  # 设置叠加图像的水平偏移量
        # y_offset = 0  # 设置叠加图像的垂直偏移量
        # background_image[y_offset:y_offset+overlay_image.shape[0], x_offset:x_offset+overlay_image.shape[1]] = \
        #     alpha * overlay_image + (1 - alpha) * background_image[y_offset:y_offset+overlay_image.shape[0], x_offset:x_offset+overlay_image.shape[1]]
        # # 使用 Matplotlib 显示叠加后的图像
        # plt.imshow(background_image)
        # plt.axis('off')
        # plt.savefig("ir_overlay_mask.png")
            
        SSIM_loss,grad_loss,pixel_loss = self.criterion_fuse(fus_img, RGB_bri, F_ir[:,0:1,:,:], \
                                                             mask, num_greater, total)
        
        losses = dict()
        losses_new = dict()
        # losses['SSIM_loss'] = 10*SSIM_loss 
        # losses['grad_loss'] = 10*grad_loss 
        # losses['pixel_loss'] = 10*pixel_loss 

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        losses_new['fusion_loss'] = 10*SSIM_loss + 10*grad_loss + 10*pixel_loss 
        losses_new['detection_loss'] = losses['loss_rpn_cls'][0] + losses['loss_rpn_cls'][1] + losses['loss_rpn_cls'][2] + losses['loss_rpn_cls'][3] + \
                losses['loss_rpn_cls'][4] +  \
                losses['loss_rpn_bbox'][0] + losses['loss_rpn_bbox'][1] + losses['loss_rpn_bbox'][2] + losses['loss_rpn_bbox'][3] + losses['loss_rpn_bbox'][4] + \
                losses['loss_cls'] + losses['loss_bbox']
        losses_new['acc'] = losses['acc']
        return losses_new
    
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                if isinstance(img, list) or isinstance(img, tuple):
                    img_meta[img_id]['batch_input_shape'] = tuple(img[0].size()[-2:])
                else:
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)