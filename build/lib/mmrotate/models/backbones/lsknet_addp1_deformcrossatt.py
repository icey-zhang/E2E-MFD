import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


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



@ROTATED_BACKBONES.register_module()
class LSKNet_addp1_deformcrossatt(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims_main=[64, 128, 256, 512], embed_dims_aux=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths_main=[3, 4, 6, 3], depths_aux=[3, 4, 6, 3], num_stages=4, fusion_stage=1,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths_main = depths_main
        self.depths_aux = depths_aux
        self.num_stages = num_stages
        assert fusion_stage > 0 and fusion_stage < num_stages, 'fusion_stage must greater than 0 and less than num_stages!'
        self.fusion_stage = fusion_stage

        dpr_main = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_main))]  # stochastic depth decay rule
        dpr_aux = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_aux))]  # stochastic depth decay rule
        cur_main = 0
        cur_aux = 0

        for i in range(num_stages):
            patch_embed_main = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims_main[i - 1],
                                            embed_dim=embed_dims_main[i], norm_cfg=norm_cfg)

            block_main = nn.ModuleList([Block(
                dim=embed_dims_main[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr_main[cur_main + j],norm_cfg=norm_cfg)
                for j in range(depths_main[i])])
            norm_main = norm_layer(embed_dims_main[i])
            cur_main += depths_main[i]

            setattr(self, f"patch_embed_main{i + 1}", patch_embed_main)
            setattr(self, f"block_main{i + 1}", block_main)
            setattr(self, f"norm_main{i + 1}", norm_main)

            patch_embed_aux = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims_aux[i - 1],
                                            embed_dim=embed_dims_aux[i], norm_cfg=norm_cfg)

            block_aux = nn.ModuleList([Block(
                dim=embed_dims_aux[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr_aux[cur_aux + j],norm_cfg=norm_cfg)
                for j in range(depths_aux[i])])
            norm_aux = norm_layer(embed_dims_aux[i])
            cur_aux += depths_aux[i]

            setattr(self, f"patch_embed_aux{i + 1}", patch_embed_aux)
            setattr(self, f"block_aux{i + 1}", block_aux)
            setattr(self, f"norm_aux{i + 1}", norm_aux)

            if i >= fusion_stage:
                DCKDAInter = DeformableCrossattKDAInter(embed_dims_main[i], embed_dims_main[i], 
                                                        embed_dims_main[i]*2)
                setattr(self, f"DCKDAInter{i + 1}", DCKDAInter)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet_addp1_deformcrossatt, self).init_weights()
    
    def freeze_patch_emb(self):
        self.patch_embed_main1.requires_grad = False
        self.patch_embed_aux1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed_main1', 'pos_embed_main2', 'pos_embed_main3', 'pos_embed_main4', 
                'pos_embed_aux1', 'pos_embed_aux2', 'pos_embed_aux3', 'pos_embed_aux4', 
                'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim_main, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x_main, x_aux = x
        B = x_main.shape[0]
        outs = []
        for i in range(self.fusion_stage):
            patch_embed_main = getattr(self, f"patch_embed_main{i + 1}")
            block_main = getattr(self, f"block_main{i + 1}")
            norm_main = getattr(self, f"norm_main{i + 1}")
            x_main, H, W = patch_embed_main(x_main)
            for blk in block_main:
                x_main = blk(x_main)
            x_main = x_main.flatten(2).transpose(1, 2)
            x_main = norm_main(x_main)
            x_main = x_main.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            patch_embed_aux = getattr(self, f"patch_embed_aux{i + 1}")
            block_aux = getattr(self, f"block_aux{i + 1}")
            norm_aux = getattr(self, f"norm_aux{i + 1}")
            x_aux, H, W = patch_embed_aux(x_aux)
            for blk in block_aux:
                x_aux = blk(x_aux)
            x_aux = x_aux.flatten(2).transpose(1, 2)
            x_aux = norm_aux(x_aux)
            x_aux = x_aux.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x_main + x_aux)
        
        x_main = outs[-1]

        for i in range(self.fusion_stage, self.num_stages):
            dckdainter = getattr(self, f"DCKDAInter{i + 1}")

            patch_embed_main = getattr(self, f"patch_embed_main{i + 1}")
            block_main = getattr(self, f"block_main{i + 1}")
            norm_main = getattr(self, f"norm_main{i + 1}")
            x_main, H, W = patch_embed_main(x_main)
            for blk in block_main:
                x_main = blk(x_main)
            x_main = x_main.flatten(2).transpose(1, 2)
            x_main = norm_main(x_main)
            x_main = x_main.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            patch_embed_aux = getattr(self, f"patch_embed_aux{i + 1}")
            block_aux = getattr(self, f"block_aux{i + 1}")
            norm_aux = getattr(self, f"norm_aux{i + 1}")
            x_aux, H, W = patch_embed_aux(x_aux)
            for blk in block_aux:
                x_aux = blk(x_aux)
            x_aux = x_aux.flatten(2).transpose(1, 2)
            x_aux = norm_aux(x_aux)
            x_aux = x_aux.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            outs.append(x_main + dckdainter((x_aux, x_main)))
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

