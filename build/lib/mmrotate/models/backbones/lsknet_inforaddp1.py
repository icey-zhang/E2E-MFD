import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
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

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Infor_addfusion(nn.Module):
    def __init__(self, c1, c2, scale=4):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c1, 1, 1, autopad(1, None, 1))
        self.cv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(c1, c1, 1, 1, autopad(1, None, 1))
        self.cv2 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        self.softmax = nn.Softmax(dim=1)

        self.scale = scale

    def forward(self, x):
        x1, x2 = x
        bs, c, height, width = x1.shape
        img_h = 512 // self.scale
        img_w = 640 // self.scale
        x1 = x1[:, :, (height - img_h) // 2:-(height - img_h) // 2, (width - img_w) // 2:-(width - img_w) // 2]
        x2 = x2[:, :, (height - img_h) // 2:-(height - img_h) // 2, (width - img_w) // 2:-(width - img_w) // 2]
        x1 = self.conv1(x1)
        x1 = self.cv1(torch.cat([torch.mean(x1, 1, keepdim=True), torch.max(x1, 1, keepdim=True)[0]], 1))
        x2 = self.conv2(x2)
        x2 = self.cv2(torch.cat([torch.mean(x2, 1, keepdim=True), torch.max(x2, 1, keepdim=True)[0]], 1))

        std1 = torch.std(x1, [-1, -2])
        x1 = torch.abs(F.conv2d(x1, self.weight_x)) + torch.abs(F.conv2d(x1, self.weight_y))
        avg_grad1 = torch.sum(x1, [-1, -2]) / (x1.shape[-1] * x1.shape[-2])
        w1 = std1 + avg_grad1

        std2 = torch.std(x2, [-1, -2])
        x2 = torch.abs(F.conv2d(x2, self.weight_x)) + torch.abs(F.conv2d(x2, self.weight_y))
        avg_grad2 = torch.sum(x2, [-1, -2]) / (x2.shape[-1] * x2.shape[-2])
        w2 = std2 + avg_grad2        

        w = self.softmax(torch.cat([w1, w2], 1))
        w1 = w[:, 0].expand(c, height, width, bs).permute(3, 0, 1, 2)
        w2 = 1 - w1
        # if w1.min() < w2.min():
        #     raise ValueError('heihei')
        return x[0]*w1 + x[1]*w2


@ROTATED_BACKBONES.register_module()
class LSKNet_inforaddp1(BaseModule):
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
        self.infor_addfusion = Infor_addfusion(embed_dims_main[fusion_stage - 1], embed_dims_main[fusion_stage - 1],
                                               scale=4)

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


            if i < fusion_stage:
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
            super(LSKNet_inforaddp1, self).init_weights()
    
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

            if i == self.fusion_stage - 1:
                outs.append(self.infor_addfusion((x_main, x_aux)))
            else:
                outs.append(x_main + x_aux)
        
        x = outs[-1]

        for i in range(self.fusion_stage, self.num_stages):
            patch_embed = getattr(self, f"patch_embed_main{i + 1}")
            block = getattr(self, f"block_main{i + 1}")
            norm = getattr(self, f"norm_main{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
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

