import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import ConvModule


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:
                    # print(grad)
                    # TODO
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

class Encoder(nn.Module):
    def __init__(self, c1,c2):
        super(Encoder, self).__init__()

        block1 = []
        
        block1.append(FusionBlock(in_block=4, out_block=32, k_size=3))#nn.Sequential(*block1)
        block1.append(FusionBlock(in_block=64, out_block=64, k_size=3))
        self.block1 =nn.Sequential(*block1)

        self.conv2 = nn.Conv2d(c2*2, c2//2, 1, bias=True)
        self.relu = nn.ReLU()
        self.att1 = Attentionregion(M=2, res_channels=256)
        self.att2 =Attentionregion(M=2, res_channels=256)
        self.att3 = Attentionregion(M=2, res_channels=256)

        self.conv_module = ConvModule(c2//2, c2//2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.conv_module_act = ConvModule(c2//2, c2//2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.conv_module_add = ConvModule(c2//2, c2//2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.sigmoid = h_sigmoid()
        self._init_weight()

    def forward(self, x, low_level_feat,factor=2):
        x1,x2,x3,x4 = x[0],x[1],x[2],x[3]

        low_level_feat = self.block1(low_level_feat)
        # import matplotlib.pylab as plt
        # for i in range(low_level_feat.shape[1]):
        #     x1_show = (low_level_feat[0,i,:,:]).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./low_level_feat/{}.png".format(i))
        # import matplotlib.pylab as plt
        # for i in range(x[0].shape[1]):
        #     x1_show = (x[0][0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./x1_before/{}.png".format(i))

        # import matplotlib.pylab as plt
        # for i in range(x[1].shape[1]):
        #     x1_show = (x[1][0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./x2_before/{}.png".format(i))

        # import matplotlib.pylab as plt
        # for i in range(x[2].shape[1]):
        #     x1_show = (x[2][0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./x3_before/{}.png".format(i))

        x1 = self.att1(x1)
        x2 = self.att2(x2)
        x3 = self.att3(x3)
        # import matplotlib.pylab as plt
        # for i in range(256,256+5):
        #     x1_show = (x1[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./fea_show/x1/{}.png".format(i))
        # # import matplotlib.pylab as plt
        # for i in range(256,256+5):
        #     x2_show = (x2[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x2_show)
        #     plt.savefig("./fea_show/x2/{}.png".format(i))
        # # import matplotlib.pylab as plt
        # for i in range(256,256+5):
        #     x3_show = (x3[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x3_show)
        #     plt.savefig("./fea_show/x3/{}.png".format(i))
        # import matplotlib.pylab as plt
        # for i in range(x2.shape[1]):
        #     x2_show = (x2[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x2_show)
        #     plt.savefig("./x2/{}.png".format(i))

        # import matplotlib.pylab as plt
        # for i in range(x2.shape[1]):
        #     x2_show = (x2[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x2_show)
        #     plt.savefig("./x2/{}.png".format(i))

        x1 = F.interpolate(x1, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x = x1+x2+x3
        x = self.relu(self.conv2(x))

        x_origin = self.conv_module(low_level_feat)
        x_act = self.conv_module_act(x)
        x_act = self.sigmoid(x_act)
        x_out = x_origin * x_act + x_origin
        # x_origin = self.conv_module(x)
        # x_act = self.conv_module_act(low_level_feat)
        # x_act = self.sigmoid(x_act)
        # x_out = x_origin * x_act + x_origin
        return x_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaConv2d):
                init.kaiming_normal(m.weight)

class Attentionregion(nn.Module):

    def __init__(self, M=32, res_channels=2048, pooling_mode='GAP', add_lambda=0.8):
        super(Attentionregion, self).__init__()
        self.M = M
        self.base_channels = res_channels
        self.out_channels = M * res_channels
        # self.conv = BasicConv2d(res_channels, self.M, kernel_size=1)
        self.conv = SKConv(res_channels,self.M)
        # self.conv = ScConv(res_channels,self.M)
        # self.conv = MetaConv2d(
        #     in_channels=res_channels,
        #     out_channels=self.M,
        #     kernel_size=1,
        #     stride=1,
        #     padding=(1 - 1) // 2,
        #     bias=True
        # )
        # self.conv = ConvModule(res_channels, self.M, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.EPSILON = 1e-6

    def bilinear_attention_pooling(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        feature_matrix = []
        for i in range(M):
            AiF = features * attentions[:, i:i + 1, ...]
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        # feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        # feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        return feature_matrix

    def forward(self, x):

        attention_maps = self.conv(x)
        feature_matrix = self.bilinear_attention_pooling(x, attention_maps)

        return feature_matrix
    
class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d),
            nn.ReLU(),
            nn.Linear(d, out_channels * M)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        feats = [conv(x) for conv in self.convs]
        U = sum(feats)  #4,3,512,512
        s = self.global_pool(U)      # 4,3,1,1
        z = self.fc(s.view(batch_size, -1))   #4,6
        z = z.view(batch_size, -1, len(self.convs))  #4,3,2
        a = self.softmax(z)
        a = a.unsqueeze(-1).unsqueeze(-1)  #4,3,2,1,1
        b1 = a[:, :, 0:1, :, :] * feats[0].unsqueeze(2)
        b2 = a[:, :, 1:, :, :] * feats[1].unsqueeze(2)
        V = torch.sum(torch.cat([b1,b2], dim=2), dim=2)
        return V

class FusionBlock(MetaModule):
    def __init__(self, in_block, out_block, k_size=3):
        super(FusionBlock, self).__init__()
        self.conv1_1 = MetaConv2d(
            in_channels=in_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_2 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_3 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )

        self.conv1_0_00 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_0_01 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.relu = nn.ReLU()
        # self.tree1 = Tree(out_block)
        # self.tree2 = Tree(out_block)
        # self.tree3 = Tree(out_block)


    def forward(self, x):
        x = self.conv1_1(x)
        # x = self.tree1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        # x = self.tree2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        # x = self.tree3(x)
        x = self.relu(x)

        # x_1 = self.conv1_1(x)
        # x_1_t = self.tree1(x_1)
        # x_1_t = self.relu(x_1_t)
        # x_2 = self.conv1_2(x)
        # x_2_t = self.tree2(x_2)
        # x_2_t = self.relu(x_2_t)
        # x_3 = self.conv1_3(x)
        # x_3_t = self.tree3(x_3)
        # x_3_t = self.relu(x_3_t)
        # x = x_1_t+x_2_t+x_3_t

        x0 = self.conv1_0_00(x)
        # x0 = self.tree3(x0)
        x0 = self.relu(x0)
        x1 = self.conv1_0_01(x)
        # x1 = self.tree4(x1)
        x1 = self.relu(x1)

        return torch.cat([x0, x1], dim=1)

class FusionNet(MetaModule):
    def __init__(self, block_num, feature_out):
        super(FusionNet, self).__init__()
        self.feature_out = feature_out
        self.block1 = Encoder(4,256)
        self.block2_in = 128
        self.block2 = nn.Sequential(
            nn.Conv2d(self.block2_in, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x1,x2):
        x = self.block1(x1,x2)
        x = self.block2(x)

        return None, x