"""Standalone Big-LaMa FFCResNetGenerator (PyTorch) — copied from advimman/lama saicinpainting
ffc.py, trimmed to the big-lama config (enable_lfu=False, no SE / spatial-transform / gating / pos-enc).
No saicinpainting deps → loads cleanly for golden generation.
"""
import torch
import torch.nn as nn


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, fft_norm='ortho'):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1).permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
        super().__init__()
        self.downsample = nn.Identity()  # big-lama stride=1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, fft_norm='ortho')
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        return self.conv2(x + output)


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, padding_type='reflect'):
        super().__init__()
        in_cg = int(in_channels * ratio_gin); in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout); out_cl = out_channels - out_cg
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        m = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = m(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type) if m is nn.Conv2d else nn.Identity()
        m = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = m(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type) if m is nn.Conv2d else nn.Identity()
        m = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = m(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type) if m is nn.Conv2d else nn.Identity()
        m = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = m(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2) if m is SpectralTransform else nn.Identity()

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity, padding_type='reflect'):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                       stride, padding, dilation, groups, bias, padding_type=padding_type)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        gch = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - gch)
        self.bn_g = gnorm(gch)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True) if lact is not nn.Identity else nn.Identity()
        self.act_g = gact(inplace=True) if gact is not nn.Identity else nn.Identity()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l)) if not isinstance(x_l, int) else x_l
        x_g = self.act_g(self.bn_g(x_g)) if not isinstance(x_g, int) else x_g
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type)
        self.conv2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


def get_activation(kind):
    return {'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU()}[kind]


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 add_out_act='sigmoid', max_features=1024):
        super().__init__()
        init_gout, ds_gin, res_g = 0, 0, 0.75
        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, 7, ratio_gin=0, ratio_gout=init_gout,
                            norm_layer=norm_layer, activation_layer=activation_layer)]
        for i in range(n_downsampling):
            mult = 2 ** i
            gout = res_g if i == n_downsampling - 1 else ds_gin
            gin = ds_gin if i < n_downsampling - 1 else ds_gin  # input local until last; last gin=0
            model += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2),
                                 3, ratio_gin=(0 if i == 0 else ds_gin), ratio_gout=gout, stride=2, padding=1,
                                 norm_layer=norm_layer, activation_layer=activation_layer)]
        mult = 2 ** n_downsampling
        feats = min(max_features, ngf * mult)
        for _ in range(n_blocks):
            model += [FFCResnetBlock(feats, padding_type, norm_layer, activation_layer)]
        model += [ConcatTupleLayer()]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)),
                                         3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))), up_activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0)]
        if add_out_act:
            model.append(get_activation('sigmoid' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
