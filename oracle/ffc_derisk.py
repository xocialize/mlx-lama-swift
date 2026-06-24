"""De-risk the LaMa FFC make-or-break op: the FourierUnit spectral block.

PyTorch FourierUnit (NCHW, copied verbatim from advimman/lama saicinpainting ffc.py) vs a
from-scratch NHWC MLX reimplementation, same random weights. Validates the rFFT axes/norm, the
real/imag interleave into 2C channels, the 1x1 conv + eval-BN + ReLU, and the inverse rFFT.

Gate: max_abs < 1e-4 (CPU fp32). If this passes, the rest of LaMa is standard conv.
"""
import numpy as np
import torch
import torch.nn as nn
import mlx.core as mx

mx.set_default_device(mx.cpu)
torch.manual_seed(0)


# ---- PyTorch reference (big-lama FourierUnit: no SE / no pos-enc / no spatial scale, fft_norm='ortho') ----
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, fft_norm='ortho'):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim, norm=self.fft_norm)


# ---- NHWC MLX reimplementation ----
def fourier_unit_mlx(x_nhwc, convw, bn_w, bn_b, bn_rm, bn_rv, eps=1e-5):
    B, H, Wd, C = x_nhwc.shape
    ft = mx.fft.rfftn(x_nhwc, axes=[1, 2], norm="ortho")           # (B,H,W//2+1,C) complex
    inter = mx.stack([ft.real, ft.imag], axis=-1).reshape(B, H, ft.shape[2], 2 * C)  # [c0r,c0i,...]
    y = mx.conv2d(inter, convw, stride=1, padding=0)               # 1x1 conv, NHWC
    y = (y - bn_rm) / mx.sqrt(bn_rv + eps) * bn_w + bn_b           # eval BN over last axis
    y = mx.maximum(y, 0)                                           # ReLU
    yc = y.reshape(B, H, ft.shape[2], C, 2)
    comp = yc[..., 0] + 1j * yc[..., 1]
    return mx.fft.irfftn(comp, s=[H, Wd], axes=[1, 2], norm="ortho")  # (B,H,W,C)


def main():
    B, C, H, W = 1, 16, 32, 40
    fu = FourierUnit(C, C).eval()
    # randomize BN running stats so eval-BN is non-trivial
    with torch.no_grad():
        fu.bn.running_mean.normal_(0, 1); fu.bn.running_var.uniform_(0.5, 1.5)
        fu.bn.weight.normal_(0, 1); fu.bn.bias.normal_(0, 1)

    x = torch.randn(B, C, H, W)
    with torch.no_grad():
        gold = fu(x).numpy()                                       # NCHW

    # MLX: NHWC input + weights
    x_nhwc = mx.array(x.permute(0, 2, 3, 1).numpy())
    convw = mx.array(fu.conv_layer.weight.detach().numpy().transpose(0, 2, 3, 1))  # (2C,1,1,2C)
    out = fourier_unit_mlx(
        x_nhwc, convw,
        mx.array(fu.bn.weight.detach().numpy()), mx.array(fu.bn.bias.detach().numpy()),
        mx.array(fu.bn.running_mean.detach().numpy()), mx.array(fu.bn.running_var.detach().numpy()))
    out_nchw = np.array(out).transpose(0, 3, 1, 2)

    d = np.max(np.abs(gold - out_nchw))
    print(f"FourierUnit parity: max_abs={d:.3e}  {'OK ✅' if d < 1e-4 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
