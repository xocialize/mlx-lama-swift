"""MI-GAN generator in MLX-Python (NHWC) — parity vs PyTorch oracle. Validates the fiddly ops
(separable conv, blur down/upsample with filter_const + asym pad, lrelu+gain/clamp, noise) before Swift.
"""
import os
import math
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)
SQRT2 = math.sqrt(2.0)


def load(res=512):
    sd = torch.load(f"{HERE}/migan/migan_{res}_places2.pt", map_location="cpu", weights_only=True)
    return {k: v.float().numpy() for k, v in sd.items()}


W = load(512)


def cw(k):  # conv (O,I,kH,kW) -> NHWC (O,kH,kW,I)
    return mx.array(np.transpose(W[k], (0, 2, 3, 1)))


def arr(k):
    return mx.array(W[k])


def lrelu(x):  # lrelu_agc(alpha=0.2, gain=sqrt2, clamp=256)
    x = mx.where(x >= 0, x, 0.2 * x) * SQRT2
    return mx.clip(x, -256.0, 256.0)


def conv(x, k, b=None, stride=1, pad=0, groups=1):
    y = mx.conv2d(x, cw(k), stride=stride, padding=pad, groups=groups)
    return y + arr(b) if b is not None else y


def downsample(x, prefix):  # depthwise blur k4 s2 p1
    C = x.shape[-1]
    return mx.conv2d(x, cw(prefix + ".filter.weight"), stride=2, padding=1, groups=C)


def nearest2x(x):  # NHWC nearest upsample ×2
    B, H, Wd, C = x.shape
    return mx.broadcast_to(x.reshape(B, H, 1, Wd, 1, C), (B, H, 2, Wd, 2, C)).reshape(B, 2 * H, 2 * Wd, C)


def upsample(x, prefix, res):  # nearest×2 → ×filter_const → pad(2,1,2,1) → blur conv k4
    C = x.shape[-1]
    x = nearest2x(x)
    fc = W[prefix + ".filter_const"]                     # (1,1,res,res)
    x = x * mx.array(np.transpose(fc, (0, 2, 3, 1)))     # (1,res,res,1) broadcast
    x = mx.pad(x, [(0, 0), (2, 1), (2, 1), (0, 0)])
    return mx.conv2d(x, cw(prefix + ".filter.weight"), stride=1, padding=0, groups=C)


def separable(x, prefix, act=True, down=False, up=None, noise=False):
    x = conv(x, prefix + ".conv1.weight", b=prefix + ".conv1.bias",
             pad=1, groups=x.shape[-1])                  # depthwise k3
    if act:
        x = lrelu(x)
    if down:
        x = downsample(x, prefix + ".downsample")
    x = conv(x, prefix + ".conv2.weight")                # pointwise 1x1
    if up is not None:
        x = upsample(x, prefix + ".upsample", up)
    if noise:
        x = x + arr(prefix + ".noise_const").reshape(1, x.shape[1], x.shape[2], 1) * arr(prefix + ".noise_strength")
    if act:
        x = lrelu(x)
    return x


def encoder(img):  # img NHWC (1,512,512,4)
    feats = {}
    x = None
    for resi in [512, 256, 128, 64, 32, 16, 8]:
        p = f"encoder.b{resi}"
        if resi == 512:
            y = lrelu(conv(img, p + ".fromrgb.weight", b=p + ".fromrgb.bias"))
            x = y if x is None else x + y
        feat = separable(x, p + ".conv1")
        x = separable(feat, p + ".conv2", down=True)
        feats[resi] = feat
    feat = separable(x, "encoder.b4.conv1")
    x = separable(feat, "encoder.b4.conv2", down=False)
    feats[4] = feat
    return x, feats


def synthesis(x, feats):
    # b4 (SynthesisBlockFirst): conv1, +feat4, conv2, torgb
    x = separable(x, "synthesis.b4.conv1")
    x = x + feats[4]
    x = separable(x, "synthesis.b4.conv2")
    img = conv(x, "synthesis.b4.torgb.weight", b="synthesis.b4.torgb.bias")
    for resj in [8, 16, 32, 64, 128, 256, 512]:
        p = f"synthesis.b{resj}"
        x = separable(x, p + ".conv1", up=resj, noise=True)
        x = x + feats[resj]
        x = separable(x, p + ".conv2", noise=True)
        img = upsample(img, p + ".upsample", resj)
        img = img + conv(x, p + ".torgb.weight", b=p + ".torgb.bias")
    return img


def main():
    inp = np.load(f"{HERE}/goldens/migan_input4.npy")           # (1,4,512,512) NCHW
    x = mx.array(np.transpose(inp, (0, 2, 3, 1)))               # NHWC
    enc, feats = encoder(x)
    out = synthesis(enc, feats)                                 # (1,512,512,3) NHWC
    gold = np.transpose(np.load(f"{HERE}/goldens/migan_output.npy"), (0, 2, 3, 1))
    d = np.max(np.abs(np.array(out) - gold))
    m = np.mean(np.abs(np.array(out) - gold))
    print(f"migan output  max_abs={d:.3e} mean={m:.3e}  {'OK ✅' if d < 1e-2 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
