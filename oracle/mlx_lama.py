"""Full Big-LaMa FFCResNetGenerator in MLX-Python (NHWC) — parity vs PyTorch goldens.

Validates the whole graph before Swift transcription. Reuses the validated FourierUnit logic.
"""
import os
import sys
import types
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)


def load_weights():
    for n in ["pytorch_lightning", "pytorch_lightning.callbacks", "pytorch_lightning.callbacks.model_checkpoint"]:
        m = types.ModuleType(n); sys.modules[n] = m
        class D:
            def __init__(s, *a, **k): pass
            def __setstate__(s, st): pass
        m.ModelCheckpoint = D; m.Callback = D
    ck = torch.load(f"{HERE}/big-lama/models/best.ckpt", map_location="cpu", weights_only=False)
    return {k[len("generator."):]: v.float().numpy()
            for k, v in ck["state_dict"].items() if k.startswith("generator.")}


W = load_weights()


def cw(k):  # conv weight (O,I,kH,kW) -> NHWC (O,kH,kW,I)
    return mx.array(np.transpose(W[k], (0, 2, 3, 1)))


def arr(k):
    return mx.array(W[k])


def reflect_pad(x, p):  # NHWC reflect pad over H,W (no edge repeat)
    if p == 0:
        return x
    H, Wd = x.shape[1], x.shape[2]
    def idx(N):
        return mx.array(np.concatenate([np.arange(p, 0, -1), np.arange(N), np.arange(N - 2, N - 2 - p, -1)]).astype(np.int32))
    x = mx.take(x, idx(H), axis=1)
    x = mx.take(x, idx(Wd), axis=2)
    return x


def conv(x, w, b=None, stride=1, pad=0):
    y = mx.conv2d(x, w, stride=stride, padding=pad)
    return y + b if b is not None else y


def bn(x, prefix, eps=1e-5):
    return (x - arr(prefix + ".running_mean")) / mx.sqrt(arr(prefix + ".running_var") + eps) \
        * arr(prefix + ".weight") + arr(prefix + ".bias")


def relu(x):
    return mx.maximum(x, 0)


def fourier_unit(x, prefix):  # x NHWC
    B, H, Wd, C = x.shape
    ft = mx.fft.rfftn(x, axes=[1, 2], norm="ortho")
    inter = mx.stack([ft.real, ft.imag], axis=-1).reshape(B, H, ft.shape[2], 2 * C)
    y = conv(inter, cw(prefix + ".conv_layer.weight"))
    y = relu(bn(y, prefix + ".bn"))
    yc = y.reshape(B, H, ft.shape[2], C, 2)
    comp = yc[..., 0] + 1j * yc[..., 1]
    return mx.fft.irfftn(comp, s=[H, Wd], axes=[1, 2], norm="ortho")


def spectral_transform(xg, prefix):  # convg2g
    x = relu(bn(conv(xg, cw(prefix + ".conv1.0.weight")), prefix + ".conv1.1"))
    out = fourier_unit(x, prefix + ".fu")
    return conv(x + out, cw(prefix + ".conv2.weight"))


def ffc_bn_act(xl, xg, prefix, gin, gout, k, stride):
    p = (k - 1) // 2
    out_xl = out_xg = None
    if gout != 1:
        out_xl = conv(reflect_pad(xl, p), cw(prefix + ".ffc.convl2l.weight"), stride=stride)
        if gin > 0:
            out_xl = out_xl + conv(reflect_pad(xg, p), cw(prefix + ".ffc.convg2l.weight"), stride=stride)
    if gout != 0:
        out_xg = conv(reflect_pad(xl, p), cw(prefix + ".ffc.convl2g.weight"), stride=stride)
        if gin > 0:
            out_xg = out_xg + spectral_transform(xg, prefix + ".ffc.convg2g")
    xl = relu(bn(out_xl, prefix + ".bn_l")) if out_xl is not None else None
    xg = relu(bn(out_xg, prefix + ".bn_g")) if out_xg is not None else None
    return xl, xg


def resblock(xl, xg, prefix):
    idl, idg = xl, xg
    xl, xg = ffc_bn_act(xl, xg, prefix + ".conv1", 0.75, 0.75, 3, 1)
    xl, xg = ffc_bn_act(xl, xg, prefix + ".conv2", 0.75, 0.75, 3, 1)
    return idl + xl, idg + xg


def conv_transpose(x, prefix):  # ConvTranspose2d s2 p1 op1; PT weight (in,out,kH,kW)
    w = W[prefix + ".weight"]                       # (in, out, kH, kW)
    wm = mx.array(np.transpose(w, (1, 2, 3, 0)))    # try (out, kH, kW, in)
    y = mx.conv_transpose2d(x, wm, stride=2, padding=1, output_padding=1)
    return y + arr(prefix + ".bias")


def generator(x):  # x NHWC (1,S,S,4)
    x = reflect_pad(x, 3)
    x = relu(bn(conv(x, cw("model.1.ffc.convl2l.weight")), "model.1.bn_l"))
    xl, xg = x, None
    xl, xg = ffc_bn_act(xl, xg, "model.2", 0, 0, 3, 2)
    xl, xg = ffc_bn_act(xl, xg, "model.3", 0, 0, 3, 2)
    xl, xg = ffc_bn_act(xl, xg, "model.4", 0, 0.75, 3, 2)
    for i in range(5, 23):
        xl, xg = resblock(xl, xg, f"model.{i}")
    x = mx.concatenate([xl, xg], axis=-1)           # ConcatTupleLayer → 512ch
    return x


def main():
    inp = np.load(f"{HERE}/goldens/input4.npy")                  # (1,4,S,S) NCHW
    x = mx.array(np.transpose(inp, (0, 2, 3, 1)))                # NHWC
    bott = generator(x)
    gold_b = np.transpose(np.load(f"{HERE}/goldens/bottleneck.npy"), (0, 2, 3, 1))
    db = np.max(np.abs(np.array(bott) - gold_b))
    print(f"bottleneck  max_abs={db:.3e}  {'OK' if db < 1e-2 else 'FAIL'}")

    # upsample + head
    x = bott
    for pfx in ["model.24", "model.27", "model.30"]:
        x = relu(bn(conv_transpose(x, pfx), pfx[:-2] + str(int(pfx.split('.')[1]) + 1)))
    x = reflect_pad(x, 3)
    x = conv(x, cw("model.34.weight"), arr("model.34.bias"))
    pred = 1 / (1 + mx.exp(-x))                                  # sigmoid
    gold_p = np.transpose(np.load(f"{HERE}/goldens/predicted.npy"), (0, 2, 3, 1))
    dp = np.max(np.abs(np.array(pred) - gold_p))
    print(f"predicted   max_abs={dp:.3e}  {'OK' if dp < 1e-2 else 'FAIL'}")


if __name__ == "__main__":
    main()
