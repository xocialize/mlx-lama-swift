"""Anchor the oracle to upstream: lama_gen.py vs an output of the original advimman/lama code.

lama_gen.py is a trimmed copy of advimman/lama, so the parity gate only proves that MLX matches
the copy. Carve/LaMa-ONNX ships `output_orig.png`, the original PyTorch model's result on its
`image.jpg` + `mask.png`. This reproduces their preprocessing (PIL default resize to 512², mask > 0)
and compares fills. Download the three files into one directory first (data only, nothing runs):

    https://huggingface.co/Carve/LaMa-ONNX/resolve/main/{image.jpg,mask.png,output_orig.png}
    python check_upstream.py <that directory>

Measured 2026-10-09: hole |d| mean 0.50, max 1 level (uint8 truncation in their PNG).
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ckpt import generator_state_dict  # noqa: E402
from lama_gen import FFCResNetGenerator  # noqa: E402


def main():
    d = sys.argv[1]
    ref = np.array(Image.open(os.path.join(d, "output_orig.png")).convert("RGB")).astype(np.float32)
    rgb = np.array(Image.open(os.path.join(d, "image.jpg")).convert("RGB").resize((512, 512))).astype(np.float32) / 255
    mask = (np.array(Image.open(os.path.join(d, "mask.png")).convert("L").resize((512, 512))) > 0).astype(np.float32)

    torch.set_grad_enabled(False)
    gen = FFCResNetGenerator().eval()
    gen.load_state_dict(generator_state_dict(os.path.join(HERE, "big-lama/models/best.ckpt")), strict=True)
    img = torch.from_numpy(rgb.transpose(2, 0, 1))[None]
    m = torch.from_numpy(mask)[None, None]
    pred = gen(torch.cat([img * (1 - m), m], 1))
    res = ((m * pred + (1 - m) * img)[0].numpy().transpose(1, 2, 0) * 255).clip(0, 255)

    hole = mask > 0.5
    diff = np.abs(res - ref)[hole]
    ok = diff.max() <= 2
    print(f"[upstream] hole {hole.sum()} px  |d| mean {diff.mean():.2f}  max {diff.max():.1f} levels  "
          f"hole mean RGB ours {res[hole].mean(0).round(1)} upstream {ref[hole].mean(0).round(1)}  "
          f"{'OK ✅' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
