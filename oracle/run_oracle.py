"""Run Big-LaMa (PyTorch, fp32 CPU) on a COLOUR image with a large hole; write the Swift parity fixture.

LaMa pipeline: input = cat[img*(1-mask), mask] (4ch) → generator → predicted (3ch, sigmoid) →
result = mask*predicted + (1-mask)*img.

The fixture must be colour. The first gate ran on a greyscale photo, and with R = G = B a fault
that depends on colour is invisible: the bf16 weight file cast large fills pink (AB-T-0213) while
that gate read 3.2e-5. The default scene is procedural (seeded, no file to go missing, R, G and B
all different) with a hole over a third of the frame. `--image`/`--box` run a real photo instead.

    python run_oracle.py                                    # goldens/colour.safetensors
    python run_oracle.py --image photo.png --box x,y,w,h --size 512 --name photo
    lama-smoke --weights lama_fp32.safetensors --parity goldens/colour.safetensors

The fixture holds `input4` and `predicted`, NHWC fp32.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ckpt import generator_state_dict  # noqa: E402
from lama_gen import FFCResNetGenerator  # noqa: E402


def colour_scene(S, seed=0):
    """Wood-grain table under a blue-green wall, a red mug-like block on the table. RGB [0,1]."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:S, 0:S].astype(np.float32) / S
    wall = np.stack([0.30 + 0.15 * x, 0.55 + 0.10 * y, 0.62 - 0.12 * x], -1)
    grain = 0.06 * np.sin(2 * np.pi * (14 * y + 2.5 * np.sin(2 * np.pi * 1.3 * x)))
    table = np.stack([0.72 + grain, 0.45 + 0.8 * grain, 0.20 + 0.5 * grain], -1)
    img = np.where((y > 0.55)[..., None], table, wall)
    mug = (np.abs(x - 0.50) < 0.13) & (y > 0.38) & (y < 0.72)
    img[mug] = (0.78, 0.12, 0.10)
    img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return img.clip(0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="real photo; default: the procedural colour scene")
    ap.add_argument("--box", help="hole x,y,w,h in the resized frame (default: over the mug, ~33%% of the frame)")
    ap.add_argument("--size", type=int, default=512, help="square side the photo is resized to")
    ap.add_argument("--name", default="colour")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "big-lama/models/best.ckpt"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    gen = FFCResNetGenerator().eval()
    gen.load_state_dict(generator_state_dict(args.ckpt), strict=True)

    S = args.size
    if args.image:
        bgr = cv2.resize(cv2.imread(args.image), (S, S), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        rgb = colour_scene(S)
    x, y, w, h = map(int, args.box.split(",")) if args.box else (S * 6 // 32, S * 9 // 32, S * 20 // 32, S * 15 // 32)
    mask = np.zeros((S, S), np.float32)
    mask[y:y + h, x:x + w] = 1.0

    img = torch.from_numpy(rgb.transpose(2, 0, 1))[None]       # (1,3,S,S)
    m = torch.from_numpy(mask)[None, None]                     # (1,1,S,S)
    inp = torch.cat([img * (1 - m), m], dim=1)                 # (1,4,S,S)
    predicted = gen(inp)                                       # (1,3,S,S) sigmoid
    result = m * predicted + (1 - m) * img

    out = os.path.join(HERE, "goldens"); os.makedirs(out, exist_ok=True)
    save_file({"input4": inp.numpy().transpose(0, 2, 3, 1).copy(),
               "predicted": predicted.numpy().transpose(0, 2, 3, 1).copy()},
              os.path.join(out, f"{args.name}.safetensors"))

    def png(t, p):
        a = (t[0].numpy().transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)
        cv2.imwrite(p, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    png(img * (1 - m) + m, os.path.join(out, f"{args.name}_masked.png"))
    png(result, os.path.join(out, f"{args.name}_result.png"))
    hole = mask > 0.5
    print(f"[oracle] {args.name}: {S}x{S}, hole {hole.mean():.0%} of the frame, "
          f"hole mean RGB {(result[0].numpy().transpose(1, 2, 0)[hole].mean(0) * 255).round(1)} "
          f"→ goldens/{args.name}.safetensors")


if __name__ == "__main__":
    main()
