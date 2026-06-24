"""MI-GAN oracle: load official weights into the standalone inference Generator, inpaint a real
masked image, dump goldens for MLX parity. Input = cat[mask-0.5, img*mask] (img in [-1,1], mask
1=keep/0=hole); output in [-1,1]; composite = img*mask + out*(1-mask).
"""
import os
import sys
import numpy as np
import cv2
import torch

REPO = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/MI-GAN"
sys.path.insert(0, REPO)
from lib.model_zoo.migan_inference import Generator  # noqa: E402

HERE = os.path.dirname(__file__)
IMG = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/DDColor/assets/test_images/Louis Armstrong practicing in his dressing room, ca 1946.jpg"


def main():
    torch.set_grad_enabled(False)
    res = 512
    gen = Generator(resolution=res).eval()
    sd = torch.load(f"{HERE}/migan/migan_512_places2.pt", map_location="cpu", weights_only=True)
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    bgr = cv2.resize(cv2.imread(IMG), (res, res))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = torch.from_numpy(rgb).permute(2, 0, 1)[None] * 2 / 255 - 1     # [-1,1] (1,3,res,res)
    keep = np.ones((res, res, 1), np.float32); keep[140:380, 90:290] = 0  # 0 = hole
    mask = torch.from_numpy(keep).permute(2, 0, 1)[None]                  # 1=keep
    x = torch.cat([mask - 0.5, img * mask], dim=1)                        # (1,4,res,res)

    out = gen(x)                                                          # (1,3,res,res) [-1,1]
    result = (out * 0.5 + 0.5).clamp(0, 1)
    composed = img * mask + (out) * (1 - mask)
    composed = ((composed * 0.5 + 0.5).clamp(0, 1)[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    g = f"{HERE}/goldens"; os.makedirs(g, exist_ok=True)
    np.save(f"{g}/migan_input4.npy", x.numpy())
    np.save(f"{g}/migan_output.npy", out.numpy())
    cv2.imwrite(f"{HERE}/migan_result.png", cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
    print(f"[done] out [{out.min():.3f},{out.max():.3f}] → goldens + migan_result.png")


if __name__ == "__main__":
    main()
