"""Load Big-LaMa weights into the standalone generator, inpaint a real masked image, dump goldens.

LaMa pipeline: input = cat[img*(1-mask), mask] (4ch) → generator → predicted (3ch, sigmoid) →
result = mask*predicted + (1-mask)*img. Dumps the 4ch input + predicted + result for MLX parity.
"""
import os
import sys
import types
import numpy as np
import cv2
import torch

# Stub pytorch_lightning so the Lightning checkpoint unpickles without the heavy framework —
# we only read ckpt["state_dict"] (plain tensors); the callback objects are reconstructed as dummies.
def _stub(name):
    m = types.ModuleType(name); sys.modules[name] = m; return m
_pl = _stub("pytorch_lightning")
_cb = _stub("pytorch_lightning.callbacks")
_mc = _stub("pytorch_lightning.callbacks.model_checkpoint")
class _Dummy:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
for mod in (_pl, _cb, _mc):
    mod.ModelCheckpoint = _Dummy
    mod.Callback = _Dummy

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from lama_gen import FFCResNetGenerator  # noqa: E402

DEF_IMG = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/DDColor/assets/test_images/Detroit circa 1915.jpg"


def main():
    torch.set_grad_enabled(False)
    gen = FFCResNetGenerator().eval()
    ckpt = torch.load(os.path.join(HERE, "big-lama/models/best.ckpt"), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    gsd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    missing, unexpected = gen.load_state_dict(gsd, strict=False)
    print(f"[load] generator keys: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:6]:", missing[:6])

    S = 512
    bgr = cv2.resize(cv2.imread(DEF_IMG), (S, S))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # HWC [0,1]
    mask = np.zeros((S, S), np.float32)
    mask[180:330, 200:360] = 1.0    # rectangular hole to fill

    img = torch.from_numpy(rgb.transpose(2, 0, 1))[None]       # (1,3,S,S)
    m = torch.from_numpy(mask)[None, None]                     # (1,1,S,S)
    inp = torch.cat([img * (1 - m), m], dim=1)                 # (1,4,S,S)

    # intermediate golden: output of ConcatTupleLayer (model[23]) — isolates the FFC stack from upsample
    bottleneck = {}
    gen.model[23].register_forward_hook(lambda _m, _i, o: bottleneck.update(v=o.detach().numpy()))

    predicted = gen(inp)                                       # (1,3,S,S) sigmoid
    result = m * predicted + (1 - m) * img
    np.save(os.path.join(HERE, "goldens", "bottleneck.npy"), bottleneck["v"])  # (1,512,S/8,S/8)

    out = os.path.join(HERE, "goldens"); os.makedirs(out, exist_ok=True)
    np.save(f"{out}/input4.npy", inp.numpy())
    np.save(f"{out}/predicted.npy", predicted.numpy())
    np.save(f"{out}/result.npy", result.numpy())

    # visuals
    def png(t, p):
        a = (t[0].numpy().transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)
        cv2.imwrite(p, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    png(img * (1 - m) + m, f"{HERE}/lama_masked.png")
    png(result, f"{HERE}/lama_result.png")
    print(f"[done] predicted [{predicted.min():.3f},{predicted.max():.3f}]  → goldens + lama_result.png")


if __name__ == "__main__":
    main()
