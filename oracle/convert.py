"""Big-LaMa PyTorch generator → canonical MLX safetensors (NHWC).

Conv weights (O,I,kH,kW) → (O,kH,kW,I); ConvTranspose (I,O,kH,kW) → (O,kH,kW,I); drop
num_batches_tracked; keep BN params. Flat torch keys (Swift WeightLoading maps them).

    python convert.py --dtype float32 --out weights/lama_fp32.safetensors           # parity
    python convert.py --dtype float32 --out publish/lama-fp32/model.safetensors     # publish

Publish fp32 only. fp16 collapses the FFC, and bf16 casts large colour fills pink: its weight
rounding compounds through the 18 FFC blocks (AB-T-0213). The parity fixture comes from
run_oracle.py.
"""
import argparse
import os
import numpy as np
import torch
import mlx.core as mx

from ckpt import generator_state_dict

HERE = os.path.dirname(__file__)

CONV_T = {"model.24.weight", "model.27.weight", "model.30.weight"}  # ConvTranspose2d


def load_gen_sd():
    return {k: v.float().numpy() for k, v in generator_state_dict(f"{HERE}/big-lama/models/best.ckpt").items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--out", default=f"{HERE}/weights/lama_fp32.safetensors")
    args = ap.parse_args()
    dt = mx.float16 if args.dtype == "float16" else mx.float32
    sd = load_gen_sd()

    out = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        if k.endswith(".weight") and v.ndim == 4:
            if k in CONV_T:                       # ConvTranspose (I,O,kH,kW) -> (O,kH,kW,I)
                v = np.transpose(v, (1, 2, 3, 0))
            else:                                 # Conv (O,I,kH,kW) -> (O,kH,kW,I)
                v = np.transpose(v, (0, 2, 3, 1))
        out[k] = mx.array(v.astype(np.float32)).astype(dt)

    mx.eval(list(out.values()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mx.save_safetensors(args.out, out)
    print(f"[convert] {len(out)} tensors ({args.dtype}) -> {args.out}")


if __name__ == "__main__":
    main()
