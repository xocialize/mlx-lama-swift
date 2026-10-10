# mlx-lama-swift

Object removal / inpainting on Apple-Silicon MLX-Swift — from-scratch architecture ports of
**LaMa** (Apache-2.0, quality) and **MI-GAN** (MIT, fast/on-device), destined for an MLXEngine
`imageInpaint` ModelPackage backing the Forge **Erase** capability.

> **Status:** both models **ported + parity-locked** (LaMa 5.0e-6 on a colour fixture · MI-GAN 5.5e-5
> vs PyTorch) and end-to-end erase validated; conformant `MLXInpaint` ModelPackage (modes best=LaMa /
> fast=MI-GAN). **LaMa loads fp32 since 0.3.1** — see [LaMa precision](#lama-precision-fp32-only-2026-10-09).
> See [PORT-STATUS.md](PORT-STATUS.md).

## Layout
- `Sources/LaMa` — FFC ResNet generator (rFFT spectral block via MLX native FFT).
- `Sources/MIGAN` — mobile-GAN U-Net (separable convs, blur up/down-sample, noise).
- `Sources/MLXInpaint` — conformant `imageInpaint` ModelPackage (image + mask → filled image).
- `oracle/` — reproducible PyTorch parity harness (colour fixture, upstream anchor) + converters/publisher.

## Use
```swift
import LaMa   // or MIGAN
let inpainter = try LaMaInpainter.fromPretrained(weightsPath)   // LaMa: fp32 weights, fp32 load
let filled: CGImage = inpainter(sourceCGImage, mask: maskCGImage)   // white mask = remove
```

## Weights — exact mlx-community repo IDs

Copy these **verbatim** (all public, ungated). Note the **hyphen in `MI-GAN`** and the dtype suffixes —
a wrong name returns HTTP **401** from the Hub (it reports missing/misnamed repos as 401, not 404, for
unauthenticated clients), which can look like a gating/auth error but is just a typo.

| Tier (mode) | Repo ID | dtype |
|---|---|---|
| `best` — LaMa (quality) | `mlx-community/LaMa-fp32` | **fp32** (bf16 casts colour fills pink, fp16 collapses the FFC) |
| `fast` — MI-GAN 512 (default fast) | `mlx-community/MI-GAN-512-places2-fp16` | fp16 |
| MI-GAN 256 places2 | `mlx-community/MI-GAN-256-places2-fp16` | fp16 |
| MI-GAN 256 FFHQ (faces) | `mlx-community/MI-GAN-256-ffhq-fp16` | fp16 |

Collection: <https://huggingface.co/collections/mlx-community/inpainting-mlx-6a3bfadea8702ef69898d2ee>.
These are the defaults baked into `InpaintConfiguration` — consume that rather than hand-typing IDs.

## LaMa precision: fp32 only (2026-10-09)

Through 0.3.0 LaMa shipped as `mlx-community/LaMa-bf16` and loaded bf16. It filled large holes in
colour photos with a pink or magenta blur (AB-T-0213). The cause was the weight file, not the port.
Big-LaMa is unusually sensitive to weight rounding. bf16 rounds each weight by up to ~0.4%, and
that error compounds through 18 residual FFC blocks whose activations reach ~1e4 (model.24's output
peaks at 8.8e3). The hole is synthesised from those features alone, so it absorbs the error, while
the known pixels are composited back unchanged. Rounding only the BN running statistics to bf16
already gives max_abs 0.44; rounding only the conv weights gives 0.13. Loading the bf16 file
"as fp32" changes nothing, since the precision is already gone from the file. fp16 is worse and
collapses the FFC.

The parity gate missed it because its fixture was a greyscale photo. With R = G = B, a fault that
depends on colour can't show. The gate fixture is now a colour scene (`oracle/run_oracle.py`).

| LaMa, CPU vs PyTorch | bf16 file | fp32 file |
|---|---|---|
| colour gate (procedural 512², 29% hole), predicted max_abs | 7.5e-2 ❌ | **5.0e-6** ✅ |
| banana on a wooden table (512², 65% hole), predicted max_abs | 3.9e-1 | 1.1e-5 |
| same, GPU erase path, 8-bit hole error max / mean | 98 / 15.5 levels (R) | **1 / 0.001** |

- The oracle (`oracle/lama_gen.py`, a trimmed copy of advimman/lama) matches the original code's
  output within 1 level: `oracle/check_upstream.py` against Carve/LaMa-ONNX's `output_orig.png`.
- **Model limit, not a port fault:** on very large box holes Big-LaMa itself fades toward a grey,
  blurry centre (for example a 680×820 box at 1024², or a hole over 65% of the frame). PyTorch, the
  original code and this port agree on it. Hand that case to a generative fill.
- Gate: `python oracle/run_oracle.py` then
  `lama-smoke --weights lama_fp32.safetensors --parity oracle/goldens/colour.safetensors`. The smoke
  prints the stored dtype of the weight file.

## GPU numerics: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

Big-LaMa's 108 FFC convs fall inside it once (H/8)·(W/8) ≥ 4096, roughly 512² and up: convl2l
128→128, convg2l 384→128 and convl2g 128→384, across 18 ResBlocks. Weights and activations are
fp32. `lama-smoke` parity ran on the CPU with fp32 weights, so the
shipped GPU lane was never checked. MI-GAN (the fast tier) is unaffected: it is depthwise and 1×1.

`LaMaModel` / `LaMaInpainter.convRoute` now route the in-window convs through conv3d with kT = 1.
**Default `.conv3d`.**

Measurements: GPU against the CPU lane, inside the hole.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| 1024×681 whole-object erase (~18% hole) | 3.4e-3 · **max 9 levels** · hole PSNR 53.1 dB | 1 level max |
| 1024×680, elliptical hole (gate case) | 1.5e-3 · max 3 levels · 56.1 dB | 1.5e-6 · 1 level · 85.7 dB |
| Forward time, 1024×680 | 162 ms | +8 ms |

- Moving only the in-window convs to the CPU, or setting `MLX_ENABLE_TF32=0`, removes all of the
  drift. The Winograd convs are the whole cause.
- Environment override: `LAMA_CONV_ROUTE=winograd`.
- Gate: `LAMA_LANE=1 swift test -c release -Xswiftc -enable-testing --filter LaMaGPULaneTests`.

## License
Port code MIT. LaMa Apache-2.0; MI-GAN MIT. See NOTICE.
