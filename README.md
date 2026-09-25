# mlx-lama-swift

Object removal / inpainting on Apple-Silicon MLX-Swift — from-scratch architecture ports of
**LaMa** (Apache-2.0, quality) and **MI-GAN** (MIT, fast/on-device), destined for an MLXEngine
`imageInpaint` ModelPackage backing the Forge **Erase** capability.

> **Status:** both models **ported + parity-locked** (LaMa 3.2e-5 · MI-GAN 5.5e-5 vs PyTorch) and
> end-to-end erase validated; conformant `MLXInpaint` ModelPackage (modes best=LaMa / fast=MI-GAN).
> See [PORT-STATUS.md](PORT-STATUS.md).

## Layout
- `Sources/LaMa` — FFC ResNet generator (rFFT spectral block via MLX native FFT).
- `Sources/MIGAN` — mobile-GAN U-Net (separable convs, blur up/down-sample, noise).
- `Sources/MLXInpaint` — conformant `imageInpaint` ModelPackage (image + mask → filled image).
- `oracle/` — reproducible PyTorch parity harness + fp16 converters/publisher.

## Use
```swift
import LaMa   // or MIGAN
let inpainter = try LaMaInpainter.fromPretrained(weightsPath, dtype: .bfloat16)  // LaMa: bf16, NOT fp16
let filled: CGImage = inpainter(sourceCGImage, mask: maskCGImage)   // white mask = remove
```

## Weights — exact mlx-community repo IDs

Copy these **verbatim** (all public, ungated). Note the **hyphen in `MI-GAN`** and the dtype suffixes —
a wrong name returns HTTP **401** from the Hub (it reports missing/misnamed repos as 401, not 404, for
unauthenticated clients), which can look like a gating/auth error but is just a typo.

| Tier (mode) | Repo ID | dtype |
|---|---|---|
| `best` — LaMa (quality) | `mlx-community/LaMa-bf16` | **bf16** (fp16 collapses the FFC → garbage) |
| `fast` — MI-GAN 512 (default fast) | `mlx-community/MI-GAN-512-places2-fp16` | fp16 |
| MI-GAN 256 places2 | `mlx-community/MI-GAN-256-places2-fp16` | fp16 |
| MI-GAN 256 FFHQ (faces) | `mlx-community/MI-GAN-256-ffhq-fp16` | fp16 |

Collection: <https://huggingface.co/collections/mlx-community/inpainting-mlx-6a3bfadea8702ef69898d2ee>.
These are the defaults baked into `InpaintConfiguration` — consume that rather than hand-typing IDs.

## GPU numerics: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

Big-LaMa's 108 FFC convs fall inside it once (H/8)·(W/8) ≥ 4096, roughly 512² and up: convl2l
128→128, convg2l 384→128 and convl2g 128→384, across 18 ResBlocks. The bf16 weights meet fp32
activations, so they compute in fp32. `lama-smoke` parity ran on the CPU with fp32 weights, so the
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
