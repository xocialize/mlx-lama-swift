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

## License
Port code MIT. LaMa Apache-2.0; MI-GAN MIT. See NOTICE.
