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
let inpainter = try LaMaInpainter.fromPretrained(weightsPath, dtype: .float16)
let filled: CGImage = inpainter(sourceCGImage, mask: maskCGImage)   // white mask = remove
```
Weights: `mlx-community/{LaMa,MI-GAN-*}-fp16` ([collection](https://huggingface.co/collections/mlx-community/inpainting-mlx)).

## License
Port code MIT. LaMa Apache-2.0; MI-GAN MIT. See NOTICE.
