# mlx-lama-swift — Port Status (Erase / inpainting)

LaMa (advimman/lama, **Apache-2.0**) + MI-GAN (Picsart-AI-Research, **MIT**) → MLX-Swift, for an
MLXEngine `imageInpaint` ModelPackage feeding the Forge **Erase** capability. From-scratch
architecture ports (no MLX donors). See ERASE-PLAN.md (Forge side).

## ✅ DONE & VERIFIED — LaMa (primary / quality tier)

| Step | Result |
|---|---|
| FFC de-risk | `oracle/ffc_derisk.py` — FourierUnit (rFFT→1×1→irFFT, ortho) NHWC vs PyTorch **1.9e-6** ✅ (MLX native FFT) |
| Reference + config | advimman/lama; Big-LaMa config (ffc_resnet, 4→3, ngf64, 18 FFC res-blocks, sigmoid out, ratios 0/0.75); weights smartywu/big-lama |
| Full MLX-Python parity | `oracle/mlx_lama.py` — predicted vs PyTorch golden **3.17e-5** (CPU fp32) ✅ |
| Weight conversion | `oracle/convert.py` — NHWC conv transpose, ConvTranspose (I,O)→(O,..,I), drop num_batches; flat keys |
| Swift core | `Sources/LaMa/` — `LaMaModel`/`LaMaOps`/`LaMaImage`/`LaMaInpainter`; reflect-pad via gather, ortho FFT via backward×√N |
| **Swift parity gate** | `lama-smoke`: predicted **max_abs 3.17e-5** (mean 2.3e-7) vs golden (CPU fp32) ✅ — but on a greyscale fixture |
| **Colour parity gate** (AB-T-0213, 2026-10-09) | `oracle/run_oracle.py` procedural colour scene, 29% hole: fp32 weights **5.0e-6** ✅ · the published bf16 weights **7.5e-2** ❌ (pink fills). LaMa now ships and loads fp32 |
| **Upstream anchor** | `oracle/check_upstream.py`: the trimmed oracle vs the original advimman/lama output (Carve/LaMa-ONNX `output_orig.png`): hole max 1 level ✅ |
| **End-to-end erase** | `lama-smoke --erase-image` → real object removal validated visually (clean fill, ~0.5s @880², peak 2.8 GB) ✅ |

## ✅ DONE & VERIFIED — MI-GAN (fast / on-device tier)

| Step | Result |
|---|---|
| Reference + weights | Picsart-AI-Research/MI-GAN `migan_inference.py` (torch+numpy only); official weights via gdown (256_ffhq / 256_places2 / 512_places2) |
| Oracle + I/O | `oracle/run_migan.py` — load missing=0; img∈[-1,1], mask **1=keep** (opposite LaMa), input=`cat[mask-0.5, img*mask]`, out∈[-1,1] |
| MLX-Python parity | `oracle/mlx_migan.py` — output **5.47e-5** vs PyTorch (separable conv, blur up/down + filter_const + asym pad, lrelu+gain/clamp, noise) ✅ |
| **Swift core + parity** | `Sources/MIGAN/` (resolution-parametric); `migan-smoke` **max_abs 5.47e-5** ✅; `MIGANInpainter` (unified white=remove mask) |

## ✅ DONE — engine integration + publish (task 13)

| Step | Result |
|---|---|
| **imageInpaint contract** | MLXToolKit 1.8.0 (`8cd0033`): `Capability.imageInpaint` + `Inpaint.swift` (Request **image+mask** / Response / Contract; best/fast modes) — the contract's first two-input surface |
| **MLXInpaint ModelPackage** | `Sources/MLXInpaint/` — one package, `best`→LaMa / `fast`→MI-GAN; Apache/MIT gate; fp16 4.5 GB footprint; builds against pinned contract ✅ |
| **Published weights** | `mlx-community/{LaMa-fp32, MI-GAN-512-places2-fp16, MI-GAN-256-places2-fp16, MI-GAN-256-ffhq-fp16}` + "Inpainting (MLX)" Collection ✅ — `LaMa-bf16` (≤ 0.3.0) is superseded: it casts colour fills pink |
| **Published code** | `github.com/xocialize/mlx-lama-swift` (see tag) ✅ |

## ▢ REMAINING → Forge agent
Forge Phase 2+ (EraseKit/UI mask-paint inspector, promptable EdgeTAM/SAM, video per-frame+SEA-RAFT).

## Notes
- MLX has no reflect-pad / flip → reflect via index-gather (`LaMaOps.reflectPad`).
- MLX-Swift FFT wrapper lacks `norm` → ortho via backward primitives ×(1/√N fwd, √N inv) (verified).
- LaMa bottleneck features reach ~1136 magnitude → deep-intermediate deltas are fp32 noise; gate on final predicted (3e-5).
