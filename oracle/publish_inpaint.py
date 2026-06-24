"""Publish LaMa + MI-GAN fp16 inpainting weights to mlx-community + group in a Collection."""
import os
from huggingface_hub import HfApi, create_collection, add_collection_item

API = HfApi(); HERE = os.path.dirname(__file__)

REPOS = [
    # (local subdir, repo, base_model, license, pipeline blurb)
    ("lama", "LaMa-fp16", "advimman/lama", "apache-2.0",
     "Big-LaMa FFC inpainting (quality tier). Large masks + structured backgrounds."),
    ("migan-512", "MI-GAN-512-places2-fp16", "Picsart-AI-Research/MI-GAN", "mit",
     "MI-GAN mobile inpainting, 512 Places2 (fast/on-device tier)."),
    ("migan-256-places2", "MI-GAN-256-places2-fp16", "Picsart-AI-Research/MI-GAN", "mit",
     "MI-GAN mobile inpainting, 256 Places2."),
    ("migan-256-ffhq", "MI-GAN-256-ffhq-fp16", "Picsart-AI-Research/MI-GAN", "mit",
     "MI-GAN mobile inpainting, 256 FFHQ (faces)."),
]


def card(repo, base, lic, blurb):
    return f"""---
library_name: mlx
license: {lic}
base_model: {base}
pipeline_tag: image-to-image
tags:
  - mlx
  - inpainting
  - object-removal
---

# mlx-community/{repo}

{blurb}

Converted to **Apple MLX** (`-fp16`) for Apple-Silicon inference via the
[`mlx-lama-swift`](https://github.com/xocialize/mlx-lama-swift) Swift package (the MLXEngine
`imageInpaint` ModelPackage / Forge **Erase** capability). From-scratch MLX-Swift architecture port
of [{base}](https://github.com/{base}); parity-locked vs the PyTorch oracle on the CPU stream
(LaMa predicted max_abs 3.2e-5 · MI-GAN 5.5e-5).

## Use

```swift
// Package.swift → .package(url: "https://github.com/xocialize/mlx-lama-swift", from: "0.1.0")
import LaMa   // or MIGAN
let inpainter = try LaMaInpainter.fromPretrained(weightsPath, dtype: .float16)
let filled: CGImage = inpainter(sourceCGImage, mask: maskCGImage)  // white mask = remove
```

Input: image + mask (white = region to remove). Output: filled image at source resolution.
Weights license: {lic}. Port code: MIT.
"""


def main():
    published = []
    for sub, repo, base, lic, blurb in REPOS:
        rid = f"mlx-community/{repo}"
        wpath = f"{HERE}/publish/{sub}/model.safetensors"
        assert os.path.exists(wpath), wpath
        print(f"[publish] {rid} …")
        API.create_repo(rid, repo_type="model", exist_ok=True)
        API.upload_file(path_or_fileobj=wpath, path_in_repo="model.safetensors", repo_id=rid)
        API.upload_file(path_or_fileobj=card(repo, base, lic, blurb).encode(),
                        path_in_repo="README.md", repo_id=rid)
        published.append((rid, blurb))
        print(f"[publish]   → https://huggingface.co/{rid}")

    try:
        col = create_collection(title="Inpainting (MLX)", namespace="mlx-community",
                                description="Apple-MLX fp16 inpainting / object-removal models "
                                            "(LaMa Apache-2.0 + MI-GAN MIT). Loaded by mlx-lama-swift.")
        for rid, note in published:
            add_collection_item(col.slug, item_id=rid, item_type="model", note=note[:500])
        print(f"[collection] → https://huggingface.co/collections/{col.slug}")
    except Exception as e:
        print("[collection] note:", str(e)[:120])


if __name__ == "__main__":
    main()
