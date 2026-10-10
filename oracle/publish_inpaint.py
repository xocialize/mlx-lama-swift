"""Publish LaMa + MI-GAN inpainting weights to mlx-community and add them to the Inpainting (MLX) Collection.

    python convert.py --dtype float32 --out publish/lama-fp32/model.safetensors
    python publish_inpaint.py LaMa-fp32            # one repo; no argument publishes them all

LaMa ships fp32 (AB-T-0213). The original `LaMa-bf16` cast large colour fills pink: bf16 weight
rounding compounds through the 18 FFC blocks. MI-GAN is well-scaled and stays fp16.
"""
import os
import sys

from huggingface_hub import HfApi, add_collection_item, create_collection

API = HfApi(); HERE = os.path.dirname(__file__)

REPOS = [
    # (local subdir, repo, base_model, license, pipeline blurb)
    ("lama-fp32", "LaMa-fp32", "advimman/lama", "apache-2.0",
     "Big-LaMa FFC inpainting (quality tier). Large masks + structured backgrounds."),
    ("migan-512", "MI-GAN-512-places2-fp16", "Picsart-AI-Research/MI-GAN", "mit",
     "MI-GAN mobile inpainting, 512 Places2 (fast/on-device tier)."),
    ("migan-256-places2", "MI-GAN-256-places2-fp16", "Picsart-AI-Research/MI-GAN", "mit",
     "MI-GAN mobile inpainting, 256 Places2."),
    ("migan-256-ffhq", "MI-GAN-256-ffhq-fp16", "Picsart-AI-Research/MI-GAN", "mit",
     "MI-GAN mobile inpainting, 256 FFHQ (faces)."),
]

LAMA_NOTE = """
## Precision: fp32 only

LaMa does not survive reduced-precision weights. bf16 rounds each weight by up to ~0.4%, and the
error compounds through 18 residual FFC blocks whose activations reach ~1e4. The earlier
`mlx-community/LaMa-bf16` cast large fills in colour photos pink: max_abs 7.5e-2 against PyTorch on
a colour fixture, where this fp32 file reads 5.0e-6. fp16 collapses the FFC outright. Casting bf16 weights up to fp32 at load does not help,
because the file has already lost the precision. Load these weights fp32 and keep them fp32.

Parity: lama-smoke predicted max_abs 5.0e-6 against the PyTorch oracle on a colour 512² fixture with a
29% hole (CPU). The oracle matches the original advimman/lama output within 1 level. The GPU erase
path matches PyTorch within 1 level in every hole pixel.
"""


def card(repo, base, lic, blurb):
    lama = repo.startswith("LaMa")
    dtype = "fp32" if lama else "fp16"
    load = (".float32" if lama else ".float16")
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

Converted to **Apple MLX** (`{dtype}`) for Apple-Silicon inference via the
[`mlx-lama-swift`](https://github.com/xocialize/mlx-lama-swift) Swift package (the MLXEngine
`imageInpaint` ModelPackage). From-scratch MLX-Swift architecture port of
[{base}](https://github.com/{base}), parity-locked against the PyTorch oracle on the CPU stream.
{LAMA_NOTE if lama else ""}
## Use

```swift
// Package.swift → .package(url: "https://github.com/xocialize/mlx-lama-swift", from: "0.3.1")
import LaMa   // or MIGAN
let inpainter = try LaMaInpainter.fromPretrained(weightsPath, dtype: {load})
let filled: CGImage = inpainter(sourceCGImage, mask: maskCGImage)  // white mask = remove
```

Input: image + mask (white = region to remove). Output: filled image at source resolution.
Weights license: {lic}. Port code: MIT.
"""


def main():
    only = set(sys.argv[1:])
    published = []
    for sub, repo, base, lic, blurb in REPOS:
        if only and repo not in only:
            continue
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
        col = create_collection(title="Inpainting (MLX)", namespace="mlx-community", exists_ok=True,
                                description="Apple-MLX inpainting / object-removal models "
                                            "(LaMa Apache-2.0 + MI-GAN MIT). Loaded by mlx-lama-swift.")
        for rid, note in published:
            add_collection_item(col.slug, item_id=rid, item_type="model", note=note[:500], exists_ok=True)
        print(f"[collection] → https://huggingface.co/collections/{col.slug}")
    except Exception as e:
        print("[collection] note:", str(e)[:120])


if __name__ == "__main__":
    main()
