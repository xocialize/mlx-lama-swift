# Efficiency Adoption Brief — `mlx-lama-swift` (LaMa + MI-GAN, `imageInpaint`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**) + references/memory-harness.md. LIGHT **split + unload-clearCache** adoption. Audited 2026-06-30.

## Package at a glance
- Three products: `LaMa` (FFC quality-tier core) + `MIGAN` (mobile-GAN fast-tier core) + `MLXInpaint`
  (`InpaintPackage: ModelPackage`). Capability **`imageInpaint`** (the first **2-input** surface: image + mask).
  Engine pinned `from: "0.10.0"`.
- Components (`InpaintPackage`): `lama: LaMaInpainter?` + `migan: MIGANInpainter?` (mode selects which runs).
- **Footprint today (FLAT residentBytes only, NO transient):** `QuantFootprint(.fp16, 4.5 GB)`.
- `unload()` is `lama = nil; migan = nil` — **no `MLX.Memory.clearCache()`**.

## ⚠️ Critical — do NOT "fix" the dtype
**LaMa is fp16-FATAL** (NaN/black output in fp16) and deliberately **ships/runs bf16**. The
`QuantFootprint(quant: .fp16, …)` is the engine **quant-tier label**, not a directive to load fp16. Do not
change the dtype/load path to fp16 — verify it stays bf16. (This is the package's hardest-won lesson.)

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.10.0 → 0.17.0 | **P0** |
| 1. Split footprint | ❌ | flat 4.5 GB, no transient | **P1 (headline)** |
| 2. Per-stage evict | 🟡 maybe | both `lama`+`migan` may load resident; if so, the unused tier could load-on-demand — but inpaint nets are small (the 4.5 GB is mostly FFC activation, not weights). Verify in `load()`; likely low value. | note |
| 3. mmap/lazy | 🟡 verify | confirm lazy weight load (floor ≈ on-disk) | note |
| 4. BudgetAware | ➖ | single quant tier | defer |

## Plan
- **P0:** `swift package update` → 0.17.0; build + fix any drift (imageInpaint 2-input surface stable; verify).
- **P1 (headline):** split the flat 4.5 GB. `residentBytes` = the resident inpainter weights floor (measure —
  check whether `load()` holds both cores or one); `peakActivationBytes` = the FFC inpaint transient (the FFC
  path's activation dominates the 4.5 GB). Adopt `QuantConfigured` (single tier — keep the bf16 reality).
- **P2:** only if `load()` holds BOTH cores resident AND they're non-trivial — load the selected tier
  on-demand and evict the other. Verify first; likely not worth it (small nets). Note the finding either way.
- **`unload()` must add `MLX.Memory.clearCache()`** after niling `lama`/`migan` (eviction-frees-RSS rule).

## Measurement — IMPORTANT (in-app phys + 2-input caveat)
Declare `residentBytes` from the measured weight floor (solid) + a **FLAGGED** `peakActivationBytes` from the
package smoke (in-app phys reads ~2.5–2.9× higher — admission basis). **2-input caveat:** imageInpaint needs
image + mask, and the image app's headless autorun can't read an arbitrary `IMAGE_IN` (sandbox) — so the
in-app phys re-baseline for LaMa is an **in-GUI** measure (note it for the Xcode side; declare from weight
floor + flag meanwhile).

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; P1 split; `unload()` clearCache; **dtype stays bf16** (not fp16).
- [ ] residentBytes = measured weight floor; peakActivationBytes = FFC transient (FLAGGED smoke est).
- [ ] Smoke green (valid inpaint over a masked region); split recorded; activation flagged.
- [ ] Registry: lama/inpaint row Eff ⬜→✅ (note "activation = smoke est, phys re-baseline pending"), Eng→0.17.0.

## Report back
flat→split, both-cores-or-one finding, the FFC transient (flagged), drift since 0.10.0, effort, commit SHA.
STAY IN SCOPE — four-lever adoption + this brief + registry row only; no testing-app/xcodeproj changes;
**do not touch the bf16 dtype path**; stop-and-report if bigger.
