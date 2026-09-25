// Route for 3×3 convs inside mlx's Winograd conv2d window (mlx-swift ≤ 0.31.6 Metal numerics).
//
// mlx's Metal conv2d (mlx/backend/metal/conv.cpp `dispatch_conv_2D_gpu`) runs a Winograd
// F(6×6,3×3) kernel when ALL of these hold: kernel 3×3, stride 1, dilation 1, groups 1,
// C % 32 == 0, O % 32 == 0, C + O ≥ 256, N·H·W ≥ 4096. On M5 that path is lossy — ~6.4e-3 relL2
// per conv in fp32 (its batched GEMM runs TF32, MLX_ENABLE_TF32 defaults on, and the output
// transform amplifies that ~8×). Every other conv path is exact-class; conv3d with kT = 1 is the
// same conv on the implicit-GEMM path — exact.
//
// Big-LaMa hits the window in all 108 FFC convs of its 18 ResBlocks (convl2l 128→128, convg2l
// 384→128, convl2g 128→384 at H/8, reflect-padded) once (H/8)·(W/8) ≥ 4096 — ≈512² and up. The
// bf16 weights meet fp32 activations, so they compute in fp32. Measured GPU vs CPU (1024×681,
// ~18% hole): 8-bit max 9 levels, hole PSNR 53 dB — Winograd is the whole drift. Default `.conv3d`.
// `LAMA_CONV_ROUTE=winograd` restores raw conv2d (validation). Removal: when a new mlx-swift pin
// reports raw conv2d exact (LaMaGPULaneTests probe), drop the route.

import Foundation
import MLX

public enum LaMaConvRoute: String, Sendable {
    /// mlx's default Winograd kernel — fastest; lossy on M5 (~6.4e-3 relL2 per conv in fp32).
    case winograd
    /// conv3d with kT = 1 on the implicit-GEMM path — exact.
    case conv3d

    /// `LAMA_CONV_ROUTE` = winograd | conv3d, if set.
    public static var environmentOverride: LaMaConvRoute? {
        getenv("LAMA_CONV_ROUTE").flatMap { LaMaConvRoute(rawValue: String(cString: $0)) }
    }

    /// mlx's Winograd dispatch predicate, evaluated on the actual input (NHWC).
    static func takesWinograd(_ x: MLXArray, _ w: MLXArray, stride: Int) -> Bool {
        guard x.ndim == 4, stride == 1, w.dim(1) == 3, w.dim(2) == 3, w.dim(3) == x.dim(3)
        else { return false }
        let (c, o) = (x.dim(3), w.dim(0))
        return c % 32 == 0 && o % 32 == 0 && c + o >= 256 && x.dim(0) * x.dim(1) * x.dim(2) >= 4096
    }
}

/// `conv2d` that takes the conv3d route for in-window shapes when the route says so.
func routedConv2d(_ x: MLXArray, _ w: MLXArray, stride: Int, padding: Int, route: LaMaConvRoute)
    -> MLXArray
{
    guard route == .conv3d, LaMaConvRoute.takesWinograd(x, w, stride: stride) else {
        return MLX.conv2d(x, w, stride: .init(stride), padding: .init(padding))
    }
    return MLX.conv3d(x.expandedDimensions(axis: 1), w.expandedDimensions(axis: 1),
                      stride: [1, 1, 1], padding: [0, padding, padding]).squeezed(axis: 1)
}
