import Foundation
import MLX
import MLXFFT

/// NHWC ops for the Big-LaMa FFC generator, transcribed 1:1 from the parity-verified
/// `oracle/mlx_lama.py` (predicted max_abs 3.2e-5 vs PyTorch, CPU fp32).
enum LaMaOps {

    /// Reflection padding over H,W (no edge repeat) — MLX has no reflect mode, so build via gather.
    static func reflectPad(_ x: MLXArray, _ p: Int) -> MLXArray {
        if p == 0 { return x }
        func idx(_ n: Int) -> MLXArray {
            var ids = [Int32]()
            ids.append(contentsOf: stride(from: p, to: 0, by: -1).map { Int32($0) })
            ids.append(contentsOf: (0 ..< n).map { Int32($0) })
            ids.append(contentsOf: stride(from: n - 2, to: n - 2 - p, by: -1).map { Int32($0) })
            return MLXArray(ids)
        }
        var y = MLX.take(x, idx(x.dim(1)), axis: 1)
        y = MLX.take(y, idx(x.dim(2)), axis: 2)
        return y
    }

    static func bn(_ x: MLXArray, _ mean: MLXArray, _ varr: MLXArray,
                   _ w: MLXArray, _ b: MLXArray, eps: Float = 1e-5) -> MLXArray {
        (x - mean) / MLX.sqrt(varr + eps) * w + b
    }

    static func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) }

    /// FFC FourierUnit. MLX rFFT defaults to 'backward' norm; LaMa uses 'ortho' → scale by 1/√N
    /// on the forward and ×√N on the inverse (verified equivalent to norm='ortho').
    static func fourierUnit(_ x: MLXArray, convW: MLXArray,
                            bnMean: MLXArray, bnVar: MLXArray, bnW: MLXArray, bnB: MLXArray) -> MLXArray {
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let n = Float(H * W).squareRoot()
        let ft = MLXFFT.rfftn(x, axes: [1, 2]) / n          // ortho forward
        let w2 = ft.dim(2)
        let inter = MLX.stacked([ft.realPart(), ft.imaginaryPart()], axis: -1)
            .reshaped([B, H, w2, 2 * C])                    // [c0r,c0i,c1r,c1i,...]
        var y = MLX.conv2d(inter, convW, stride: 1, padding: 0)
        y = relu(bn(y, bnMean, bnVar, bnW, bnB))
        let yc = y.reshaped([B, H, w2, C, 2])
        let comp = yc[0..., 0..., 0..., 0..., 0] + yc[0..., 0..., 0..., 0..., 1].asImaginary()
        return MLXFFT.irfftn(comp, s: [H, W], axes: [1, 2]) * n   // ortho inverse
    }
}
