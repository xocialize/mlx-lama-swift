import Foundation
import MLX

/// Big-LaMa FFCResNetGenerator forward over a flat weights dict (canonical NHWC keys from
/// `oracle/convert.py`). Functional style, transcribed 1:1 from the parity-verified `oracle/mlx_lama.py`.
///
/// Input: 4-channel NHWC `(1,H,W,4)` = `concat[image*(1-mask), mask]` (H,W divisible by 8).
/// Output: predicted RGB `(1,H,W,3)` in `[0,1]` (sigmoid). Compositing with the source is the caller's.
public final class LaMaModel: @unchecked Sendable {

    private let w: [String: MLXArray]
    public init(weights: [String: MLXArray]) { self.w = weights }

    private func a(_ k: String) -> MLXArray { w[k]! }                 // weight (convs already NHWC)
    private func conv(_ x: MLXArray, _ k: String, b: String? = nil, stride: Int = 1, pad: Int = 0) -> MLXArray {
        let y = MLX.conv2d(x, a(k), stride: .init(stride), padding: .init(pad))
        return b == nil ? y : y + a(b!)
    }
    private func bn(_ x: MLXArray, _ p: String) -> MLXArray {
        LaMaOps.bn(x, a(p + ".running_mean"), a(p + ".running_var"), a(p + ".weight"), a(p + ".bias"))
    }
    private func relu(_ x: MLXArray) -> MLXArray { LaMaOps.relu(x) }

    private func fourierUnit(_ x: MLXArray, _ p: String) -> MLXArray {
        LaMaOps.fourierUnit(x, convW: a(p + ".conv_layer.weight"),
                            bnMean: a(p + ".bn.running_mean"), bnVar: a(p + ".bn.running_var"),
                            bnW: a(p + ".bn.weight"), bnB: a(p + ".bn.bias"))
    }

    private func spectralTransform(_ xg: MLXArray, _ p: String) -> MLXArray {
        var x = relu(bn(conv(xg, p + ".conv1.0.weight"), p + ".conv1.1"))
        let out = fourierUnit(x, p + ".fu")
        return conv(x + out, p + ".conv2.weight")
    }

    /// One FFC_BN_ACT. Returns (local, global) — either may be nil when its ratio is 0.
    private func ffcBnAct(_ xl: MLXArray, _ xg: MLXArray?, _ p: String,
                          gin: Float, gout: Float, k: Int, stride: Int) -> (MLXArray?, MLXArray?) {
        let pad = (k - 1) / 2
        var outL: MLXArray?, outG: MLXArray?
        if gout != 1 {
            outL = conv(LaMaOps.reflectPad(xl, pad), p + ".ffc.convl2l.weight", stride: stride)
            if gin > 0, let xg { outL = outL! + conv(LaMaOps.reflectPad(xg, pad), p + ".ffc.convg2l.weight", stride: stride) }
        }
        if gout != 0 {
            outG = conv(LaMaOps.reflectPad(xl, pad), p + ".ffc.convl2g.weight", stride: stride)
            if gin > 0, let xg { outG = outG! + spectralTransform(xg, p + ".ffc.convg2g") }
        }
        let rl = outL.map { relu(bn($0, p + ".bn_l")) }
        let rg = outG.map { relu(bn($0, p + ".bn_g")) }
        return (rl, rg)
    }

    private func resblock(_ xl: MLXArray, _ xg: MLXArray, _ p: String) -> (MLXArray, MLXArray) {
        var (l, g) = ffcBnAct(xl, xg, p + ".conv1", gin: 0.75, gout: 0.75, k: 3, stride: 1)
        (l, g) = ffcBnAct(l!, g, p + ".conv2", gin: 0.75, gout: 0.75, k: 3, stride: 1)
        return (xl + l!, xg + g!)
    }

    private func convT(_ x: MLXArray, _ p: String) -> MLXArray {
        MLX.convTransposed2d(x, a(p + ".weight"), stride: 2, padding: 1, outputPadding: 1) + a(p + ".bias")
    }

    /// Full generator forward → predicted RGB `(1,H,W,3)` in `[0,1]`.
    public func callAsFunction(_ input4: MLXArray) -> MLXArray {
        // init conv (ReflectionPad2d(3) + FFC k7, all-local)
        var x = relu(bn(conv(LaMaOps.reflectPad(input4, 3), "model.1.ffc.convl2l.weight"), "model.1.bn_l"))
        var xl: MLXArray? = x
        var xg: MLXArray? = nil
        (xl, xg) = ffcBnAct(xl!, xg, "model.2", gin: 0, gout: 0, k: 3, stride: 2)
        (xl, xg) = ffcBnAct(xl!, xg, "model.3", gin: 0, gout: 0, k: 3, stride: 2)
        (xl, xg) = ffcBnAct(xl!, xg, "model.4", gin: 0, gout: 0.75, k: 3, stride: 2)
        for i in 5 ... 22 { (xl, xg) = resblock(xl!, xg!, "model.\(i)") }
        x = MLX.concatenated([xl!, xg!], axis: -1)           // ConcatTupleLayer → 512ch
        // upsample (ConvTranspose s2 + BN + ReLU) × 3
        x = relu(bn(convT(x, "model.24"), "model.25"))
        x = relu(bn(convT(x, "model.27"), "model.28"))
        x = relu(bn(convT(x, "model.30"), "model.31"))
        // head: ReflectionPad2d(3) + Conv k7 + sigmoid
        x = conv(LaMaOps.reflectPad(x, 3), "model.34.weight", b: "model.34.bias")
        return MLX.sigmoid(x)
    }
}
