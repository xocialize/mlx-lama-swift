import Foundation
import MLX

/// MI-GAN generator forward (NHWC), resolution-parametric, over a flat weights dict (canonical NHWC
/// keys from convert). Transcribed 1:1 from the parity-verified `oracle/mlx_migan.py` (max_abs 5.5e-5).
///
/// Input 4ch NHWC = `concat[mask-0.5, img*mask]` (img∈[-1,1], mask 1=keep/0=hole).
/// Output 3ch NHWC in [-1,1]. Compositing is the caller's.
public final class MIGANModel: @unchecked Sendable {

    public let resolution: Int
    private let w: [String: MLXArray]
    public init(resolution: Int, weights: [String: MLXArray]) {
        self.resolution = resolution; self.w = weights
    }

    private func a(_ k: String) -> MLXArray { w[k]! }
    private static let sqrt2: Float = Float(2).squareRoot()

    private func lrelu(_ x: MLXArray) -> MLXArray {  // lrelu_agc(0.2, gain√2, clamp256)
        MLX.clip(MLX.where(x .>= 0, x, 0.2 * x) * Self.sqrt2, min: -256, max: 256)
    }
    private func conv(_ x: MLXArray, _ k: String, b: String? = nil, stride: Int = 1, pad: Int = 0, groups: Int = 1) -> MLXArray {
        let y = MLX.conv2d(x, a(k), stride: .init(stride), padding: .init(pad), groups: groups)
        return b == nil ? y : y + a(b!)
    }
    private func downsample(_ x: MLXArray, _ p: String) -> MLXArray {
        conv(x, p + ".filter.weight", stride: 2, pad: 1, groups: x.dim(-1))
    }
    private func nearest2x(_ x: MLXArray) -> MLXArray {
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return MLX.broadcast(x.reshaped([B, H, 1, W, 1, C]), to: [B, H, 2, W, 2, C]).reshaped([B, 2 * H, 2 * W, C])
    }
    private func upsample(_ x0: MLXArray, _ p: String) -> MLXArray {
        var x = nearest2x(x0)
        x = x * a(p + ".filter_const")                                   // (1,res,res,1) broadcast
        x = MLX.padded(x, widths: [.init(0), .init((2, 1)), .init((2, 1)), .init(0)])
        return conv(x, p + ".filter.weight", groups: x.dim(-1))          // depthwise blur k4
    }

    private func separable(_ x0: MLXArray, _ p: String, act: Bool = true,
                           down: Bool = false, up: Bool = false, noise: Bool = false) -> MLXArray {
        var x = conv(x0, p + ".conv1.weight", b: p + ".conv1.bias", pad: 1, groups: x0.dim(-1))  // depthwise k3
        if act { x = lrelu(x) }
        if down { x = downsample(x, p + ".downsample") }
        x = conv(x, p + ".conv2.weight")                                 // pointwise 1x1
        if up { x = upsample(x, p + ".upsample") }
        if noise {
            x = x + a(p + ".noise_const").reshaped([1, x.dim(1), x.dim(2), 1]) * a(p + ".noise_strength")
        }
        if act { x = lrelu(x) }
        return x
    }

    private func encRes() -> [Int] {                                     // [R, R/2, ..., 8]
        var r = resolution, out: [Int] = []
        while r > 4 { out.append(r); r /= 2 }
        return out
    }

    /// Full forward → predicted RGB `(1,res,res,3)` in [-1,1].
    public func callAsFunction(_ input4: MLXArray) -> MLXArray {
        // encoder
        var feats: [Int: MLXArray] = [:]
        var x: MLXArray? = nil
        for resi in encRes() {
            let p = "encoder.b\(resi)"
            if resi == resolution {
                let y = lrelu(conv(input4, p + ".fromrgb.weight", b: p + ".fromrgb.bias"))
                x = x == nil ? y : x! + y
            }
            let feat = separable(x!, p + ".conv1")
            x = separable(feat, p + ".conv2", down: true)
            feats[resi] = feat
        }
        let feat4 = separable(x!, "encoder.b4.conv1")
        x = separable(feat4, "encoder.b4.conv2", down: false)
        feats[4] = feat4
        // synthesis
        var s = separable(x!, "synthesis.b4.conv1")
        s = s + feats[4]!
        s = separable(s, "synthesis.b4.conv2")
        var img = conv(s, "synthesis.b4.torgb.weight", b: "synthesis.b4.torgb.bias")
        var resj = 8
        while resj <= resolution {
            let p = "synthesis.b\(resj)"
            s = separable(s, p + ".conv1", up: true, noise: true)
            s = s + feats[resj]!
            s = separable(s, p + ".conv2", noise: true)
            img = upsample(img, p + ".upsample")
            img = img + conv(s, p + ".torgb.weight", b: p + ".torgb.bias")
            resj *= 2
        }
        return img
    }
}
