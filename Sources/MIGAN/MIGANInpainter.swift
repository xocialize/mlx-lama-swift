import Foundation
import CoreGraphics
import MLX

/// End-to-end MI-GAN eraser: (source, mask) → inpainted CGImage. MI-GAN is fixed-resolution, so the
/// source is resized to `resolution`, filled, and the hole region is composited back at source size.
///
/// Mask convention unified with LaMa: **white = remove (hole)**. Internally MI-GAN uses keep=1-remove,
/// img∈[-1,1], input=concat[keep-0.5, img*keep], out∈[-1,1], result=img*keep + out*(1-keep).
public final class MIGANInpainter: @unchecked Sendable {

    private let model: MIGANModel
    public let resolution: Int
    public init(resolution: Int, weights: [String: MLXArray]) {
        self.resolution = resolution
        self.model = MIGANModel(resolution: resolution, weights: weights)
    }

    public static func fromPretrained(_ path: String, resolution: Int, dtype: DType = .float32) throws -> MIGANInpainter {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(dtype) }
        return MIGANInpainter(resolution: resolution, weights: w)
    }

    public func callAsFunction(_ source: CGImage, mask: CGImage) -> CGImage {
        let R = resolution
        let rgb = MIGANImage.rgb(from: source, width: R, height: R)         // (R,R,3) [0,1]
        var remove = MIGANImage.gray(from: mask, width: R, height: R)       // white = remove
        remove = (remove .> 0.5)
        let keep = 1 - remove
        let img = rgb * 2 - 1                                               // [-1,1]
        let input4 = MLX.concatenated([keep - 0.5, img * keep], axis: -1).expandedDimensions(axis: 0)
        let out = model(input4)[0]                                         // (R,R,3) [-1,1]
        let out01 = MLX.clip(out * 0.5 + 0.5, min: 0, max: 1)
        let result = rgb * keep + out01 * remove                          // fill only the hole
        result.eval()
        let filled = MIGANImage.cgImage(fromRGB: result)                  // at R×R
        // composite back at source resolution: redraw fill scaled, blend under the mask
        return MIGANImage.compositeToSource(source: source, fillAtRes: filled, mask: mask)
    }
}
