import Foundation
import CoreGraphics
import MLX

/// End-to-end LaMa eraser: (source CGImage, mask CGImage) → inpainted CGImage, mirroring the LaMa
/// predict pipeline. Holds the loaded generator; the engine wrapper owns lifecycle.
///
///   input4  = concat[ rgb*(1-mask), mask ]   (H,W padded to ×8)
///   pred    = generator(input4)              (sigmoid RGB)
///   result  = mask*pred + (1-mask)*rgb       (composite; only the hole changes), cropped to source
public final class LaMaInpainter: @unchecked Sendable {

    private let model: LaMaModel
    public init(weights: [String: MLXArray]) { self.model = LaMaModel(weights: weights) }

    public static func fromPretrained(_ weightsPath: String, dtype: DType = .float32) throws -> LaMaInpainter {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)).mapValues { $0.asType(dtype) }
        return LaMaInpainter(weights: w)
    }

    /// `mask`: grayscale CGImage, white (>0.5) = remove. Output preserves source resolution.
    public func callAsFunction(_ source: CGImage, mask: CGImage) -> CGImage {
        let H = source.height, W = source.width
        let rgb = LaMaImage.rgb(from: source, width: W, height: H)          // (H,W,3) [0,1]
        var m = LaMaImage.gray(from: mask, width: W, height: H)            // (H,W,1) [0,1]
        m = (m .> 0.5)                                                     // binarize → 0/1 float
        // pad H,W up to a multiple of 8 (3 downsamples) with reflection
        let ph = (8 - H % 8) % 8, pw = (8 - W % 8) % 8
        let rgbP = LaMaImage.padHW(rgb, ph, pw), mP = LaMaImage.padHW(m, ph, pw)
        let masked = rgbP * (1 - mP)
        let input4 = MLX.concatenated([masked, mP], axis: -1).expandedDimensions(axis: 0)
        let pred = model(input4)[0]                                        // (Hp,Wp,3)
        let result = mP * pred + (1 - mP) * rgbP
        let cropped = result[0 ..< H, 0 ..< W, 0...]
        cropped.eval()
        return LaMaImage.cgImage(fromRGB: cropped)
    }
}
