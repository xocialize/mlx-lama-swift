import Foundation
import CoreGraphics
import MLX

/// CGImage ⇄ MLXArray bridging for MI-GAN (RGB `[0,1]`, NHWC). Fixed-resolution model → resize via CG.
enum MIGANImage {

    private static func draw(_ cg: CGImage, _ W: Int, _ H: Int) -> [UInt8] {
        let cs = CGColorSpaceCreateDeviceRGB()
        var buf = [UInt8](repeating: 0, count: W * H * 4)
        buf.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                                bytesPerRow: W * 4, space: cs,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
        }
        return buf
    }

    static func rgb(from cg: CGImage, width W: Int, height H: Int) -> MLXArray {
        let buf = draw(cg, W, H)
        var v = [Float](repeating: 0, count: W * H * 3)
        for p in 0 ..< W * H { for c in 0 ..< 3 { v[p * 3 + c] = Float(buf[p * 4 + c]) / 255 } }
        return MLXArray(v, [H, W, 3])
    }

    static func gray(from cg: CGImage, width W: Int, height H: Int) -> MLXArray {
        let buf = draw(cg, W, H)
        var v = [Float](repeating: 0, count: W * H)
        for p in 0 ..< W * H { v[p] = Float(buf[p * 4 + 0]) / 255 }
        return MLXArray(v, [H, W, 1])
    }

    static func cgImage(fromRGB arr: MLXArray) -> CGImage {
        let H = arr.dim(0), W = arr.dim(1)
        let rgb = MLX.clip(arr, min: 0, max: 1).asArray(Float.self)
        var buf = [UInt8](repeating: 255, count: W * H * 4)
        for p in 0 ..< W * H { for c in 0 ..< 3 { buf[p * 4 + c] = UInt8((rgb[p * 3 + c] * 255).rounded()) } }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = buf.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                      bytesPerRow: W * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        }
        return ctx.makeImage()!
    }

    /// Composite the (resolution-sized) fill back onto the source: keep source where mask is black,
    /// take the upscaled fill where mask is white. Done in CG at source resolution.
    static func compositeToSource(source: CGImage, fillAtRes: CGImage, mask: CGImage) -> CGImage {
        let W = source.width, H = source.height
        let cs = CGColorSpaceCreateDeviceRGB()
        let src = draw(source, W, H), fill = draw(fillAtRes, W, H), m = draw(mask, W, H)
        var out = [UInt8](repeating: 255, count: W * H * 4)
        for p in 0 ..< W * H {
            let hole = m[p * 4 + 0] > 127
            for c in 0 ..< 3 { out[p * 4 + c] = hole ? fill[p * 4 + c] : src[p * 4 + c] }
        }
        let ctx = out.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                      bytesPerRow: W * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        }
        return ctx.makeImage()!
    }
}
