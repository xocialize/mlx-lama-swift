import Foundation
import CoreGraphics
import MLX

/// CGImage ⇄ MLXArray bridging for LaMa (RGB `[0,1]`, NHWC `(H,W,3)`; masks `(H,W,1)`).
enum LaMaImage {

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
        var rgb = [Float](repeating: 0, count: W * H * 3)
        for p in 0 ..< W * H {
            rgb[p * 3 + 0] = Float(buf[p * 4 + 0]) / 255
            rgb[p * 3 + 1] = Float(buf[p * 4 + 1]) / 255
            rgb[p * 3 + 2] = Float(buf[p * 4 + 2]) / 255
        }
        return MLXArray(rgb, [H, W, 3])
    }

    static func gray(from cg: CGImage, width W: Int, height H: Int) -> MLXArray {
        let buf = draw(cg, W, H)
        var g = [Float](repeating: 0, count: W * H)
        for p in 0 ..< W * H { g[p] = Float(buf[p * 4 + 0]) / 255 }    // R channel as luminance
        return MLXArray(g, [H, W, 1])
    }

    /// Reflect-pad the bottom/right by (ph,pw) to reach a multiple of 8.
    static func padHW(_ x: MLXArray, _ ph: Int, _ pw: Int) -> MLXArray {
        if ph == 0 && pw == 0 { return x }
        let H = x.dim(0), W = x.dim(1)
        func idx(_ n: Int, _ pad: Int) -> MLXArray {
            var ids = (0 ..< n).map { Int32($0) }
            ids.append(contentsOf: stride(from: n - 2, to: n - 2 - pad, by: -1).map { Int32($0) })
            return MLXArray(ids)
        }
        var y = MLX.take(x, idx(H, ph), axis: 0)
        y = MLX.take(y, idx(W, pw), axis: 1)
        return y
    }

    static func cgImage(fromRGB arr: MLXArray) -> CGImage {
        let H = arr.dim(0), W = arr.dim(1)
        let rgb = MLX.clip(arr, min: 0, max: 1).asArray(Float.self)
        var buf = [UInt8](repeating: 255, count: W * H * 4)
        for p in 0 ..< W * H {
            buf[p * 4 + 0] = UInt8((rgb[p * 3 + 0] * 255).rounded())
            buf[p * 4 + 1] = UInt8((rgb[p * 3 + 1] * 255).rounded())
            buf[p * 4 + 2] = UInt8((rgb[p * 3 + 2] * 255).rounded())
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = buf.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                      bytesPerRow: W * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        }
        return ctx.makeImage()!
    }
}
