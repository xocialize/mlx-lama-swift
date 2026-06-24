import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import ArgumentParser
import MLX
import LaMa

/// LaMa parity gate + end-to-end erase. Parity: load converted weights + fixture (4ch input +
/// golden predicted), run LaMaModel, assert max_abs (CPU stream). Erase: real CGImage + box mask
/// → inpainted PNG (GPU, production path).
@main
struct Smoke: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "lama-smoke",
        abstract: "LaMa MLX-Swift parity gate + end-to-end erase.")

    @Option(name: .long) var weights: String
    @Option(name: .long) var parity: String?
    @Option(name: .long) var threshold: Float = 1e-3
    @Option(name: .long, help: "Erase: input image path") var eraseImage: String?
    @Option(name: .long, help: "Erase: box to remove, 'x,y,w,h'") var box: String?
    @Option(name: .long, help: "Erase: output PNG") var out: String?

    func run() throws {
        if let eraseImage, let box, let out { try erase(eraseImage, box, out); return }
        Device.setDefault(device: Device(.cpu))
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.float32) }
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: parity!))
        let out = LaMaModel(weights: w)(fx["input4"]!.asType(.float32)); out.eval()
        let golden = fx["predicted"]!.asType(.float32)
        let maxAbs = MLX.abs(out - golden).max().item(Float.self)
        print(String(format: "[lama-smoke] predicted out%@ golden%@  max_abs=%.3e mean=%.3e  %@",
                     shape(out), shape(golden), maxAbs, MLX.abs(out - golden).mean().item(Float.self),
                     maxAbs < threshold ? "OK ✅" : "FAIL ❌"))
        if maxAbs >= threshold { throw ExitCode(1) }
    }

    private func erase(_ imgPath: String, _ box: String, _ outPath: String) throws {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imgPath) as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw ExitCode(1) }
        let c = box.split(separator: ",").map { Int($0)! }   // x,y,w,h
        let mask = boxMask(width: cg.width, height: cg.height, x: c[0], y: c[1], w: c[2], h: c[3])
        let inpainter = try LaMaInpainter.fromPretrained(weights, dtype: .float32)
        MLX.GPU.resetPeakMemory()
        let start = Date()
        let result = inpainter(cg, mask: mask)
        let secs = Date().timeIntervalSince(start)
        try writePNG(result, outPath)
        print(String(format: "[lama-smoke] erase %dx%d box=%@ → %@ (%.2fs, peak %.2f GB)",
                     cg.width, cg.height, box, outPath, secs, Double(MLX.GPU.peakMemory) / 1e9))
    }

    private func boxMask(width: Int, height: Int, x: Int, y: Int, w: Int, h: Int) -> CGImage {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.setFillColor(gray: 0, alpha: 1); ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        // interpret (x,y) as top-down image coords (CGContext origin is bottom-left)
        ctx.setFillColor(gray: 1, alpha: 1); ctx.fill(CGRect(x: x, y: height - y - h, width: w, height: h))
        return ctx.makeImage()!
    }

    private func writePNG(_ cg: CGImage, _ path: String) throws {
        guard let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                      UTType.png.identifier as CFString, 1, nil) else { throw ExitCode(1) }
        CGImageDestinationAddImage(d, cg, nil)
        guard CGImageDestinationFinalize(d) else { throw ExitCode(1) }
    }

    private func shape(_ a: MLXArray) -> String { "(" + a.shape.map(String.init).joined(separator: ",") + ")" }
}
