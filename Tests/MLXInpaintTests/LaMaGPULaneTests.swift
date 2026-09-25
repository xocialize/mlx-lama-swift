// GPU-lane gate for mlx's lossy Winograd conv2d window in Big-LaMa (Sources/LaMa/
// WinogradConvRoute.swift): the production-dtype forward (bf16 weights × fp32 activations) on a
// real photo with a fixed hole, on the CPU stream (exact-class reference) and on the GPU with the
// conv route on (default) and off (raw Winograd). lama-smoke parity runs on the CPU with fp32
// weights, so the shipped GPU lane was never checked numerically.
//
// Measured 2026-09-24 (M5 Max, mlx-swift 0.31.6; 1024×681 whole-object erase): raw GPU vs CPU in
// the hole relL2 3.4e-3, 8-bit max 9 levels, hole PSNR 53 dB; with MLX_ENABLE_TF32=0 or the
// in-window convs on CPU, 1 level max — the Winograd convs are the whole drift.
//
// Run: LAMA_LANE=1 swift test -c release -Xswiftc -enable-testing --filter LaMaGPULaneTests
// Overrides: LAMA_WEIGHTS (model.safetensors), LAMA_REAL_IMAGE.

import CoreGraphics
import Foundation
import ImageIO
import LaMa
import MLX
import XCTest

final class LaMaGPULaneTests: XCTestCase {
    func testHoleGPUvsCPU() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["LAMA_LANE"] == "1", "set LAMA_LANE=1 to run")
        let weights = env["LAMA_WEIGHTS"]
            ?? "/Volumes/Satechi/Models/models/mlx-community/LaMa-bf16/model.safetensors"
        let image = URL(fileURLWithPath: env["LAMA_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.bfloat16) }
        let model = LaMaModel(weights: w)

        // Center crop 1024×680 (a multiple of 8, as production pads to), [0,1] RGB.
        guard let src = CGImageSourceCreateWithURL(image as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "LaMa", code: 1) }
        let (iw, ih, W, H) = (cg.width, cg.height, 1024, 680)
        var rgba = [UInt8](repeating: 0, count: iw * ih * 4)
        let ctx = CGContext(
            data: &rgba, width: iw, height: ih, bitsPerComponent: 8, bytesPerRow: iw * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: iw, height: ih))
        let (x0, y0) = ((iw - W) / 2, (ih - H) / 2)
        var rgb = [Float](repeating: 0, count: H * W * 3), m = [Float](repeating: 0, count: H * W)
        for y in 0..<H {
            for x in 0..<W {
                let p = ((y0 + y) * iw + (x0 + x)) * 4
                for c in 0..<3 { rgb[(y * W + x) * 3 + c] = Float(rgba[p + c]) / 255 }
                // fixed elliptical hole, ~18% of the frame
                let dx = (Float(x) - Float(W) / 2) / (0.26 * Float(W))
                let dy = (Float(y) - Float(H) / 2) / (0.28 * Float(H))
                m[y * W + x] = dx * dx + dy * dy <= 1 ? 1 : 0
            }
        }
        let img = MLXArray(rgb, [H, W, 3]), mask = MLXArray(m, [H, W, 1])
        let input4 = concatenated([img * (1 - mask), mask], axis: -1).expandedDimensions(axis: 0)

        let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = model(input4)
            eval(r)
            return r
        }
        Memory.clearCache()
        var t: [LaMaConvRoute: [Double]] = [:], out: [LaMaConvRoute: MLXArray] = [:]
        for _ in 0..<3 {
            for route in [LaMaConvRoute.conv3d, .winograd] {
                model.convRoute = route
                var y = model(input4)
                eval(y)
                let t0 = Date()
                for _ in 0..<3 { y = model(input4); eval(y) }
                t[route, default: []].append(Date().timeIntervalSince(t0) / 3 * 1000)
                out[route] = y
            }
        }
        model.convRoute = .conv3d
        let hole = mask.expandedDimensions(axis: 0)
        func holeStats(_ a: MLXArray) -> (rel: Float, text: String) {
            let d = (a - ref) * hole
            let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref * hole)))
            let lv = abs(round(a * 255) - round(ref * 255)) * hole
            let mse = sum(square(round(a * 255) - round(ref * 255)) * hole) / sum(hole) / 3
            eval(rel, lv, mse)
            let psnr = 10 * log10(255 * 255 / max(mse.item(Float.self), 1e-12))
            return (rel.item(Float.self), String(format: "hole relL2 %.2e  8-bit max %d levels  hole PSNR %.1f dB",
                rel.item(Float.self), Int(lv.max().item(Float.self)), psnr))
        }
        func med(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        let sR = holeStats(out[.conv3d]!), sW = holeStats(out[.winograd]!)
        print("[\(image.lastPathComponent) 1024×680, ~18% hole, bf16 weights × fp32, GPU vs CPU lane]")
        print(String(format: "  conv3d route   %@  %6.1f ms", sR.text, med(t[.conv3d]!)))
        print(String(format: "  raw Winograd   %@  %6.1f ms", sW.text, med(t[.winograd]!)))
        XCTAssertLessThan(sR.rel, 1e-4, "conv3d route vs CPU lane")
    }
}
