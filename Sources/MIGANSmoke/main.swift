import Foundation
import ArgumentParser
import MLX
import MIGAN

/// MI-GAN parity gate vs the PyTorch golden output (CPU stream).
@main
struct MIGANSmoke: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "migan-smoke",
        abstract: "MI-GAN MLX-Swift parity gate.")

    @Option(name: .long) var weights: String
    @Option(name: .long) var parity: String
    @Option(name: .long) var resolution: Int = 512
    @Option(name: .long) var threshold: Float = 1e-3

    func run() throws {
        Device.setDefault(device: Device(.cpu))
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.float32) }
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: parity))
        let out = MIGANModel(resolution: resolution, weights: w)(fx["input4"]!.asType(.float32)); out.eval()
        let golden = fx["output"]!.asType(.float32)
        let maxAbs = MLX.abs(out - golden).max().item(Float.self)
        print(String(format: "[migan-smoke] output out%@ golden%@  max_abs=%.3e mean=%.3e  %@",
                     shape(out), shape(golden), maxAbs, MLX.abs(out - golden).mean().item(Float.self),
                     maxAbs < threshold ? "OK ✅" : "FAIL ❌"))
        if maxAbs >= threshold { throw ExitCode(1) }
    }
    private func shape(_ a: MLXArray) -> String { "(" + a.shape.map(String.init).joined(separator: ",") + ")" }
}
