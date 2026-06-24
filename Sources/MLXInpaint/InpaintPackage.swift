import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import Hub
import LaMa
import MIGAN

/// The conformant `imageInpaint` ModelPackage. One package, two tier Modes dispatched in `run(_:)` on
/// `request.mode` (`best` → LaMa FFC · `fast` → MI-GAN). Takes the contract's two-input shape (image +
/// mask, white=remove) and returns a filled `Image`. The engine constructs / loads / evicts it (C13).
@InferenceActor
public final class InpaintPackage: ModelPackage {
    public typealias Configuration = InpaintConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // LaMa Apache-2.0 + MI-GAN MIT (both permissive); port code MIT. Declare the more-restrictive
            // permissive layer (Apache) for the weight gate — both pass C7. (NOTICE records both.)
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/LaMa-bf16", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // Measured (M-Max): LaMa erase peak ~2.8 GB fp32 @880²; MI-GAN fixed 512² is lighter.
                // Activation-dominated, both consumer-friendly → one fp16 envelope covers both modes.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 4_500_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                InpaintContract.descriptor(
                    name: "inpaint",
                    summary: "Object removal / inpainting (image + mask → filled image). "
                        + "best = LaMa (large masks, structured bg) · fast = MI-GAN (mobile, 512).",
                    modes: [InpaintContract.best, InpaintContract.fast])
            ])
    }

    private let configuration: Configuration
    private var lama: LaMaInpainter?
    private var migan: MIGANInpainter?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws { if lama == nil { lama = try await buildLaMa() } }
    public func unload() async { lama = nil; migan = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard request.capability == .imageInpaint, let req = request as? InpaintRequest else {
            throw InpaintError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()
        let image = try Self.decode(req.image)
        let mask = try Self.decode(req.mask)
        let output: CGImage
        if req.mode == InpaintContract.fast {
            if migan == nil { migan = try await buildMIGAN() }
            output = migan!(image, mask: mask)
        } else {
            if lama == nil { lama = try await buildLaMa() }
            output = lama!(image, mask: mask)
        }
        try Task.checkCancellation()
        let png = try Self.encodePNG(output)
        return InpaintResponse(image: Image(format: .png, data: png, width: output.width, height: output.height))
    }

    // MARK: build

    private func buildLaMa() async throws -> LaMaInpainter {
        let url = try await weightsURL(configuration.lamaWeightsURL, configuration.lamaRepo)
        // LaMa MUST run at bf16+ — its FFC bottleneck activations (~1e3) collapse under fp16 (mean err 0.55).
        return try LaMaInpainter.fromPretrained(url.path, dtype: .bfloat16)
    }
    private func buildMIGAN() async throws -> MIGANInpainter {
        let url = try await weightsURL(configuration.miganWeightsURL, configuration.miganRepo)
        // MI-GAN is well-scaled → fp16 is accurate (mean 3e-4) and the smallest footprint.
        return try MIGANInpainter.fromPretrained(url.path, resolution: configuration.miganResolution, dtype: .float16)
    }

    private func weightsURL(_ override: URL?, _ repo: String) async throws -> URL {
        if let override {
            guard FileManager.default.fileExists(atPath: override.path) else { throw InpaintError.weightsMissing(override) }
            return override
        }
        let hub = HubApi(downloadBase: configuration.modelsRootDirectory)
        let dir = try await hub.snapshot(from: repo, matching: ["*.safetensors"]) { @Sendable p in
            WeightDownloadProgress.report(fraction: p.fractionCompleted)
        }
        let url = dir.appendingPathComponent(configuration.weightsFile)
        guard FileManager.default.fileExists(atPath: url.path) else { throw InpaintError.weightsMissing(url) }
        return url
    }

    private nonisolated static func dtype(_ q: Quant) -> DType {
        switch q { case .fp32: return .float32; case .bf16: return .bfloat16; default: return .float16 }
    }

    private nonisolated static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw InpaintError.decodeFailed }
        return cg
    }
    private nonisolated static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw InpaintError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw InpaintError.encodeFailed }
        return data as Data
    }

    public enum InpaintError: Error {
        case unsupportedCapability(Capability)
        case weightsMissing(URL)
        case decodeFailed, encodeFailed
    }
}

public extension InpaintPackage {
    nonisolated static var registration: PackageRegistration { .of(InpaintPackage.self) }
}
