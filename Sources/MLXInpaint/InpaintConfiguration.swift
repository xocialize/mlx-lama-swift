import Foundation
import MLXToolKit

/// Configuration for the inpaint `imageInpaint` package. Two models back the two tier Modes —
/// `InpaintContract.best` = LaMa (FFC, quality) · `.fast` = MI-GAN (mobile GAN). Both repos are declared
/// `weightSources`, which the engine downloads into its model store before `load()`.
public struct InpaintConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var lamaRepo: String
    public var miganRepo: String
    public var miganResolution: Int          // MI-GAN is fixed-resolution (512 default)
    public var weightsFile: String
    public var quant: Quant
    public var modelsRootDirectory: URL?
    public var lamaWeightsURL: URL?
    public var miganWeightsURL: URL?

    public init(lamaRepo: String = "mlx-community/LaMa-fp32",   // fp32: bf16 casts fills, fp16 collapses
                miganRepo: String = "mlx-community/MI-GAN-512-places2-fp16",
                miganResolution: Int = 512,
                weightsFile: String = "model.safetensors",
                quant: Quant = .fp16,
                modelsRootDirectory: URL? = nil,
                lamaWeightsURL: URL? = nil,
                miganWeightsURL: URL? = nil) {
        self.lamaRepo = lamaRepo
        self.miganRepo = miganRepo
        self.miganResolution = miganResolution
        self.weightsFile = weightsFile
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.lamaWeightsURL = lamaWeightsURL
        self.miganWeightsURL = miganWeightsURL
    }
}

/// The weights the engine materializes before `load()` (contract 1.24): both tiers' checkpoints, so the engine can
/// size, preview and download them (and resume a broken transfer) instead of the package fetching on first use.
/// MI-GAN is 15 MB beside LaMa's 204 MB.
extension InpaintConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "lama", repo: lamaRepo, matching: [weightsFile]),
         WeightSource(role: "migan", repo: miganRepo, matching: [weightsFile])]
    }

    /// The explicit weights URLs first, then the store's layouts, including where 0.3.x downloaded them.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        defaultMissingWeightSources(storeRoot: storeRoot).filter { source in
            let override = source.role == "lama" ? lamaWeightsURL : miganWeightsURL
            if let override, FileManager.default.fileExists(atPath: override.path) { return false }
            return !StoreWeights.installedByHubAPI(weightsFile, repo: source.repo, root: storeRoot)
        }
    }
}
