import Foundation
import MLXToolKit

/// Configuration for the inpaint `imageInpaint` package. Two models back the two tier Modes —
/// `InpaintContract.best` = LaMa (FFC, quality) · `.fast` = MI-GAN (mobile GAN). Weights resolve under
/// the engine's model store (`modelsRootDirectory` + repo dir + `weightsFile`).
public struct InpaintConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var lamaRepo: String
    public var miganRepo: String
    public var miganResolution: Int          // MI-GAN is fixed-resolution (512 default)
    public var weightsFile: String
    public var quant: Quant
    public var modelsRootDirectory: URL?
    public var lamaWeightsURL: URL?
    public var miganWeightsURL: URL?

    public init(lamaRepo: String = "mlx-community/LaMa-bf16",   // bf16: LaMa's FFC breaks at fp16
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
