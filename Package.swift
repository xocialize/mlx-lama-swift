// swift-tools-version: 6.2
// mlx-lama-swift — LaMa (Big-LaMa) FFC inpainting on Swift/MLX, destined for an MLXEngine
// `imageInpaint` ModelPackage (Forge "Erase" capability). From-scratch architecture port of
// advimman/lama (Apache-2.0); the FFC spectral block uses MLX's native rFFT. See PORT-STATUS.md.
import PackageDescription

let mlxCore: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFFT", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
]

let package = Package(
    name: "mlx-lama-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "LaMa", targets: ["LaMa"]),         // FFC quality tier (core)
        .library(name: "MIGAN", targets: ["MIGAN"]),        // mobile-GAN fast tier (core)
        .library(name: "MLXInpaint", targets: ["MLXInpaint"]),  // engine-consumable ModelPackage
        .executable(name: "lama-smoke", targets: ["Smoke"]),
        .executable(name: "migan-smoke", targets: ["MIGANSmoke"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // MLXToolKit contract (imageInpaint @ 1.8.0). Pinned to the revision that introduced it.
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", revision: "8cd0033"),
    ],
    targets: [
        .target(name: "LaMa", dependencies: mlxCore, path: "Sources/LaMa",
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "MIGAN", dependencies: mlxCore, path: "Sources/MIGAN",
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "MLXInpaint",
            dependencies: [
                "LaMa", "MIGAN",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/MLXInpaint"),
        .executableTarget(
            name: "Smoke",
            dependencies: ["LaMa", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/Smoke", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(
            name: "MIGANSmoke",
            dependencies: ["MIGAN", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/MIGANSmoke", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
