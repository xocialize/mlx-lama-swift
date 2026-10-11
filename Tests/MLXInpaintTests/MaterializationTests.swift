// MaterializationTests.swift — the engine materializes LaMa's and MI-GAN's weights before load() (contract 1.24,
// AB-T-0222). Offline: no network, no weights, no MLX kernels.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXInpaint

final class MaterializationTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "inpaint-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func place(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(path: "model.safetensors")
        try Data([0]).write(to: file)
        return file
    }

    func testMATGate() throws {
        let lama = try place(root.appending(path: "explicit/lama"))
        let migan = try place(root.appending(path: "explicit/migan"))
        let report = MaterializationConformance.check(
            freshConfiguration: InpaintConfiguration(),
            satisfiedConfiguration: InpaintConfiguration(lamaWeightsURL: lama, miganWeightsURL: migan))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testBothTiersAreDeclared() {
        let sources = InpaintConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["lama", "migan"])
        XCTAssertEqual(sources.map(\.repo), ["mlx-community/LaMa-fp32", "mlx-community/MI-GAN-512-places2-fp16"])
        XCTAssertEqual(sources.map(\.matching), [["model.safetensors"], ["model.safetensors"]])
    }

    /// The engine's flat layout satisfies the probe, and the loader finds the file there.
    func testEngineFlatLayout() throws {
        let config = InpaintConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).map(\.role), ["lama", "migan"])
        let file = try place(ModelStore(root: root).directory(for: config.lamaRepo)!)
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).map(\.role), ["migan"])
        XCTAssertEqual(StoreWeights.file("model.safetensors", repo: config.lamaRepo, root: root)?.path, file.path)
    }

    /// The hub-client snapshot layout (`refs/main` → `snapshots/<commit>/`).
    func testSnapshotLayout() throws {
        let repo = ModelStore(root: root).directory(for: "mlx-community/MI-GAN-512-places2-fp16")!
        try FileManager.default.createDirectory(at: repo.appending(path: "refs"), withIntermediateDirectories: true)
        try "abc123".write(to: repo.appending(path: "refs/main"), atomically: true, encoding: .utf8)
        let file = try place(repo.appending(path: "snapshots/abc123"))
        XCTAssertEqual(StoreWeights.file("model.safetensors", repo: "mlx-community/MI-GAN-512-places2-fp16", root: root)?.path,
                       file.path)
    }

    /// 0.3.x downloaded through `HubApi(downloadBase:)` into `<root>/models/<org>/<name>/`: those installs read as
    /// present, so the engine doesn't download them again, and the loader uses them.
    func testPre040InstallIsNotDownloadedAgain() throws {
        let config = InpaintConfiguration(modelsRootDirectory: root)
        let lama = try place(root.appending(path: "models/mlx-community/LaMa-fp32"))
        _ = try place(root.appending(path: "models/mlx-community/MI-GAN-512-places2-fp16"))
        XCTAssertTrue(config.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(StoreWeights.file("model.safetensors", repo: config.lamaRepo, root: root)?.path, lama.path)
    }

    /// An explicit URL satisfies its own tier only.
    func testExplicitURLSatisfiesItsTier() throws {
        let lama = try place(root.appending(path: "explicit"))
        let config = InpaintConfiguration(modelsRootDirectory: root, lamaWeightsURL: lama)
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).map(\.role), ["migan"])
    }

    func testNoStoreFindsNothing() {
        XCTAssertNil(StoreWeights.file("model.safetensors", repo: "mlx-community/LaMa-fp32", root: nil))
        XCTAssertEqual(InpaintConfiguration().missingWeightSources(storeRoot: nil).count, 2)
    }
}
