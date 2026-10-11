import Foundation
import MLXToolKit

/// Where a weights file of a declared `WeightSource` sits under the engine's model store.
///
/// Since 0.4.0 the engine materializes `InpaintConfiguration.weightSources` before `load()` (contract 1.24), so a
/// file is found in one of three layouts, tried in order:
/// - the engine's flat layout, `<root>/models--<org>--<name>/<file>`;
/// - the hub-client snapshot, `<root>/models--<org>--<name>/snapshots/<commit>/<file>`;
/// - swift-transformers' `HubApi(downloadBase: root)` layout, `<root>/models/<org>/<name>/<file>`, where this package
///   downloaded its own weights before 0.4.0. Existing installs keep loading from there, with no second download.
enum StoreWeights {
    static func file(_ name: String, repo: String, root: URL?) -> URL? {
        guard let root else { return nil }
        let store = ModelStore(root: root)
        let directories = [store.directory(for: repo), store.snapshotDirectory(for: repo), hubAPIDirectory(repo, root: root)]
        return directories.lazy.compactMap { $0?.appendingPathComponent(name) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// The pre-0.4.0 download location.
    static func hubAPIDirectory(_ repo: String, root: URL) -> URL {
        root.appendingPathComponent("models").appendingPathComponent(repo)
    }

    /// Whether `name` is installed in the pre-0.4.0 layout, which the engine's own probe doesn't know.
    static func installedByHubAPI(_ name: String, repo: String, root: URL?) -> Bool {
        guard let root else { return false }
        return FileManager.default.fileExists(atPath: hubAPIDirectory(repo, root: root).appendingPathComponent(name).path)
    }
}
