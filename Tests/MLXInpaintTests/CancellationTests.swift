// CancellationTests.swift — LaMa/MI-GAN inpaint through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before
// capability validation, image decode, or weights); CAN-3 is the document of record for the
// checkpoint cadence. Inpaint is a CAN-3 judgment case: each tier (best = LaMa FFC, fast =
// MI-GAN) is a SINGLE forward — one monolithic lazy MLX graph self-evaled before CGImage
// materialization, no tiling, no iterative loop — yet peakActivationBytes 4.0 GB ≥ 2 GB makes
// the manifest long-run implied. The declared cadence therefore names the REAL seams run() has,
// once per inpainted frame each:
//   • entry checkpoint (first act of run(), InpaintPackage.run)
//   • pre-forward, after the lazy tier build — a first-request weight download can precede
//     the forward (InpaintPackage.run, both mode arms)
//   • post-forward / pre-PNG-encode (InpaintPackage.run, after endRun)
// No per-step checkpoints are fabricated — there is no loop to hang them on.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXInpaint

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = InpaintPackage(configuration: InpaintConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: InpaintRequest(image: Image(format: .png, data: Data()),
                                    mask: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // peakActivationBytes 4.0 GB ≥ 2 GB ⇒ long-run implied — the sub-second exemption is
        // not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: InpaintPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: InpaintPackage.manifest,
            posture: .cadence([
                // Pre-forward seam: once per frame, after the lazy tier build / possible weight
                // download, before committing to the monolithic FFC (or MI-GAN) eval.
                .init(phase: .encode, unit: .frame),
                // Post-forward seam: once per frame, between materialization and PNG encode.
                .init(phase: .postprocess, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
