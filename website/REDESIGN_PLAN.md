# Explainer redesign · 2026-10-02

The subject architecture below remains the basis of the explainer. The visual treatment was revised after the user supplied a dark, dense instrumentation reference. The current site uses six top-bar views instead of revealing every subject by continuous scroll. Its first view uses the repository's five-point M logo, a black compute field, compact status rail, operator controls, and a calculated telemetry dock. The unrelated bot, login, spending, and throughput content from the reference was not reused.

## Reference principle, not visual imitation

Transformer Explainer puts a manipulable computation and its controls on the first screen. Labels, intermediate states, and output share one frame, so experimentation explains the system before the long-form text. Its actual lower page is a technical article; the scroll-driven execution and phase scenes here are our own extension. We do not reuse its diagram, color, layout, or components.

## Audit of the prior page

- A 5rem headline, dark vignette, and pointer-reactive ambient nodes dominate the first viewport. Those nodes do not encode a computation.
- Its three-step hero visualizes a fixed Morton example. FPS, kNN, and Ball Query cannot be manipulated there.
- Installation and an exploratory SciPy comparison appear before the computation is explained. That comparison timed unequal radius workloads and sums separate stage medians.
- Static tables are precise but make the hardware, operator, fixture, and release scope hard to inspect. The source contains no complete hardware × size × operator matrix.
- Strengths worth retaining: exact BigInt Morton teaching model, deterministic AABB demo, keyboard controls, English copy, and linked raw evidence.

## Information architecture

1. **Compute** — one fixed toy point set, switching FPS / kNN / Ball Query and controlling the relevant parameter. Result indices and rule are visible with the scene.
2. **Route** — a single execution diagram shifts from blocked CUDA-only call to PyTorch → compatibility → MPS → Metal → Apple GPU. Scroll selects a step; buttons and keyboard provide the same states.
3. **Run** — an ecosystem map answers model-level support with separate *release*, *validated fixture*, forward/backward, device, and limit fields.
4. **Grow** — one scene evolves from Phase 1 through Phase 5. Future layers remain visibly proposed; no unfinished phase receives a completed state.
5. **Trust** — correctness, compatibility, and performance each have evidence, a contract, and a limitation. Morton and BVH hands-on details follow as optional deeper investigation.
6. **Measure** — JSON-derived hardware/size/operator selectors show only recorded matching fixtures. Exact baseline and timing scope accompany every bar; unavailable combinations say so.
7. **Use** — released v0.8.0 installation and a small compatibility example after the explanation. A source-only v0.9 research path is explicitly separate.
8. **Provenance** — raw logs, source commits, licenses, citation and open validation limits.

## Visual system

- Near-black observatory surface, fine structural rules, restrained cyan/coral/amber computational accents, and a full-width point-cloud field. The central relationships are derived from the current algorithm state, not ambient network decoration.
- Semantic roles: FPS selection = coral, kNN = cyan, Ball Query radius = amber; selected points add white cores and colored rings. Text and explicit results accompany every color cue.
- Compact monospaced status, coordinate, index, and evidence labels sit beside readable sans headlines and explanatory copy. The top rail identifies the scene as a JavaScript 2D model with no GPU timing.
- Desktop keeps a field, inspector, and four calculated telemetry cells in one instrument. Mobile stacks these in reading order while retaining full-width operator controls and textual output.

## Interaction and motion contracts

- A deterministic 2D teaching fixture persists across operator tabs. FPS follows actual farthest-first selection, kNN ranks distance with a fixed tie rule, and Ball Query takes the first K input indices passing strict radius comparison. The canvas is a teaching model, not GPU footage or float32 parity evidence.
- Playback steps through FPS, kNN, or Ball Query selection. Long Ball Query scans condense rejection-only frames for bounded playback while preserving all accepted frames and exact final inspection counts. Focus, labels and a live text result make canvas output readable without color or pointer input.
- Animation stops offscreen or in a hidden tab; reduced-motion disables autoplay and nonessential transitions. No scroll hijacking or parallax.
- Scroll-linked story uses intersection state only. It never traps scroll and exposes an equivalent numbered step control.

## Source-of-truth boundary

- Published package: v0.8.0. Current merged main: `905fce5`; source-only v0.9 SpatialIndex and expanded Chamfer are research paths, not PyPI features.
- Model fixtures have bounded claims: PointNet++ synthetic segmentation, DGCNN synthetic classifier, PyG 2.8 subset, Pointcept PTv1 subset; Sparse3D remains planned. A synthetic pass is never called production model support.
- Do not equate stage medians with end-to-end latency, prototype SciPy all-neighbor work with first-K Metal work, M1 data with M5 data, or PyTorch allocator peaks with physical GPU peaks.

## Build and review gates

1. Implement hero/operator playground and inspect the first desktop/mobile screen.
2. Implement route, capability and phase scenes; verify controls and text equivalents.
3. Implement evidence explorer and correctness/provenance; verify every displayed number against its bundled raw record.
4. Run algorithm/content tests, then desktop/mobile/reduced-motion/keyboard browser QA and a design review for hierarchy, spacing, comprehension, animation and accessibility.
