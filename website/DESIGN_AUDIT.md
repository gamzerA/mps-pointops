# Explainer redesign audit

Local draft, 2026-10-02. The reference pages were studied for teaching structure and interaction. Their source code, copy, illustrations, and visual identity were not copied.

## Reference → design decision

| Reference | Observation | Application here |
| --- | --- | --- |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | A real, connected computation occupies most of the initial viewport; controls change the diagram rather than decorate a headline. | The first viewport is a working FPS/kNN/Ball Query point-cloud instrument. The selected indices and the reason for selection change with the controls. |
| [VAE Explainer](https://xnought.github.io/vae-explainer/) | Input, operation, and output remain legible as one process, with math available on demand. | One point set persists across operator tabs; the inspector names the current step and result. Deeper Morton and pruning derivations follow later. |
| [Interactive Attention](https://nipunbatra.github.io/interactive-articles/attention/index.html) | Small, adjustable examples make a calculation easier to test than a static diagram. | Point count, `k`, radius, query, and FPS start can be changed directly; the browser computes each step. |

The page now uses the user-supplied dark instrumentation screenshot as a visual reference: a thin status rail, a dominant dense compute field, and a bottom diagnostic dock. Its bot/scraper labels, spending gauge, activity figures, and artwork were not copied. The existing five-point M logo from the mps-pointops README appears once, without a surrounding box, beside a larger header wordmark. Coral, cyan, and amber encode the actual FPS, kNN, and Ball Query state. The field displays a sparse sample of computed relationship lines; selected indices and the underlying trace remain complete. The top rail and dock explicitly identify this as a JavaScript 2D teaching model, not a Metal trace or performance measurement.

## Earlier-page findings and replacement

| Previous problem | Revised behavior |
| --- | --- |
| A dark oversized title and ambient nodes led before the actual operation. | The first screen centers a large, editable point cloud. The single header logo and wordmark replace the question headline; a screen-reader heading remains in Compute. Ambient decorative animation is no longer mounted. |
| A fixed Morton/radius animation explained one research concept before FPS or kNN. | Operation tabs share one deterministic point set and expose the actual FPS, kNN, and input-order first-`K` Ball Query choices. The Morton and AABB demonstrations move into the methods section. |
| Installation and a prototype time ratio appeared too early. | Install has its own view, while the computation and evidence are directly available through the top bar. The unequal SciPy/prototype comparison is not in the benchmark explorer. |
| Source state and package state could appear interchangeable. | PyPI v0.8.0, archived source snapshots, and unreleased v0.9 BVH measurements use separate labels. A selected ecosystem item lists both release and validation status. |
| A broad ecosystem claim could be read as a whole-model guarantee. | Each item includes named fixture, forward/backward coverage, hardware, limitations, and an evidence link. Sparse 3D remains planned. |
| A table of a few benchmark rows hid missing combinations. | Hardware/operation/`N` filters display only raw-record-backed fixtures and an explicit empty state for unmeasured selections. Each bar resets scale within its own fixture. |

## Page architecture and interaction contract

The top bar now switches six URL-hash views without a delayed smooth scroll. Existing deep links such as `#pruning` and `#provenance`, browser history, active menu state, and the skip link resolve to the visible view. The large numbered labels that made the long page resemble slides were removed from section headings; numbers remain where they represent actual execution steps or roadmap phases.

1. **Compute:** `src/point-playground.js` supplies the deterministic 2D selection rules; `src/compute-field.js` renders up to 1,800 input points and only relationships computed from those rules. FPS maximizes minimum squared distance to selected points; kNN sorts by squared distance and original index; Ball Query scans input order and accepts `d² < r²` until `K` hits. The canvas uses equal x/y pixel scales, so the drawn radius is a real circle. Long Ball scans condense rejection-only playback frames, with accepted frames and final counts intact. Active controls remain visible; point count and step/reset live in a disclosure. The dock derives frame progress, inspected count, indices, and rule from the same state. This JavaScript calculation does not promise Metal float32 parity.
2. **Execution:** six compact buttons directly select CUDA-only blockage, PyTorch, compatibility routing, MPS, Metal, and result tensor. Fine-pointer hover on a step or diagram node previews it after 80 ms; leaving restores the last deliberate selection after 100 ms. Click and keyboard activation commit immediately. The diagram and current-step detail fit in the first desktop viewport; scrolling does not change the selected step. This is conceptual, not a runtime trace.
3. **Ecosystem:** a two-button chooser opens model evidence or the compute stack within What runs. Those top-level panels switch on click so passing the pointer over them cannot hide a focused evidence link. Model cards respond to a 90 ms fine-pointer hover or direct click/keyboard selection. `data/capabilities.json` pins merged source and published package separately. The map does not turn a passing synthetic fixture into a blanket support claim.
4. **Phase progression:** five direct buttons also respond to 90 ms fine-pointer hover and update one SVG scene with feature, graph/grid, geometry, and planned sparse layers. This depicts a research roadmap, not five completed released layers.
5. **Method:** exact BigInt Morton keys and a 2D AABB pruning model remain inspectable. The browser model has 16-point bricks and reports visited subtree roots and distance checks; the source GPU BVH is 3D with 128-point bricks. The numerical boundary arithmetic differs.
6. **Evidence and use:** the Measurements view contains validation, the explorer backed by 17 bundled raw records, and provenance. The Install view contains the recorded v0.8.0 package instructions.

## Evidence and limitations

- The benchmark explorer contains M5 Pro dense 20k/100k source snapshots, M5 Pro flat 1k/4k public-API snapshots, physical M1 B=1 FPS 500k/1M pairs, and three M5 Pro unreleased BVH/scan fixtures at 1M points. It does not interpolate a missing point count or substitute a different GPU's result.
- The M5 Pro dense source-synchronized record lacks an original source commit in the raw JSON. It contains per-file SHA-256 hashes and an archived Git record; the UI names this gap. The M1 and flat source snapshots record clean commits. The v0.9 spatial records pin a clean measurement commit.
- The M5 Pro v0.9 BVH examples compare **steady query** with preloaded tensors and, where applicable, an already built index. Build, transfer, and shader compilation are outside that paired bar. Auto can choose scan even when opt-in BVH is faster on a specific fixture.
- Physical M1 large spatial validation, M2–M4 measurements, transient GPU physical-memory peak, real-dataset model accuracy, and blanket PyG/Pointcept/Sparse 3D compatibility remain open. The page does not imply that v0.9 source-only APIs ship in PyPI 0.8.0.
- The original `256.175 → 6.669 ms` prototype comparison is unequal work: SciPy timed all radius hits and MPS timed first `K=16`; its total sums separate build and query medians. It remains archived as evidence, not a matched speedup claim in the redesigned interface.

## Motion and accessibility design

- Motion follows computation: selected or inspected points progress through actual model frames; route/phase emphasis follows the selected explanation step. The Compute stack scene now pulses the sampled neighborhood, moves feature and graph packets, and highlights BVH bounds while its panel is visible. The planned sparse stage only reveals a dashed scaffold once. The dense field is deterministic data, not random animation.
- The playground respects `prefers-reduced-motion`, pauses when offscreen or the document is hidden, and exposes play, pause, step, and reset controls. Execution and What runs respond to fine-pointer hover without requiring a scroll sequence; click, touch, and keyboard controls remain available. Rapid pointer pass-through does not switch the detail, and Execution hover suppresses repeated live announcements.
- Semantic headings, a skip link, visible focus outlines, labeled controls, textual status/results, `aria-pressed` or `aria-current` state, and English copy are part of the implementation. Canvas interaction has a keyboard path. The measurement chart supplies method names and numeric timings outside the bars, so color or length is never the only carrier of a result.
- Responsive layout puts the instrument's canvas and inspector into one column on narrow screens; sticky narratives relax to normal flow. Claims of a complete accessibility certification would require assistive-technology and browser testing beyond this source audit.

## Verification and remaining manual QA

- `npm test` passed locally on 2026-10-02 at this revision (65 tests). The router and ecosystem tests cover grouped views, deep links, history, skip-link destinations, model/phase hover intent, touch and keyboard selection, focused evidence protection, and phase-motion visibility. The suite also checks English-only markup, Morton BigInt derivation, the dense fixture, computed relationship lines, responsive Euclidean geometry, Ball Query order/boundary, bounded playback, and numerical/evidence records.
- Manual browser QA of this revision covered desktop 1500×756, 1280×720, 1024×768, 820×760, a short 1191×635 viewport, and mobile 390/320 widths. At 1191×635, the Compute field and full diagnostic dock fit above the fold, while all six Execution steps and all five What runs phase controls are visible without manual scrolling. The operator buttons sit next to Input Points and the logo appears once without a surrounding box. Narrow layouts had no document horizontal overflow. Six-view navigation, direct `#phases`, English-only copy, Compute mode changes, Execution node hover, and What runs click selection were checked. A runtime canvas error introduced while reducing decorative lines was corrected and the current build loaded without a new console error. A fresh screen-reader pass remains open.
- On the current English-only build, all six views were checked again at 390×844 with no visible Korean, document horizontal overflow, data-load error, or language switch. The Morton workbench was subsequently rechecked at 1191×635 and 390×844: the computed key rank and editable X/Y/Z coordinates now sit directly above the fully visible grid; the horizontal sample rail follows the grid. `Bit i` is a bounded numeric input (0–20). Direct coordinate entry updated the point, key, and rank; entering bit 5 updated the derivation. The mobile viewport override was reset after inspection.
- Better Design spacing inspection of the 390/1440/1728 rendered hero reported no spacing faults. Its text-only comprehension check passed the out-of-team and screenshot tests but still flags the seven visible mode/parameter/playback/path controls against a three-action landing-screen heuristic. This is an interactive scientific instrument, so those controls remain; the point-count and advanced playback controls were moved behind a disclosure. The heuristic cannot assess actual visual hierarchy.
- Reduced-motion behavior is present in CSS and the playground controller, but browser media emulation and a screen-reader pass remain to do. The Node suite cannot prove contrast, visual legibility, or assistive-technology output; this audit does not claim that certification.

## Scope

This explainer remains local and uses no remote rendering dependency. There has been no hosting, GitHub push, or publication of this draft.
