# mps-pointops interactive explainer

This directory contains the dependency-free [mps-pointops](https://github.com/gamzerA/mps-pointops) interactive explainer. After GitHub Pages is enabled with **GitHub Actions** as its source, `.github/workflows/explainer-pages.yml` verifies and publishes this directory at `https://gamzerA.github.io/mps-pointops/` on a `main` push. The page fetches its own JSON, JavaScript, CSS, and evidence files only; it loads no external fonts, scripts, telemetry, or benchmark data. External links open source documents and archives only when selected.

## Run locally

From this directory:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/`. Use HTTP: browsers may block module and JSON requests from a `file://` URL.

## Navigate the explainer

Six compact buttons in the top bar open **Compute, Execution, What runs, Methods, Measurements, and Install**. The site is provided in English. Clicking a view changes the URL hash, so direct links and browser back/forward work. The repository logo appears once in the header. Execution steps and the two What runs panels can be selected without manual scrolling; longer methods and evidence remain readable within their own views. The top bar switches views immediately so it does not move a hovered target during a page change.

1. **Compute:** The dark observatory opens on 1,200 deterministic input points. Change FPS, kNN, and Ball Query, then adjust `k`, radius, point count, query, or FPS start. Every relationship ray represents a distance comparison or a nearest-selected-point assignment from the browser teaching model. The frame counter, comparison count, selected indices, and rule update with the same calculation. Open “Point count · playback” for the point slider and manual step/reset controls. Equal pixel scales keep the radius circular. The scene is not Metal footage, a GPU benchmark, or a float32 parity oracle.
2. **Execution path:** Six compact controls select the conceptual CUDA-only blockage and the PyTorch → compatibility layer → MPS dispatch → Metal kernel → output path within one screen. A mouse can preview a stage from either its card or its diagram node after a short dwell; leaving restores the last clicked stage. Click or keyboard activation commits it. This is an architectural explanation, not a traced runtime call.
3. **Ecosystem and phases:** Click the two What runs tabs to switch between model validation and the compute stack. Hover or select PointNet++, PyTorch3D, PyG, DGCNN, Pointcept, or Sparse 3D to see the precise released API, synthetic/model fixture, forward and backward status, tested hardware, and remaining limits. Hover or select one of five compact phase controls to add a computational layer. Its scene animates the selected operation only while visible; the planned layer shows a one-time dashed scaffold. Reduced-motion preferences disable these effects. Touch uses normal taps.
4. **Methods and measurements:** The Morton and AABB views remain interactive 2D teaching models. Drag the horizontal Morton sample rail, move the query point, and step through the formula derivations. The Measurements view presents recorded fixtures in a visible list and groups validation rules, the raw-record-backed benchmark explorer, and provenance. Empty selections mean there is no saved measurement. Install is its own view with the opt-in compatibility example.

## Evidence boundaries

- `data/benchmarks.json` holds release metadata, the bounded source-only spatial records used by provenance tests, Safe/Fast test-log counts, and method-demo configuration. `data/capabilities.json` holds the model/operator support statements. `data/benchmark-explorer.json` holds the 17 measured benchmark fixtures displayed in the explorer. The English dictionary and module-local copy supply explanatory text.
- The explorer separates **measurements of code later included in the published v0.8.0 release** from **unreleased v0.9 source measurements**. The former are historical source snapshots, not a fresh timing of a reinstalled v0.8.0 wheel. The M5 Pro dense 20k/100k raw record has file SHA-256 values and a Git archive commit but no recorded source commit. Its source provenance is labeled accordingly.
- A bar chart compares implementations only within one raw fixture and timing scope. It does not treat SciPy cKDTree build-plus-query as an equal-workload speedup baseline for MPS query-only measurements. Each record shows inputs, device, math mode, repetition count, timing exclusions, parity, limitations, and a bundled raw JSON link. The M1 FPS 500k/1M measurements compare two Metal implementations on the same physical M1, not M1 against M5 Pro.
- The Measurements hardware switch shows six physical M1 8 GB records from source commit `ef07a9337b40065c18a6c950bda4240c789f31f1`: kNN and radius query, each on uniform, dense-cluster/sparse-background, and collapsed 1M-point fixtures with 65,536 queries. Bars are medians of three synchronized steady queries on preloaded MPS inputs. They exclude index build, transfer, and shader compilation. On M1, the public `auto` decision selected **scan in all six records**; the BVH timings are from an explicit opt-in path. The three kNN records have zero index and distance mismatches versus native MPS scan; the three radius records additionally have zero squared-distance bit and original-order mismatches. The collapsed kNN fixture is a counterexample: BVH is slower than scan.
- A separate M1 memory record measures a 59.70 MiB PyTorch tensor allocator peak and a 1,035.84 MiB *sampled* driver high on the uniform 1M-point BVH path. Neither is a physical GPU peak; Instruments capture remains open. M5 Pro spatial timings use source commit `62fcc0f295e127be47a475fbc122d4de686b4c58`; the physical M1 records use `ef07a9337b40065c18a6c950bda4240c789f31f1`. These source paths are absent from the published PyPI `0.8.0` package.
- The older `256.175 → 6.669 ms` Morton-grid prototype result is retained in `data/benchmarks.json` and its raw evidence for provenance, but it is not a matched benchmark claim in the new explorer. Timed SciPy returned all radius neighbors while timed MPS stopped at the first `K=16`, and its total is the sum of separate stage medians.

The page's BigInt Morton example uses 21 bits per axis because JavaScript `Number` cannot exactly represent every 63-bit key. The browser AABB model uses 2D, 16-point bricks; the research GPU BVH uses 3D, 128-point bricks. Neither demo reproduces Metal float32 boundary arithmetic or GPU scheduling.

## Verify

```sh
npm test
```

Node tests cover the six-view router and deep links, dense field fixture and relationship rays, FPS/kNN/Ball Query selection and boundaries, responsive coordinate geometry, BigInt Morton interleaving, browser BVH/full-scan parity, narrative/controller and capability states, English copy references, bundled evidence, and benchmark fixture provenance. With a source checkout at `/private/tmp/mps-pointops-v090-work`, the provenance tests also compare recorded medians, metadata, pytest summaries, Git blob SHA-256 values at the recorded commits, and bundled raw files. Set `MPS_POINTOPS_SOURCE_ROOT` to use a different checkout. The M1 files may also be supplied in their recorded source path or through `MPS_POINTOPS_M1_RESULTS` (default `/private/tmp/mps-v090-m1-results`). Source-dependent tests skip if no checkout is present; the standalone tests still run.

For a visual and interaction check, run the server and review the first screen, all three operation tabs, the Execution and phase selectors, both What runs panels, every capability, a benchmark empty state, the English copy, keyboard controls, narrow/mobile layouts, and `prefers-reduced-motion`. The [design audit](DESIGN_AUDIT.md) records what was checked in the browser and what remains. Passing Node tests alone does not establish visual, screen-reader, or cross-browser quality.

## Project map

| File | Role |
| --- | --- |
| `index.html`, `src/styles.css`, `src/observatory-theme.css`, `src/interaction-motion.css`, `src/app.js` | Semantic view structure, dark observatory visual system, entry/state motion, data loading, telemetry, and interaction wiring |
| `src/view-router.js`, `assets/pointops-mark.svg` | URL-hash view navigation and the repository's existing five-point M logo |
| `src/point-playground.js`, `src/compute-field.js` | Pure FPS/kNN/Ball Query selection model and the dense, responsive first-screen field |
| `src/narratives.js`, `src/execution-layout.css` | Directly selectable execution route and progressive Phase 1–5 scene |
| `src/ecosystem-panels.js`, `src/ecosystem.css` | What runs panel chooser and compact model/phase layouts |
| `src/capability-map.js`, `data/capabilities.json` | Bounded ecosystem map and evidence links |
| `src/benchmark-explorer.js`, `data/benchmark-explorer.json` | Filterable, raw-record-backed performance view |
| `src/lib/`, `src/formula-animation.js` | Morton coding, deterministic points, 2D BVH, and optional formula walk-through |
| `data/benchmarks.json`, `evidence/` | Release/source metadata, verification summaries, and local raw records |
| `data/i18n/en.json`, `src/page-copy.js` | English explanatory copy |
| `tests/` | Numerical, content, controller, and provenance checks |

The Pages workflow uploads a static bundle containing `index.html`, `assets/`, `data/`, `evidence/`, and runtime `src/` files. It omits the local tests and design notes. The publication does not change the package release: the install command and DOI still refer to v0.8.0, while the M1/M5 spatial-index results are explicitly labeled as measurements of unreleased source. Physical GPU peak memory remains unmeasured.

The repository source is Apache-2.0, with the identified Ball Query portions under MIT; see the root `LICENSE` and `LICENSES/MIT-ball-query.txt` for their exact scopes.
