// Page-level prose stays separate from model claims and measured JSON.
const COPY = [
  [".primary-nav a[href=\"#compute\"]", "Compute"],
  [".primary-nav a[href=\"#route\"]", "Execution"],
  [".primary-nav a[href=\"#capabilities\"]", "What runs"],
  [".primary-nav a[href=\"#methods\"]", "Methods"],
  [".primary-nav a[href=\"#benchmarks\"]", "Measurements"],
  [".primary-nav a[href=\"#quickstart\"]", "Install"],
  ["#hero-title", "mps-pointops · Compute"],
  [".obs-field-hint", "POINTER OR ARROWS → QUERY / FPS START"],
  [".instrument-toolbar-label strong", "One cloud, three questions"],
  [".scene-key span:nth-child(1) b", "Input"],
  [".scene-key span:nth-child(2) b", "Query"],
  [".scene-key span:nth-child(3) b", "Selected"],
  [".inspector-intro", "Fixed 2D teaching model. This is not Metal execution or a numerical parity test."],
  [".count-control b", "Point count"],
  [".k-control b", "Selection count k"],
  [".radius-control b", "Radius r"],
  [".playground-more summary", "Point count · playback"],
  ["[data-query-label]", "Query position / starting point"],
  [".instrument-footer span:first-child", "Move the query or change the operator and parameters."],
  [".obs-evidence p", "This scene explains selection. Measurements below were recorded separately under named hardware and input conditions."],
  [".obs-evidence a", "Inspect measurements ↗"],
  [".first-note > span", "These rules lead into PyTorch/MPS execution."],
  [".first-note > a", "Open execution →"],
  ["#route-title", "PyTorch → MPS → Metal"],
  ["#capabilities-title", "Scope"],
  ["#phases-title", "Implementation stages"],
  ["#trust-title", "Speed alone does not make the same operator."],
  ["#trust .section-heading p", "We check outputs, indices, gradients, device path, and measurement conditions together."],
  [".trust-columns > div:nth-child(1) h3", "Compare selected results"],
  [".trust-columns > div:nth-child(1) > p:not(.trust-metric)", "Check indices, distances, gradients, padding, and boundary rules. Safe and Fast Math run in separate processes."],
  [".trust-metric span", "M5 Pro Safe passes / research-source log"],
  [".trust-columns > div:nth-child(2) h3", "Name only tested model paths"],
  [".trust-columns > div:nth-child(2) p", "Synthetic PointNet++ and DGCNN runs are distinct from bounded PyG and Pointcept paths. No real-data accuracy or universal compatibility claim."],
  [".trust-columns > div:nth-child(2) a", "Inspect model scope ↗"],
  [".trust-columns > div:nth-child(3) h3", "Measure the same workload"],
  [".trust-columns > div:nth-child(3) p", "Hardware, N, distribution, synchronization, and build cost travel with every result. Different CPU/GPU workloads are never presented as a speedup."],
  [".trust-columns > div:nth-child(3) a", "Inspect measurements ↗"],
  [".truth-strip > span", "Numerical contracts"],
  [".truth-strip > p", "Ball Query takes the first K hits in input order. kNN ranks distance and resolves ties by index. Runtime depends on hardware and point distribution."],
  [".truth-strip > a", "Read numerical contracts ↗"],
  ["#methods-title", "Spatial index computation"],
  ["#methods .section-heading p", "The two scenes below are 2D teaching models, not renders of released GPU kernels."],
  ["#benchmarks-title", "Read a measurement only with its conditions."],
  ["#benchmarks .section-heading p", "Only hardware, operator, and point-count combinations in recorded JSON appear. An empty cell remains unmeasured."],
  ["#quickstart-title", "Install and run"],
  [".install-main > p", "Place these two lines before CUDA-oriented imports to enable the compatibility path. Registered namespaces and coverage vary by operation."],
  [".install-aside > p", "A CPU reference path may run without MPS. Metal execution requires Apple Silicon and an MPS-enabled PyTorch build."],
  [".install-aside > a", "Installation and support policy ↗"],
  ["#provenance-title", "Show the boundaries with the evidence."],
  ["#provenance .section-heading p", "Release, research source, and test logs have distinct provenance."],
  [".provenance-grid > div:nth-child(1) p", "The v1.0.0 package includes bounded spatial and Chamfer APIs. Private sparse experiments remain outside the public compatibility promise."],
  [".provenance-grid > div:nth-child(1) a", "Version DOI ↗"],
  [".provenance-grid > div:nth-child(2) p", "Released source and historical measurements retain separate commit provenance."],
  [".provenance-grid > div:nth-child(2) a", "View source ↗"],
  [".provenance-grid > div:nth-child(3) p", "Separate Safe/Fast runs on M5 Pro research source, including skips and expected failures."],
  [".open-limits h3", "Still to verify"],
  [".open-limits li:nth-child(1)", "Broader workloads beyond the recorded M1 spatial fixtures; physical M2–M4 matrix"],
  [".open-limits li:nth-child(2)", "Exact physical GPU peak and runtime Occupancy; M5 allocation traces are recorded"],
  [".open-limits li:nth-child(3)", "Real-data model accuracy and broad Pointcept/Sparse 3D compatibility"],
  [".site-footer .footer-grid > div:first-child p", "Explaining point-cloud computation with a bounded evidence record."],
  [".site-footer .footer-grid > div:last-child p:last-child", "A paper is in preparation; no submission or publication is claimed."],
];

export function applyPageCopy() {
  for (const [selector, copy] of COPY) {
    const element = document.querySelector(selector);
    if (element) element.textContent = copy;
  }
  document.querySelector('.primary-nav')?.setAttribute('aria-label', 'Pages');
  document.querySelector('.instrument-inspector')?.setAttribute('aria-label', 'Operator controls and result');
  document.querySelector('.operator-tabs')?.setAttribute('aria-label', 'Point-cloud operator');
  document.querySelectorAll('.copy-button').forEach(button => {
    button.textContent = 'Copy';
  });
}
