---
title: "mps-pointops: Metal point-cloud operators for PyTorch on Apple Silicon"
tags:
  - Python
  - Metal
  - point clouds
  - PyTorch
  - Apple Silicon
authors:
  - name: YeYoung Lee
    orcid: 0009-0001-8245-1803
    affiliation: "1"
affiliations:
  - index: 1
    name: "Independent Researcher"
date: 2 October 2026
bibliography: paper.bib
---

<!-- WORKING DRAFT. Do not submit: the repository has not yet met JOSS's
six-month public-development gate, and AI model versions,
human-validation attestation, and research-impact evidence need review. -->

# Summary

Point-cloud analysis uses sets of three-dimensional samples to represent the
surfaces of objects and scenes. Neural networks repeatedly choose representative
points, find nearby samples, and pool their features. `mps-pointops` implements
selected point-cloud operations for PyTorch tensors on Apple Silicon through
native Metal kernels. Its core operations are farthest point sampling, nearest
neighbor search, and radius-based neighbor selection. It also provides bounded
compatibility interfaces for selected PointNet++, PyTorch3D, and PyTorch
Geometric call sites, alongside CPU reference paths, tests, numerical contracts,
and benchmark scripts. The software is archived with a versioned Zenodo DOI
lineage [@mps-pointops-archive].

# Statement of need

Models such as PointNet++ use farthest point sampling and local grouping to
build increasingly abstract representations of irregular point sets
[@qi2017pointnetpp]. Researchers using Apple Silicon can run ordinary PyTorch
tensor operations on MPS, but model code depending on CUDA-specific point-cloud
extensions cannot simply route those calls to MPS. Replacing each missing
operator with a general tensor expression is possible for some cases, but can
create repeated GPU dispatches or large intermediate distance matrices. This
matters when experimentation, model reproduction, or data analysis is carried
out on Mac hardware without a separate CUDA machine. The intended users are
researchers and developers testing point-cloud and geometric-learning
pipelines on Apple Silicon. `mps-pointops` makes a documented subset of those
pipelines executable on MPS and gives users reference code and exact tests for
the supported contracts.

# State of the field

PointNet++ introduced the sampling and local-neighborhood pattern
[@qi2017pointnetpp]. PyTorch3D supplies a wider library of three-dimensional
operators and differentiable losses [@ravi2020pytorch3d]. PyTorch Geometric
supports graph and point-cloud workflows with its own operator ecosystem
[@fey2019pyg]. These projects define important APIs and mathematical behavior;
`mps-pointops` is not a replacement for their full feature sets. Its scoped
contribution is an Apple Silicon Metal implementation of selected search and
sampling paths, with explicit limits on dtype, dimensions, padding,
tie-breaking, and autograd behavior. Where upstream behavior differs, the
package keeps separate adapters rather than treating all "ball query" or
"radius" APIs as interchangeable.

The decision to develop a standalone package made it possible to test the MPS
execution and numerical contracts independently while offering adapters to
existing Python call sites. Upstream integration remains a separate objective;
the existence of the corresponding CUDA libraries is acknowledged and cited.
No claim of being the first or only Metal point-cloud implementation is made.

# Software design

The Metal FPS path keeps iterative distance updates and selection within a
single GPU dispatch for its supported input regime, avoiding a Python-to-GPU
launch for every selected point. The kNN kernel maintains a bounded candidate
set and visits spatially ordered data in a dispersed order to avoid the
measured slowdown of an order-preserving linear scan. Ball Query has a
different contract: it returns the first valid neighbors in original input
order. A SIMD ballot and prefix rank scheme lets lanes examine points in
parallel while preserving that order. These are architectural trade-offs:
changing the search order is acceptable for a distance-ranked kNN result with
deterministic tie rules, but would change Ball Query's first-neighbor result.

Numerical behavior is specified independently of a particular compiler. The
repository documents float32 radius construction, strict boundary tests,
non-finite inputs, Safe and Fast Math modes, and the distinction between exact
index agreement and tolerated floating-point differences. Tests cover
forward results and available first derivatives. The package retains a
PyTorch CPU implementation for reference and for CPU tensors, while MPS
requests fail explicitly when the required device is unavailable. A reusable
Morton-based spatial index is an opt-in development path; its benchmark results
are separated from the released v0.8.0 paths and from automatic dispatch
claims.

# Research impact statement

The repository contains a reproducible grouping example using 30 real
MulSen-AD point clouds and records per-cloud latency and selected-neighbor
comparisons. On the stated M5 Pro setup, its median MPS grouping time was
36.0 ms, compared with 163.4 ms for a composition of CPU FPS and nearest
neighbor libraries. The CPU figure excludes some grouping steps included in
the MPS figure, so these are not matched full-pipeline timings or a universal
speedup claim. Source, machine, and benchmark details
are in the repository's [real-data results](examples/results/mulsen_grouping.json)
and [README](README.md). The same repository also records synthetic
counterexamples where an alternative spatial-search path is slower. A separate
45-run anomaly-detector comparison is described in the README, but its split,
scores, and runner scripts live outside this repository; it should not be
treated as independently reproducible impact evidence until that record is
made available to editors. External research adoption has not yet been
documented.

# AI usage disclosure

Anthropic Claude Opus 5.5 through Claude Code helped draft FPS and kNN Metal
kernels, reference implementations, compatibility wrappers, tests, benchmark
tools, examples, and documentation. OpenAI Codex helped develop and port Ball
Query, investigate its floating-point behavior, and subsequently assisted
with additional code, tests, documentation, measurements, and this paper
draft. AI agents executed some reported benchmarks and validations on the
author's machines under the author's direction. The author chose the research
problem and target workflows, made design and scope decisions, and reviewed
and revised project materials. Before submission, the historical Codex model
versions and the division of work across later sessions must be reconstructed
from the private usage record, and the author must personally confirm that
all AI-assisted outputs have been reviewed, edited, and validated. The author
remains responsible for originality, licensing, numerical accuracy, and all
claims in the paper.

# Acknowledgements

Funding and any sponsor role are to be confirmed by the author before
submission. No financial-support claim is made in this draft.

# References
