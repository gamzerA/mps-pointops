# JOSS submission readiness — 2026-10-02

This is a preparation record, not a submission or an eligibility claim. The
working `paper.md` and `paper.bib` are drafts and must not be submitted until
the gates below are satisfied and the author approves the final text.

## Identity and review format

JOSS requires author names and affiliations in the paper, a GitHub account for
the submitting major contributor, and public review through a GitHub issue.
The paper draft uses the name **YeYoung Lee** and ORCID
**0009-0001-8245-1803**, matching `CITATION.cff`. The author has confirmed
**Independent Researcher** as the affiliation; no institution or country is
inferred. Making unrelated GitHub repositories private would not make this
submission anonymous: this repository,
its citation metadata, its Zenodo archive, and JOSS's review are public.

## Pre-review gates

| JOSS gate | Evidence now | Action before submission |
| --- | --- | --- |
| More than six months of public development with ongoing iteration | GitHub API reports repository creation at **2026-09-30 18:01:16 UTC**; the oldest current commit is 2026-10-01; today is 2026-10-02. | Continue genuinely public, incremental development, releases, issues, and responses. Earliest possible calendar window is after March 2027, but elapsed time alone is insufficient. |
| Demonstrated research use | README describes 30-cloud MulSen-AD grouping with public JSON and a 45-run anomaly-detector comparison whose split, scores, and runner are outside this repository. | Publish or provide to editors inspectable research-use evidence, including the private workflow where appropriate. Seek independent user testing and document real integration. Do not count package downloads or future plans as demonstrated research use. |
| Open-source practice | Public source, Apache-2.0 and MIT notices, issue templates, CONTRIBUTING, tagged releases, tests and CI, raw benchmark records. | Maintain these over the required public period. Confirm a colleague can install and run the package, as the JOSS guidance asks of new software. |
| Feature-complete, usable research software | Released v0.8.0 provides selected Metal point-cloud operators; unreleased and experimental paths have separate scopes. | Frame the paper around a bounded, stable released capability set; make installation and representative examples reproducible on available review hardware. Do not imply full PyTorch3D/PyG or `spconv` compatibility. |

These are the JOSS [submission requirements](https://joss.readthedocs.io/en/latest/submitting.html)
and [review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html),
not a prediction of an editor's decision.

## Paper-specific work before submission

1. Confirm any conflicts of interest, funding, and sponsor role. The author has
   selected Independent Researcher as the affiliation; do not infer an
   institution or country.
2. Recheck every benchmark claim against the release being submitted, the raw
   records, and the matching source hashes. Cite a **version DOI** for the
   accepted tagged release, not only the concept DOI used in this draft.
3. Complete the AI disclosure with each tool/model version and its role. A
   private usage ledger records Claude Opus 5.5 and earlier Codex assistance,
   but not every historical Codex model version or later division of work.
   Review, edit, and validate all AI-assisted code, docs, and paper claims with
   human oversight before the author attests to JOSS's required wording.
   Agent-executed tests must not be rephrased as personally executed by the
   author. JOSS bars AI-generated conversational replies to editors/reviewers
   except translation.
4. Keep the paper within [JOSS's 750–1750 word format](https://joss.readthedocs.io/en/latest/paper.html)
   and retain its required sections: Summary, Statement of need, State of the
   field, Software design, Research impact statement, AI usage disclosure,
   Acknowledgements, and References. Move API details to project docs.
5. Compile the final paper with the [Open Journals paper toolchain](https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles)
   and review the generated PDF. This draft has not been compiled by Inara.

JOSS asks for a Git-based paper in the software repository; it permits a
short-lived paper branch. The paper draft is in this branch for revision and
does not indicate that a submission has been made.
