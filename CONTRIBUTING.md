# Contributing to mps-pointops

Thanks for helping improve point-cloud operations on Apple Silicon. The
[README](README.md) describes the supported APIs, numerical contracts, and
current limits. Please keep compatibility and performance claims tied to a
specific input, device, and version.

## Report a bug or propose a change

Search [existing issues](https://github.com/gamzerA/mps-pointops/issues) and
pull requests before opening a new issue. Use an issue to describe a bug,
missing operator, compatibility request, or performance regression. Choose the
bug report or feature request form; both require the Apple chip, macOS version,
and PyTorch version. Include:

- The expected result, the actual result, and a small runnable reproducer.
- The mps-pointops, Python, PyTorch, macOS, and (if relevant) PyG and pyg-lib
  versions; the Apple chip; the input shape, dtype, batch layout, and point
  order.
- Whether the failure occurs with Metal Safe Math, Fast Math, or both, plus
  the traceback or test output. Say if MPS was unavailable and tests skipped.
- For performance reports, the benchmark command, warmup and synchronization
  method, and whether device transfers or a CPU search index build were timed.

Remove private data and credentials from reproductions and logs. For a small
documentation fix, you can open a pull request directly.

## Submit a pull request

1. Create a branch from the latest `main` and keep the change focused. Link a
   relevant issue in the pull request when one exists.
2. Explain the behavior changed, the supported inputs, and any known
   limitations. For a kernel or compatibility change, document its selection
   rule, numerical behavior, and a meaningful regression case. Cite the source
   of borrowed ideas or equations; identify any reused code and preserve its
   license notice.
3. Run the relevant local tests below. In the pull request, report the exact
   commands and results, including skipped tests and any configuration you
   could not run locally.
4. Open the pull request against `main`. Direct pushes to `main` are blocked;
   merging requires a pull request and all seven required CI checks to pass.

The seven required checks in the active
[`main` ruleset](https://github.com/gamzerA/mps-pointops/rules/24270544) are:

1. `test (macos-latest, Python 3.10, torch latest)`
2. `test (macos-latest, Python 3.12, torch latest)`
3. `test (macos-latest, Python 3.10, torch 2.7.0)`
4. `test (ubuntu-latest, Python 3.12, torch latest)`
5. `package`
6. `PyG 2.8 MPS operators`
7. `PyTorch3D Chamfer MPS parity`

## Run tests locally

On an Apple Silicon Mac, use a Python version supported by PyTorch (Python
3.12 is exercised in CI). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install . pytest numpy

# Metal Safe Math: one pytest process.
PYTORCH_ENABLE_MPS_FALLBACK=0 pytest -ra tests

# Metal Fast Math: a separate pytest process.
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 pytest -ra tests
```

Fast Math is selected per process, so run the two commands separately. CI
uses `pytest` rather than `python -m pytest` so the tests import the installed
package and its bundled Metal files. On a machine without MPS, MPS-specific
tests skip; the macOS CI jobs provide GPU coverage. Linux CI also runs the
CPU tests and builds the source distribution and wheel.

For changes to the PyG 2.8 bridge, install the same pinned integration stack
used by CI, then run its tests in separate Safe and Fast processes:

```bash
python -m pip install "torch==2.12.0" "torch-geometric==2.8.0"
python -m pip install --no-index \
  --find-links 'https://data.pyg.org/whl/torch-2.12.0+cpu.html' \
  'pyg-lib==0.7.0+pt212'

PYTORCH_ENABLE_MPS_FALLBACK=0 pytest -ra tests/test_pyg28.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 pytest -ra tests/test_pyg28.py
```

The PyG integration job explicitly fails if MPS is unavailable. See
[the CI workflow](.github/workflows/ci.yml) for the authoritative dependency
versions and commands if they change.

## Release archival and DOI lineage

This project prioritizes one Zenodo DOI lineage. The
[v0.7.0 archive](https://zenodo.org/records/23087369) has version DOI
`10.5281/zenodo.23087369`. The
[v0.6.0 archive](https://zenodo.org/records/23086417) has version DOI
`10.5281/zenodo.23086417`. The
[v0.5.0 archive](https://zenodo.org/records/23080506) has version DOI
`10.5281/zenodo.23080506`. The
[v0.4.0 archive](https://zenodo.org/records/23078860) has version DOI
`10.5281/zenodo.23078860`; the earlier
[v0.3.0 archive](https://zenodo.org/records/23076058) has version DOI
`10.5281/zenodo.23076058`. All five belong to concept DOI
`10.5281/zenodo.23076057`.
The README badge uses the concept DOI; papers and `CITATION.cff` cite the
specific release's version DOI.

For each release, use **New version** from the latest Zenodo record,
following [Zenodo's versioning guide](https://help.zenodo.org/docs/deposit/manage-versions/):

1. Create a new version draft from the existing record and reserve its DOI.
   Before publishing, check that it remains under concept DOI
   `10.5281/zenodo.23076057`.
2. Update the package version and `CITATION.cff` version, date, release URL,
   and reserved version DOI through a pull request. Merge only after the seven
   required checks pass, then create the Git tag, GitHub release, and PyPI
   distribution from the same commit.
3. Archive that exact tag as one source ZIP. Compare its files with the Git
   tree and verify its checksum. Set the Zenodo version, author/ORCID, release
   links, and component licenses; keep the MIT Ball Query notice in the ZIP
   and explain the mixed license in the description. Publish the Zenodo draft
   after verifying the preview.
4. Confirm the public version DOI, concept DOI, uploaded ZIP checksum, and
   GitHub/PyPI release identity. Put the version DOI in the GitHub release
   notes. The README concept DOI badge stays unchanged.

The required `package` CI check and the publishing workflow run the same
offline release validator. It checks `pyproject.toml`, the package
`__version__`, and `CITATION.cff` for matching versions, release URLs, and a
version DOI. It compares every packaged Python/Metal file and both license
notices byte-for-byte with the checkout, then records wheel and sdist SHA-256
hashes in a `release-validation` workflow artifact. Reproduce it using
Python 3.11 or newer in a clean build directory:

```bash
python -m pip install build twine PyYAML
python -m unittest discover -s tools -p test_validate_release.py
python -m build
python -m twine check dist/*
python tools/validate_release.py --dist dist --output release-validation.json
```

For a release, also pass `--tag vX.Y.Z` using its exact Git tag. This offline
check does not verify DOI reservation/publication or the uploaded Zenodo ZIP;
complete the archive checks above before reporting a release as archived.

GitHub auto-archiving is not enabled for this repository. Zenodo's GitHub
integration documentation does not guarantee that an automatically captured
release will join this existing, manually created concept DOI lineage. Do not
enable it for a release until a verified migration plan preserves the citation
history. See [Zenodo's DOI guidance](https://zenodo.org/help/versioning).

## If required CI fails because of an external outage

First inspect the failed job's log. Distinguish a code or test failure from
an unavailable macOS runner, GitHub Actions incident, or external package
wheel/index outage. Do not bypass a reproducible code failure.

1. Re-run the failed jobs from the Actions page, or run
   `gh run rerun RUN_ID --failed`. A re-run uses the original commit and ref;
   it does not test new code. See [GitHub's re-run guide](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).
2. If the external problem persists, link the failed run and relevant outage
   evidence in the pull request. Add the local test commands and results. Ask
   a repository administrator to review the exception; contributors cannot
   disable the rule themselves.
3. If merging cannot wait, the administrator may **temporarily remove only
   the affected required check** from the `main` ruleset. Keep the pull
   request requirement and every unaffected check in force. Record the reason
   and the exact rule change in the pull request, confirm the other checks
   passed, then merge through the pull request.
4. Immediately restore the removed check after merging. Read back the ruleset
   and verify that it is Active and all seven checks above are required again.
   Record the restoration in the pull request.

As a last resort, an administrator can temporarily set the entire ruleset to
Disabled, merge the already-reviewed pull request, and immediately set it back
to Active. **Disabling the entire ruleset also removes its pull request,
force-push, and deletion protections for that interval.** Record the start,
reason, and restoration, and verify all seven required checks after re-enabling.
The narrower check-level exception above is preferred. GitHub documents
[ruleset editing](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/managing-rulesets-for-a-repository).

## Keep required check names in sync

GitHub matches required checks by their reported job names. If a job name in
[`ci.yml`](.github/workflows/ci.yml) or
[`chamfer-upstream-parity.yml`](.github/workflows/chamfer-upstream-parity.yml)
changes, coordinate the matching required-check update with an administrator
before merging the rename pull request, and update the seven-name list in this
file. Once the new job reports success on that pull request, replace its old
required context;
do not leave both names required unless both jobs actually run. Otherwise,
the old name can remain Pending and block merging. Verify that all seven
required jobs appear and pass before merging. See
[GitHub's required-check troubleshooting guide](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/troubleshooting-rules).
