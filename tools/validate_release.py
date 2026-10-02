"""Validate source/release metadata and the exact files shipped to PyPI.

Run with Python 3.11+ and PyYAML. This is an offline consistency check:
the release operator must separately verify the reserved Zenodo DOI and archive.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

import yaml


CONCEPT_DOI = "10.5281/zenodo.23076057"
REPOSITORY = "https://github.com/gamzerA/mps-pointops"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_source(root: Path, tag: str | None = None) -> dict:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    require(isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version),
            "package version must be a final MAJOR.MINOR.PATCH version")
    assignments = [node for node in ast.parse((root / "mps_pointops/__init__.py").read_text()).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    require(len(assignments) == 1, "expected exactly one __version__ assignment")
    require(ast.literal_eval(assignments[0].value) == version,
            "__version__ differs from pyproject.toml")
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    require(str(citation.get("version")) == version, "CITATION.cff version differs from package")
    expected_tag = f"v{version}"
    require(tag is None or tag == expected_tag, "release tag differs from package version")
    require(citation.get("repository-code") == REPOSITORY, "citation repository differs")
    require(citation.get("url") == f"{REPOSITORY}/releases/tag/{expected_tag}",
            "citation release URL differs from package version")
    date = dt.date.fromisoformat(str(citation.get("date-released")))
    require(date <= dt.date.today(), "citation release date is in the future")
    doi = citation.get("doi", "")
    require(isinstance(doi, str) and re.fullmatch(r"10\.5281/zenodo\.\d+", doi),
            "citation must contain a reserved Zenodo version DOI")
    require(doi != CONCEPT_DOI, "citation must use the version DOI, not the concept DOI")
    require(project["license"] == "Apache-2.0 AND MIT", "mixed license expression differs")
    return {"name": project["name"], "version": version, "tag": expected_tag,
            "doi": doi, "date": date.isoformat(), "license": project["license"]}


def validate_distributions(root: Path, dist: Path, source: dict) -> list[dict]:
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    require(len(wheels) == len(sdists) == 1,
            "dist must contain exactly one wheel and one sdist; remove stale builds")
    package_files = {p.relative_to(root).as_posix(): p.read_bytes()
                     for p in (root / "mps_pointops").rglob("*")
                     if p.is_file() and p.suffix in {".py", ".metal"}}
    licenses = {p: (root / p).read_bytes()
                for p in ("LICENSE", "LICENSES/MIT-ball-query.txt")}
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    with tarfile.open(sdists[0]) as archive:
        sdist = {}
        for member in archive.getmembers():
            if member.isfile():
                relative = member.name.split("/", 1)
                require(len(relative) == 2, "sdist member has no root directory")
                sdist[relative[1]] = archive.extractfile(member).read()
    for label, files in (("wheel", wheel), ("sdist", sdist)):
        shipped_package = {name for name in files if name.startswith("mps_pointops/")
                           and Path(name).suffix in {".py", ".metal"}}
        require(shipped_package == set(package_files), f"{label}: package file list differs from source")
        for name, contents in package_files.items():
            require(files.get(name) == contents, f"{label}: missing or stale source file {name}")
        if label == "wheel":
            metas = [data for name, data in files.items() if name.endswith(".dist-info/METADATA")]
        else:
            metas = [files.get("PKG-INFO", b"")]
        require(len(metas) == 1 and bool(metas[0]), f"{label}: expected one package metadata file")
        meta = BytesParser().parsebytes(metas[0])
        for field, expected in (("Name", source["name"]), ("Version", source["version"]),
                                ("License-Expression", source["license"])):
            require(meta.get(field) == expected, f"{label}: {field} differs from source")
        for name, contents in licenses.items():
            if label == "wheel":
                matches = [data for path, data in files.items()
                           if path.endswith(f".dist-info/licenses/{name}")]
                require(matches == [contents], f"wheel: missing or stale license {name}")
            else:
                require(files.get(name) == contents, f"sdist: missing or stale license {name}")
    for name in ("pyproject.toml", "CITATION.cff", "README.md", "CHANGELOG.md"):
        require(sdist.get(name) == (root / name).read_bytes(), f"sdist: missing or stale {name}")
    return [{"file": path.name, "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (wheels[0], sdists[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", help="GitHub release tag, when validating a release")
    parser.add_argument("--dist", type=Path, help="also validate a built wheel and sdist")
    parser.add_argument("--output", type=Path, help="write the validated metadata and artifact hashes")
    args = parser.parse_args()
    try:
        result = validate_source(args.root, args.tag)
        if args.dist is not None:
            result["artifacts"] = validate_distributions(args.root, args.dist, result)
    except (ValueError, KeyError, SyntaxError, TypeError) as exc:
        parser.exit(1, f"Release validation failed: {exc}\n")
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
