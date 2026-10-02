"""Failure-injection checks for the release gate, independent of Torch/MPS."""

from pathlib import Path
import io
import tarfile
import tempfile
import unittest
import zipfile

from validate_release import validate_distributions, validate_source


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {
            "pyproject.toml": '[project]\nname = "mps-pointops"\nversion = "0.8.0"\nlicense = "Apache-2.0 AND MIT"\n',
            "CITATION.cff": 'version: "0.8.0"\ndate-released: "2020-01-01"\ndoi: "10.5281/zenodo.23092167"\nrepository-code: https://github.com/gamzerA/mps-pointops\nurl: https://github.com/gamzerA/mps-pointops/releases/tag/v0.8.0\n',
            "mps_pointops/__init__.py": '__version__ = "0.8.0"\n',
            "mps_pointops/kernels/new_kernel.metal": "kernel void new_kernel() {}\n",
            "LICENSE": "Apache fixture\n",
            "LICENSES/MIT-ball-query.txt": "MIT fixture\n",
            "README.md": "Readme fixture\n",
            "CHANGELOG.md": "Changelog fixture\n",
        }
        for name, content in self.files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.dist = self.root / "dist"
        self.dist.mkdir()

    def artifacts(self, *, omit=None, stale=None, version="0.8.0"):
        metadata = f"Name: mps-pointops\nVersion: {version}\nLicense-Expression: Apache-2.0 AND MIT\n\n"
        prefix = "mps_pointops-0.8.0"
        with zipfile.ZipFile(self.dist / f"{prefix}-py3-none-any.whl", "w") as archive:
            for name, contents in self.files.items():
                if name == omit:
                    continue
                if name.startswith("mps_pointops/"):
                    archive.writestr(name, "stale" if name == stale else contents)
                elif name in ("LICENSE", "LICENSES/MIT-ball-query.txt"):
                    archive.writestr(f"{prefix}.dist-info/licenses/{name}", contents)
            archive.writestr(f"{prefix}.dist-info/METADATA", metadata)
            archive.writestr(f"{prefix}.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
            archive.writestr(f"{prefix}.dist-info/RECORD", "")
        with tarfile.open(self.dist / f"{prefix}.tar.gz", "w:gz") as archive:
            for name, contents in {**self.files, "PKG-INFO": metadata}.items():
                data = contents.encode()
                info = tarfile.TarInfo(f"{prefix}/{name}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def test_complete_release_validates_without_importing_torch(self):
        source = validate_source(self.root, "v0.8.0")
        self.artifacts()
        artifacts = validate_distributions(self.root, self.dist, source)
        self.assertEqual(len(artifacts), 2)
        self.assertTrue(all(len(a["sha256"]) == 64 for a in artifacts))

    def test_runtime_version_and_tag_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "release tag"):
            validate_source(self.root, "v1.0.0")
        (self.root / "mps_pointops/__init__.py").write_text('__version__ = "1.0.0"')
        with self.assertRaisesRegex(ValueError, "__version__"):
            validate_source(self.root)

    def test_concept_doi_rejected(self):
        citation = self.root / "CITATION.cff"
        citation.write_text(citation.read_text().replace("23092167", "23076057"))
        with self.assertRaisesRegex(ValueError, "concept DOI"):
            validate_source(self.root)

    def test_new_kernel_missing_from_wheel_rejected(self):
        self.artifacts(omit="mps_pointops/kernels/new_kernel.metal")
        with self.assertRaisesRegex(ValueError, "package file list"):
            validate_distributions(self.root, self.dist, validate_source(self.root))

    def test_stale_kernel_in_wheel_rejected(self):
        self.artifacts(stale="mps_pointops/kernels/new_kernel.metal")
        with self.assertRaisesRegex(ValueError, "stale source"):
            validate_distributions(self.root, self.dist, validate_source(self.root))

    def test_missing_mit_notice_rejected(self):
        self.artifacts(omit="LICENSES/MIT-ball-query.txt")
        with self.assertRaisesRegex(ValueError, "license"):
            validate_distributions(self.root, self.dist, validate_source(self.root))

    def test_distribution_version_drift_rejected(self):
        self.artifacts(version="1.0.0")
        with self.assertRaisesRegex(ValueError, "Version differs"):
            validate_distributions(self.root, self.dist, validate_source(self.root))

    def test_extra_importable_wheel_module_rejected(self):
        self.artifacts()
        with zipfile.ZipFile(next(self.dist.glob("*.whl")), "a") as archive:
            archive.writestr("extra_importable_module.py", "value = 1\n")
        with self.assertRaisesRegex(ValueError, "unexpected members"):
            validate_distributions(self.root, self.dist, validate_source(self.root))

    def test_artifact_filename_version_drift_rejected(self):
        for extension in ("*.whl", "*.tar.gz"):
            with self.subTest(extension=extension):
                self.artifacts()
                path = next(self.dist.glob(extension))
                wrong = path.with_name(path.name.replace("0.8.0", "9.9.9"))
                path.rename(wrong)
                with self.assertRaisesRegex(ValueError, "filename name/version"):
                    validate_distributions(self.root, self.dist, validate_source(self.root))
                wrong.rename(path)


if __name__ == "__main__":
    unittest.main()
