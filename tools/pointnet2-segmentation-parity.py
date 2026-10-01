#!/usr/bin/env python3
"""Keyboard-layout friendly launcher for pointnet2_segmentation_parity.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).with_name("pointnet2_segmentation_parity.py")),
    run_name="__main__",
)
