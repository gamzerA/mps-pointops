"""Render the original operator diagram and the measured README chart.

The benchmark chart reads the committed JSON files. No downloaded artwork or
third-party fonts are used; run with Python from any working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
BASELINE = ROOT / "bench" / "results" / "2026-10-01-apple-m5-pro.json"
BALL = ROOT / "bench" / "results" / "2026-10-01-apple-m5-pro-ball-query-port.json"


def benchmark_ms(path: Path, op: str, implementation: str) -> float:
    rows = json.loads(path.read_text())["rows"]
    matches = [
        row["median_ms"]
        for row in rows
        if row["op"] == op
        and row["n_points"] == 100_000
        and row["impl"] == implementation
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {op}/{implementation} row in {path}")
    return matches[0]


def svg_open(title: str, description: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{title}</title>',
        f'<desc id="description">{description}</desc>',
        '<defs>',
        '<pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#9fb9d4" stroke-opacity=".07"/></pattern>',
        '<linearGradient id="base" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#0b1421"/><stop offset="1" stop-color="#172438"/></linearGradient>',
        '</defs>',
        f'<rect width="{width}" height="{height}" rx="18" fill="url(#base)"/>',
    ]


def write_svg(name: str, elements: list[str]) -> None:
    (ASSETS / name).write_text("\n".join([*elements, "</svg>"]) + "\n")


def render_hero() -> None:
    lines = svg_open(
        "Three point-cloud operators on Apple Silicon",
        "Farthest point sampling selects spread-out centers; k nearest neighbors ranks by distance; Ball Query keeps the first K points inside a radius.",
        1200,
        276,
    )
    panels = [
        (16, "01", "FPS", "select spread-out centers", "#61d6c2"),
        (412, "02", "kNN", "rank by squared distance", "#80aaff"),
        (808, "03", "BALL QUERY", "first K inside the radius", "#f2ba73"),
    ]
    for left, number, name, rule, color in panels:
        lines.extend(
            [
                f'<rect x="{left}" y="16" width="376" height="244" rx="12" fill="#142237" stroke="#30445e" stroke-width="1"/>',
                f'<rect x="{left+1}" y="17" width="374" height="242" rx="11" fill="url(#grid)"/>',
                f'<rect x="{left+17}" y="34" width="30" height="28" rx="5" fill="{color}" fill-opacity=".15"/>',
                f'<text x="{left+23}" y="54" fill="{color}" font-family="ui-monospace,SFMono-Regular,monospace" font-size="19" font-weight="700">{number}</text>',
                f'<text x="{left+60}" y="55" fill="#f4f8ff" font-family="system-ui,-apple-system,sans-serif" font-size="27" font-weight="750">{name}</text>',
                f'<path d="M{left+18} 73H{left+358}" stroke="#5d7693" stroke-opacity=".4"/>',
                f'<text x="{left+19}" y="240" fill="#c4d2e2" font-family="system-ui,-apple-system,sans-serif" font-size="22">{rule}</text>',
            ]
        )

    # FPS: two centers are already selected; the outlined candidate is the next.
    fps_points = [(71, 139), (94, 103), (119, 160), (151, 119), (174, 187),
                  (207, 91), (225, 158), (253, 115), (281, 179), (307, 102),
                  (339, 153), (329, 201)]
    for x, y in fps_points:
        lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#7993ad"/>')
    lines.extend([
        '<path d="M329 201L253 115" stroke="#61d6c2" stroke-opacity=".65" stroke-width="1.7" stroke-dasharray="5 6"/>',
        '<circle cx="94" cy="103" r="11" fill="#61d6c2" fill-opacity=".13" stroke="#61d6c2" stroke-width="2"/>',
        '<circle cx="253" cy="115" r="11" fill="#61d6c2" fill-opacity=".13" stroke="#61d6c2" stroke-width="2"/>',
        '<circle cx="329" cy="201" r="15" fill="#61d6c2" fill-opacity=".12" stroke="#61d6c2" stroke-width="2.5"/>',
        '<path d="M310 188L298 174" stroke="#61d6c2" stroke-width="1.5"/>',
        '<text x="92" y="207" fill="#8ce6d6" font-family="ui-monospace,SFMono-Regular,monospace" font-size="19">max min d²</text>',
    ])

    # kNN: the three selected neighbors are visibly linked to the query.
    knn_points = [(469, 109), (494, 177), (529, 94), (559, 126), (570, 181),
                  (609, 83), (635, 145), (671, 94), (705, 177), (749, 129)]
    for x, y in knn_points:
        lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#7993ad"/>')
    for rank, x, y in [(1, 635, 145), (2, 570, 181), (3, 559, 126)]:
        lines.extend([
            f'<path d="M609 146L{x} {y}" stroke="#80aaff" stroke-width="2" stroke-opacity=".85"/>',
            f'<circle cx="{x}" cy="{y}" r="10" fill="#80aaff"/>',
            f'<text x="{x-4}" y="{y+5}" fill="#142237" font-family="ui-monospace,SFMono-Regular,monospace" font-size="13" font-weight="700">{rank}</text>',
        ])
    lines.extend([
        '<circle cx="609" cy="146" r="9" fill="#f4f8ff" stroke="#80aaff" stroke-width="2"/>',
        '<text x="621" y="204" fill="#a9c3ff" font-family="ui-monospace,SFMono-Regular,monospace" font-size="19">query → top-k</text>',
    ])

    # Ball Query: input indices 0, 2, 4 win; index 7 is inside but beyond K=3.
    lines.extend([
        '<circle cx="1000" cy="148" r="66" fill="#f2ba73" fill-opacity=".07" stroke="#f2ba73" stroke-width="2" stroke-dasharray="6 6"/>',
        '<circle cx="1000" cy="148" r="7" fill="#f4f8ff"/>',
        '<text x="1102" y="107" fill="#f2ba73" font-family="ui-monospace,SFMono-Regular,monospace" font-size="19">K = 3</text>',
    ])
    for x, y, index in [(968, 121, "0"), (1030, 140, "2"), (981, 182, "4")]:
        lines.extend([
            f'<circle cx="{x}" cy="{y}" r="12" fill="#f2ba73"/>',
            f'<text x="{x-4}" y="{y+5}" fill="#172438" font-family="ui-monospace,SFMono-Regular,monospace" font-size="14" font-weight="700">{index}</text>',
        ])
    lines.extend([
        '<circle cx="1018" cy="102" r="10" fill="#7993ad" fill-opacity=".4" stroke="#7993ad"/>',
        '<text x="1014" y="107" fill="#d6e0ed" font-family="ui-monospace,SFMono-Regular,monospace" font-size="13">7</text>',
        '<circle cx="1087" cy="179" r="4" fill="#7993ad"/>',
    ])
    write_svg("pointops-hero.svg", lines)


def render_speedups() -> None:
    rows = [
        ("FPS", "1,024 samples", benchmark_ms(BASELINE, "fps", "mps-pointops Metal (MPS)"),
         benchmark_ms(BASELINE, "fps", "fpsample vanilla (CPU)"), "#61d6c2"),
        ("kNN", "1,024 queries · k=128", benchmark_ms(BASELINE, "knn", "mps-pointops Metal (MPS)"),
         benchmark_ms(BASELINE, "knn", "scipy cKDTree build+query (CPU)"), "#80aaff"),
        ("Ball Query", "1,024 queries · K=64", benchmark_ms(BALL, "ball_query", "mps-pointops Metal (MPS)"),
         benchmark_ms(BALL, "ball_query", "scipy cKDTree build+query (CPU)"), "#f2ba73"),
    ]
    lines = svg_open(
        "M5 Pro speedups against tested CPU libraries",
        "At 100,000 randomly ordered reference points: FPS 5.2 times, kNN 2.9 times, and Ball Query 1.6 times faster than the fastest tested CPU implementations.",
        1200,
        375,
    )
    lines.extend([
        '<text x="54" y="55" fill="#f4f8ff" font-family="system-ui,-apple-system,sans-serif" font-size="35" font-weight="750">Measured speedup on M5 Pro</text>',
        '<text x="56" y="83" fill="#b5c5d7" font-family="system-ui,-apple-system,sans-serif" font-size="21">100k reference points · random order · median of 5 · vs tested CPU library</text>',
        '<path d="M56 101H1144" stroke="#5d7693" stroke-opacity=".45"/>',
    ])
    for number, (name, setup, metal, cpu, color) in enumerate(rows):
        top = 118 + number * 82
        speedup = cpu / metal
        width = 105 * speedup
        lines.extend([
            f'<text x="56" y="{top+22}" fill="#f4f8ff" font-family="system-ui,-apple-system,sans-serif" font-size="29" font-weight="700">{name}</text>',
            f'<text x="56" y="{top+46}" fill="#a9bed3" font-family="system-ui,-apple-system,sans-serif" font-size="19">{setup}</text>',
            f'<rect x="347" y="{top+2}" width="630" height="28" rx="6" fill="#25344a"/>',
            f'<rect x="347" y="{top+2}" width="{width:.1f}" height="28" rx="6" fill="{color}"/>',
            f'<text x="{361+width:.1f}" y="{top+25}" fill="{color}" font-family="system-ui,-apple-system,sans-serif" font-size="29" font-weight="750">{speedup:.1f}×</text>',
            f'<text x="349" y="{top+57}" fill="#bbcadb" font-family="ui-monospace,SFMono-Regular,monospace" font-size="18">Metal {metal:.1f} ms  /  CPU {cpu:.1f} ms</text>',
        ])
    write_svg("m5-pro-speedup.svg", lines)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    render_hero()
    render_speedups()
    print("Wrote docs/assets/pointops-hero.svg and docs/assets/m5-pro-speedup.svg")


if __name__ == "__main__":
    main()
