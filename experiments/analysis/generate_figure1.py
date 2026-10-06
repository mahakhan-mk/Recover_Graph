"""Generate the deterministic RecoverGraph pipeline diagram as SVG."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def generate_svg() -> str:
    nodes = [
        ("a1", 20, 95, 135, "Earlier Executions"),
        ("a2", 185, 95, 120, "FailureEpisode"),
        ("a3", 335, 95, 150, "Resolution / Concrete Recovery"),
        ("a4", 515, 95, 135, "Objective Evaluator"),
        ("a5", 680, 95, 155, "Outcome-Verified Evidence"),
        ("a6", 865, 95, 135, "Recovery Abstraction"),
        ("a7", 1030, 95, 120, "RecoveryPattern"),
        ("a8", 1180, 95, 105, "Embed + Index"),
        ("a9", 1315, 95, 145, "Frozen Recovery Memory"),
        ("t1", 40, 345, 145, "New Agent Execution"),
        ("t2", 225, 345, 140, "Planned Tool Action"),
        ("t3", 405, 345, 155, "Semantic Candidate Retrieval"),
        ("t4", 600, 345, 190, "Chronology / Verification / Applicability Checks"),
        ("t5", 870, 285, 115, "Advice"),
        ("t6", 1030, 285, 165, "Agent Reconsideration"),
        ("t7", 1240, 285, 125, "Tool Action"),
        ("t8", 870, 415, 115, "Abstain"),
        ("t9", 1030, 415, 165, "Tool Action"),
    ]
    edges = [
        ("a1", "a2"),
        ("a2", "a3"),
        ("a3", "a4"),
        ("a4", "a5"),
        ("a5", "a6"),
        ("a6", "a7"),
        ("a7", "a8"),
        ("a8", "a9"),
        ("t1", "t2"),
        ("t2", "t3"),
        ("t3", "t4"),
        ("t4", "t5"),
        ("t4", "t8"),
        ("t5", "t6"),
        ("t6", "t7"),
        ("t8", "t9"),
    ]
    boxes = {key: (x, y, w, 52) for key, x, y, w, _ in nodes}
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="540" '
        'viewBox="0 0 1500 540" role="img" aria-labelledby="title desc">',
        '<title id="title">RecoverGraph recovery pipeline</title>'
        '<desc id="desc">Acquisition and frozen recovery memory, followed by transfer-time '
        'retrieval and advice or abstention.</desc>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" '
        'orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#475569"/>'
        '</marker></defs>',
        '<rect width="1500" height="540" fill="#ffffff"/>',
        '<text x="20" y="35" font-family="Arial,sans-serif" font-size="23" '
        'font-weight="700" fill="#0f172a">RecoverGraph</text>',
        '<text x="20" y="70" font-family="Arial,sans-serif" font-size="15" '
        'font-weight="700" fill="#334155">ACQUISITION</text>',
        '<text x="20" y="255" font-family="Arial,sans-serif" font-size="15" '
        'font-weight="700" fill="#334155">TRANSFER</text>',
        '<line x1="20" y1="205" x2="1480" y2="205" stroke="#dc2626" '
        'stroke-width="2" stroke-dasharray="8 6"/>',
        '<rect x="645" y="185" width="210" height="35" rx="8" fill="#fff"/>'
        '<text x="750" y="208" text-anchor="middle" font-family="Arial,sans-serif" '
        'font-size="15" font-weight="700" fill="#b91c1c">Freeze before transfer</text>',
    ]
    for source, target in edges:
        x1, y1, w1, h1 = boxes[source]
        x2, y2, w2, h2 = boxes[target]
        sx, sy = x1 + w1, y1 + h1 / 2
        ex, ey = x2, y2 + h2 / 2
        if ex < sx:
            sx, ex = x1 + w1 / 2, x2 + w2 / 2
            sy, ey = y1 + h1, y2
        out.append(
            f'<path d="M{sx:g},{sy:g} L{ex:g},{ey:g}" fill="none" '
            'stroke="#475569" stroke-width="1.7" marker-end="url(#arrow)"/>'
        )
    for key, x, y, width, label in nodes:
        fill = (
            "#e0f2fe" if key.startswith("a") else ("#dcfce7" if key in {"t5", "t6"} else "#f1f5f9")
        )
        out.append(
            f'<rect id="{key}" x="{x}" y="{y}" width="{width}" height="52" '
            f'rx="9" fill="{fill}" stroke="#64748b"/>'
        )
        words = label.split(" ")
        lines = [" ".join(words)]
        if len(label) > 24:
            midpoint = len(words) // 2
            lines = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
        start = y + (22 if len(lines) == 1 else 19)
        for idx, line in enumerate(lines):
            out.append(
                f'<text x="{x + width / 2:g}" y="{start + idx * 16:g}" '
                'text-anchor="middle" font-family="Arial,sans-serif" font-size="12" '
                f'fill="#0f172a">{escape(line)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/figures/figure1_recovergraph.svg"
    )
    output = parser.parse_args().output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_svg(), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
