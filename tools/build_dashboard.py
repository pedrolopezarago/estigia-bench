#!/usr/bin/env python3
"""Embeds demo bundles into dashboard.html (so it works without loading anything).

    python tools/build_dashboard.py runs_demo/*/bundle.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SAMPLES = 1800


def r(v, nd=2):
    return round(v, nd) if isinstance(v, float) else v


def slim(b: dict) -> dict:
    b = dict(b)
    rows = b["samples"]["rows"]
    step = max(1, len(rows) // MAX_SAMPLES)
    b["samples"] = {"cols": b["samples"]["cols"], "rows": [[r(x) for x in row] for row in rows[::step]]}
    b["requests"] = [{"id": q["id"], "suite": q["suite"], "warmup": q["warmup"],
                      "tg_tps_client": r(q.get("tg_tps_client")), "t0": r(q["t0"]), "t2": r(q["t2"])}
                     for q in b["requests"]]
    return b


def main(paths):
    demo = [slim(json.loads(Path(p).read_text(encoding="utf-8"))) for p in paths]
    html = (ROOT / "dashboard.html").read_text(encoding="utf-8")
    blob = json.dumps(demo, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = re.sub(r"/\*__DEMO__\*/.*?/\*__END__\*/", lambda m: f"/*__DEMO__*/{blob}/*__END__*/",
                  html, count=1, flags=re.S)
    (ROOT / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"dashboard.html with {len(demo)} demo runs ({len(blob) / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
