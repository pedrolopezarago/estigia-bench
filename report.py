"""Report for an estigia-bench run.

Reads results.jsonl + samples.csv + meta/card/extras and produces:
  summary.json         every aggregated figure (median, quartiles, n)
  baseline_tables.md   tables ready to paste into the baseline document
  bundle.json          single file loaded by the dashboard (dashboard.html)

Protocol rules applied here:
  * warmup requests and failed requests are discarded;
  * median + IQR is reported, never a single value;
  * energy is integrated from the power samples (trapezoids with linear
    interpolation at the edges of every phase).
"""
from __future__ import annotations

import bisect
import csv
import json
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _num(v):
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def read_samples(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("t_mono") == "t_mono":  # header repeated when appending
                continue
            d = {k: (_num(v) if k not in ("phase", "req_id", "throttled") else v)
                 for k, v in r.items()}
            if d.get("t_mono") is not None:
                rows.append(d)
    rows.sort(key=lambda r: r["t_mono"])
    return rows


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def stats(vals) -> dict | None:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return {"med": v[0], "q1": v[0], "q3": v[0], "n": 1}
    q = st.quantiles(v, n=4, method="inclusive")
    return {"med": st.median(v), "q1": q[0], "q3": q[2], "n": len(v)}


class Series:
    """Time series with linear interpolation (used to integrate power)."""

    def __init__(self, ts, vs):
        pts = [(t, v) for t, v in zip(ts, vs) if t is not None and v is not None]
        self.t = [p[0] for p in pts]
        self.v = [p[1] for p in pts]

    def at(self, x):
        if not self.t or x < self.t[0] or x > self.t[-1]:
            return None
        i = bisect.bisect_left(self.t, x)
        if i < len(self.t) and self.t[i] == x:
            return self.v[i]
        t0, t1, v0, v1 = self.t[i - 1], self.t[i], self.v[i - 1], self.v[i]
        return v0 + (v1 - v0) * (x - t0) / (t1 - t0)

    def integral(self, a, b):
        if b is None or a is None or b <= a:
            return None
        va, vb = self.at(a), self.at(b)
        if va is None or vb is None:
            return None
        i, j = bisect.bisect_right(self.t, a), bisect.bisect_left(self.t, b)
        pts = [(a, va)] + list(zip(self.t[i:j], self.v[i:j])) + [(b, vb)]
        return sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(pts, pts[1:]))

    def mean(self, a, b):
        e = self.integral(a, b)
        return None if e is None else e / (b - a)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
METRICS = ["p_actual", "n_actual", "ttft_s", "e2e_s", "pp_tps_backend", "tg_tps_backend",
           "tg_tps_client", "itl_p50_ms", "itl_p95_ms", "prompt_eval_s", "eval_s",
           "energy_j", "energy_prefill_j", "energy_decode_j", "j_per_token", "power_prefill_w",
           "power_decode_w", "cpu_prefill_pct", "cpu_decode_pct", "rss_peak_mb",
           "words", "words_per_s", "tokens_per_word",
           "ram_peak_mb", "prefill_share", "temp_end_c", "freq_min_mhz"]


def external_power(run_dir: Path, samples: list[dict]) -> dict | None:
    """Replace power with the readings of an external meter (power_external.csv).

    Accepted columns: t_wall or timestamp (epoch s or ISO 8601) and power_w, or
    else voltage_v + current_a. Aligned by wall clock: keep both clocks on NTP.
    """
    p = run_dir / "power_external.csv"
    if not p.exists() or not samples:
        return None
    offs = [s["t_wall"] - s["t_mono"] for s in samples if s.get("t_wall") is not None]
    if not offs:
        return None
    off = st.median(offs)
    pts = []
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            t = row.get("t_wall") or row.get("timestamp") or row.get("time")
            w = row.get("power_w") or row.get("power")
            try:
                tt = float(t)
            except (TypeError, ValueError):
                try:
                    tt = datetime.fromisoformat(t).timestamp()
                except Exception:  # noqa: BLE001
                    continue
            try:
                ww = float(w) if w else float(row["voltage_v"]) * float(row["current_a"])
            except Exception:  # noqa: BLE001
                continue
            pts.append((tt - off, ww))
    if len(pts) < 2:
        return None
    pts.sort()
    ser = Series([p[0] for p in pts], [p[1] for p in pts])
    covered = 0
    for s in samples:
        v = ser.at(s["t_mono"])
        s["power_w"] = v
        covered += v is not None
    return {"source": "external meter (power_external.csv)", "rows": len(pts),
            "samples_covered_pct": round(100 * covered / len(samples), 1)}


def enrich(reqs: list[dict], samples: list[dict]):
    pw = Series([s["t_mono"] for s in samples], [s.get("power_w") for s in samples])
    by_req = defaultdict(list)
    for s in samples:
        if s.get("req_id"):
            by_req[s["req_id"]].append(s)
    for r in reqs:
        ts, tf, te = r.get("t_send"), r.get("t_first"), r.get("t_end")
        if ts is None or te is None:
            continue
        r["energy_j"] = pw.integral(ts, te)
        r["energy_prefill_j"] = pw.integral(ts, tf) if tf else None
        r["energy_decode_j"] = pw.integral(tf, te) if tf else None
        r["power_prefill_w"] = pw.mean(ts, tf) if tf else None
        r["power_decode_w"] = pw.mean(tf, te) if tf else None
        n = r.get("n_actual")
        r["j_per_token"] = (r["energy_decode_j"] / (n - 1)) if r.get("energy_decode_j") and n and n > 1 else None
        if r.get("ttft_s") and r.get("e2e_s"):
            r["prefill_share"] = r["ttft_s"] / r["e2e_s"]
        if r["suite"] in ("S6_kv_cache", "S7_scenario") and r.get("text") and tf:
            w = len(r["text"].split())
            r["words"] = w
            r["words_per_s"] = w / (te - tf) if te > tf and w else None
            r["tokens_per_word"] = (n / w) if n and w else None
        ss = by_req.get(r["id"], [])
        pre = [s["cpu_total"] for s in ss if s["phase"] == "prefill" and s.get("cpu_total") is not None]
        dec = [s["cpu_total"] for s in ss if s["phase"] == "decode" and s.get("cpu_total") is not None]
        r["cpu_prefill_pct"] = st.mean(pre) if pre else None
        r["cpu_decode_pct"] = st.mean(dec) if dec else None
        rss = [s["rss_server_mb"] for s in ss if s.get("rss_server_mb") is not None]
        ram = [s["ram_used_mb"] for s in ss if s.get("ram_used_mb") is not None]
        fr = [s["freq_mhz"] for s in ss if s.get("freq_mhz") is not None]
        r["rss_peak_mb"] = max(rss) if rss else None
        r["ram_peak_mb"] = max(ram) if ram else None
        r["freq_min_mhz"] = min(fr) if fr else None


def group(reqs: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for r in reqs:
        if r.get("warmup") or r.get("error") or r["suite"] == "S2_load":
            continue
        g[(r["suite"], r["variant"], r.get("kv"), r.get("p_target"), r.get("n_target"))].append(r)
    out = []
    for (suite, variant, kv, pt, nt), rs in sorted(g.items(), key=lambda kv_: (
            kv_[0][0], kv_[0][3] or 0, kv_[0][4] or 0, str(kv_[0][1]), str(kv_[0][2]))):
        d = {"suite": suite, "variant": variant, "kv": kv, "p_target": pt, "n_target": nt,
             "n": len(rs), "flags": sorted({f for r in rs for f in r.get("flags", [])})}
        for m in METRICS:
            d[m] = stats(r.get(m) for r in rs)
        out.append(d)
    return out


def idle_summary(samples: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for s in samples:
        if s["phase"].startswith("idle:"):
            g[s["phase"][5:]].append(s)
    out = []
    for label, ss in g.items():
        def med(k):
            v = [s[k] for s in ss if s.get(k) is not None]
            return st.median(v) if v else None
        out.append({"label": label, "seconds": round(ss[-1]["t_mono"] - ss[0]["t_mono"], 1),
                    "power_w": med("power_w"), "cpu_pct": med("cpu_total"),
                    "ram_used_mb": med("ram_used_mb"), "rss_server_mb": med("rss_server_mb"),
                    "temp_c": med("temp_c")})
    order = {"no_server": 0, "server_no_model": 1, "model_loaded": 2}
    return sorted(out, key=lambda x: order.get(x["label"], 9))


def load_summary(reqs):
    out = {}
    for mode in ("cold", "warm"):
        rs = [r for r in reqs if r["suite"] == "S2_load" and r["variant"] == mode]
        if rs:
            out[mode] = {"load_client_s": stats(r.get("load_client_s") for r in rs),
                         "load_backend_s": stats(r.get("load_backend_s") for r in rs),
                         "ram_delta_mb": stats(r.get("ram_delta_mb") for r in rs)}
    return out


def _find(groups, suite, **kw):
    for g in groups:
        if g["suite"] == suite and all(g.get(k) == v for k, v in kw.items()):
            return g
    return None


def _med(g, m):
    return g[m]["med"] if g and g.get(m) else None


def model_bytes(card: dict) -> int | None:
    g = card.get("gguf") or {}
    if g.get("tensor_bytes"):
        return g["tensor_bytes"]
    try:
        name = card["backend"]["model"]
        for m in card["backend"]["tags"]["models"]:
            if m.get("name") == name or m.get("model") == name:
                return m.get("size")
    except Exception:  # noqa: BLE001
        pass
    return None


def roofline(groups, card, extras):
    bw = (extras.get("membw") or {}).get("gbps_best")
    mb = model_bytes(card)
    s4 = [g for g in groups if g["suite"] == "S4_generation" and g.get("tg_tps_client")]
    ref = min(s4, key=lambda g: abs((g["n_target"] or 0) - 128)) if s4 else None
    tg = _med(ref, "tg_tps_client")
    ceil = (bw * 1e9 / mb) if bw and mb else None
    return {"membw_gbps": bw, "model_bytes": mb, "ceiling_tps": ceil, "tg_tps": tg,
            "tg_ref_n": ref["n_target"] if ref else None,
            "pct": (100 * tg / ceil) if tg and ceil else None,
            "effective_gbps": (tg * mb / 1e9) if tg and mb else None,
            "sanity_ok": (tg * mb / 1e9 <= bw * 1.05) if tg and mb and bw else None}


def soak_summary(reqs):
    rs = sorted([r for r in reqs if r["suite"] == "S8_soak" and not r.get("error")
                 and r.get("t_send")], key=lambda r: r["t_send"])
    if len(rs) < 4:
        return None
    t0 = rs[0]["t_send"]
    first = [r["tg_tps_client"] for r in rs if r["t_send"] - t0 <= 60 and r.get("tg_tps_client")]
    tend = rs[-1]["t_send"]
    last = [r["tg_tps_client"] for r in rs if tend - r["t_send"] <= 60 and r.get("tg_tps_client")]
    a, b = (st.median(first) if first else None), (st.median(last) if last else None)
    return {"duration_s": round(rs[-1]["t_end"] - t0, 1), "requests": len(rs),
            "tg_first_min": a, "tg_last_min": b,
            "drop_pct": (100 * (a - b) / a) if a and b else None,
            "temp_max_c": max((r["temp_end_c"] for r in rs if r.get("temp_end_c") is not None),
                              default=None),
            "freq_min_mhz": min((r.get("freq_min_mhz") or 1e9) for r in rs) if any(
                r.get("freq_min_mhz") for r in rs) else None}


def diagnosis(groups, roof, soak, meta, reqs):
    cold = _find(groups, "S6_kv_cache", variant="cold")
    warm = _find(groups, "S6_kv_cache", variant="warm")
    prod = _find(groups, "S7_scenario", variant="production")
    share = _med(cold, "prefill_share") or _med(prod, "prefill_share")
    ttft_c, ttft_w = _med(cold, "ttft_s"), _med(warm, "ttft_s")
    gain = (100 * (ttft_c - ttft_w) / ttft_c) if ttft_c and ttft_w else None
    throttled = sorted({f for r in reqs for f in r.get("flags", []) if f.startswith("throttling")})
    v = []
    if share is not None:
        v.append({"signal": "% of the time in prefill", "value": 100 * share,
                  "rule": "> 50 %", "hit": share > 0.5,
                  "then": "Priority: trim the system prompt and reuse the KV cache" if share > 0.5
                  else "Priority: cap the output and use ARM-optimized quantization"})
    if gain is not None:
        v.append({"signal": "TTFT improvement with a warm KV cache", "value": gain, "rule": "> 40 %",
                  "hit": gain > 40, "then": "Cache reuse comes first" if gain > 40
                  else "The KV cache contributes little on its own"})
    if roof.get("pct") is not None:
        v.append({"signal": "% of the bandwidth ceiling", "value": roof["pct"], "rule": "> 70 %",
                  "hit": roof["pct"] > 70,
                  "then": "Objective 1 expectation: a factor of 2-3 at most" if roof["pct"] > 70
                  else "There is software headroom (llama.cpp, threads, ARM repacking)"})
    if soak and soak.get("drop_pct") is not None:
        v.append({"signal": "Generation drop under continuous load", "value": soak["drop_pct"],
                  "rule": "> 10 %", "hit": soak["drop_pct"] > 10,
                  "then": "Fix cooling before optimizing" if soak["drop_pct"] > 10
                  else "Thermally stable"})
    v.append({"signal": "Throttling during the run", "value": len(throttled), "rule": "any",
              "hit": bool(throttled), "then": ("Affected measurements: " + ", ".join(throttled))
              if throttled else "No throttling"})
    sw = meta.get("swap_max_mb") or 0
    v.append({"signal": "Peak swap (MB)", "value": sw, "rule": "> 0", "hit": sw > 1,
              "then": "The figures are void: reduce num_ctx or the model" if sw > 1 else "No swap"})
    return {"prefill_share": share, "ttft_cold_s": ttft_c, "ttft_warm_s": ttft_w,
            "ttft_gain_warm_pct": gain, "throttling": throttled, "verdicts": v}


# --------------------------------------------------------------------------- #
# Markdown for the baseline document
# --------------------------------------------------------------------------- #
def _fmt(v, nd=2, unit=""):
    if v is None:
        return "—"
    if isinstance(v, dict):
        v = v.get("med")
        if v is None:
            return "—"
    return f"{v:,.{nd}f}{unit}"


def _iqr(s, nd=2):
    return "—" if not s else f"{s['q1']:.{nd}f}–{s['q3']:.{nd}f}"


def markdown(summary) -> str:
    m, card, groups = summary["meta"], summary["card"], summary["groups"]
    show = (card.get("backend") or {}).get("show") or {}
    det, mi = show.get("details", {}), show.get("model_info", {}) or {}
    sysc = card.get("system", {})
    L = [f"# Baseline tables · {m['run_id']}", "",
         f"Board {m.get('board', '—')} · engine {m['backend']} · model `{m['model']}` · "
         f"method_hash `{m['method_hash']}` · "
         f"{m.get('started')}", ""]
    L += ["## Card (auto)", "", "| Field | Value |", "| --- | --- |"]
    gg = card.get("gguf") or {}
    for k, v in [("Board", sysc.get("board")), ("Kernel", sysc.get("kernel")),
                 ("Governor", sysc.get("governor")), ("Total RAM (MB)", sysc.get("ram_total_mb")),
                 ("Storage", (sysc.get("storage") or {}).get("mount")),
                 ("PMIC input voltage (V)", sysc.get("pmic_ext5v_v")),
                 ("Engine version", ((card.get("backend") or {}).get("version") or {}).get("version")
                  if isinstance((card.get("backend") or {}).get("version"), dict)
                  else (card.get("backend") or {}).get("version")),
                 ("Architecture", mi.get("general.architecture") or det.get("family")),
                 ("Parameters", mi.get("general.parameter_count") or det.get("parameter_size")),
                 ("Format", det.get("format") or "gguf"),
                 ("Declared quantization", det.get("quantization_level")),
                 ("Dominant quantization (bytes)", gg.get("dominant")),
                 ("Effective bits per weight", gg.get("bits_per_weight")),
                 ("File size (GB)", round(gg["file_bytes"] / 1e9, 3) if gg.get("file_bytes") else None),
                 ("GGUF path", card.get("model_path"))]:
        L.append(f"| {k} | {v if v not in (None, '') else '—'} |")
    if gg.get("types"):
        L += ["", "## Effective quantization", "",
              "| Tensor type | Tensors | Bytes | % of bytes |",
              "| --- | --- | --- | --- |"]
        for t in gg["types"]:
            L.append(f"| {t['type']} | {t['tensors']} | {t['bytes']:,} | {t['pct_bytes']:.1f} % |")

    L += ["", "## Idle", "", "| State | Power (W) | CPU (%) | RAM used (MB) | Temperature (°C) |",
          "| --- | --- | --- | --- | --- |"]
    for i in summary["idle"]:
        L.append(f"| {i['label']} | {_fmt(i['power_w'])} | {_fmt(i['cpu_pct'], 1)} | "
                 f"{_fmt(i['ram_used_mb'], 0)} | {_fmt(i['temp_c'], 1)} |")
    ld, ex = summary["load"], summary["extras"]
    L += ["", "## Model load", "", "| Load | Value |", "| --- | --- |",
          f"| Cold (s) | {_fmt((ld.get('cold') or {}).get('load_client_s'))} |",
          f"| Warm (s) | {_fmt((ld.get('warm') or {}).get('load_client_s'))} |",
          f"| RAM added (MB) | {_fmt((ld.get('warm') or {}).get('ram_delta_mb'), 0)} |",
          f"| Storage read (MB/s) | {_fmt((ex.get('storage_read') or {}).get('mbps'), 1)} |"]

    L += ["", "## Prompt processing (S3)", "",
          "| Target prompt | Actual prompt | Processing (tok/s) | TTFT (s) | TTFT IQR (s) |",
          "| --- | --- | --- | --- | --- |"]
    for g in [g for g in groups if g["suite"] == "S3_prefill"]:
        L.append(f"| {g['p_target']} | {_fmt(g['p_actual'], 0)} | {_fmt(g['pp_tps_backend'], 1)} | "
                 f"{_fmt(g['ttft_s'], 3)} | {_iqr(g['ttft_s'], 3)} |")
    L += ["", "## Generation (S4)", "",
          "| Target output | Actual output | Backend (tok/s) | Client (tok/s) | Inter-token p50 / p95 (ms) |",
          "| --- | --- | --- | --- | --- |"]
    for g in [g for g in groups if g["suite"] == "S4_generation"]:
        L.append(f"| {g['n_target']} | {_fmt(g['n_actual'], 0)} | {_fmt(g['tg_tps_backend'])} | "
                 f"{_fmt(g['tg_tps_client'])} | {_fmt(g['itl_p50_ms'], 1)} / {_fmt(g['itl_p95_ms'], 1)} |")

    L += ["", "## Reference scenario (S6, S7)", "",
          "| Variant | Prompt tokens | Output tokens | TTFT (s) | Backend prefill (s) | Backend generation (s) | End to end (s) | % in prefill | Words/s |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for suite, var, label in (("S6_kv_cache", "cold", "Cold KV cache"),
                              ("S6_kv_cache", "warm", "Warm KV cache"),
                              ("S7_scenario", "production", "Production as is")):
        g = _find(groups, suite, variant=var)
        if g:
            ps = g["prefill_share"]["med"] * 100 if g.get("prefill_share") else None
            L.append(f"| {label} | {_fmt(g['p_actual'], 0)} | {_fmt(g['n_actual'], 0)} | "
                     f"{_fmt(g['ttft_s'])} | {_fmt(g['prompt_eval_s'])} | {_fmt(g['eval_s'])} | "
                     f"{_fmt(g['e2e_s'])} | {_fmt(ps, 0, ' %')} | {_fmt(g.get('words_per_s'), 2)} |")

    ref = _find(groups, "S6_kv_cache", variant="cold") or _find(groups, "S7_scenario", variant="production")
    if ref:
        L += ["", "## Resources and energy (scenario, cold KV)", "",
              "| Metric | Prefill | Generation |", "| --- | --- | --- |",
              f"| Mean CPU (%) | {_fmt(ref['cpu_prefill_pct'], 0)} | {_fmt(ref['cpu_decode_pct'], 0)} |",
              f"| Mean power (W) | {_fmt(ref['power_prefill_w'])} | {_fmt(ref['power_decode_w'])} |",
              "", "| Metric | Value |", "| --- | --- |",
              f"| Peak RAM used (MB) | {_fmt(ref['ram_peak_mb'], 0)} |",
              f"| Peak engine RSS (MB) | {_fmt(ref['rss_peak_mb'], 0)} |",
              f"| Peak swap of the run (MB) | {m.get('swap_max_mb', '—')} |",
              f"| Energy per response (J) | {_fmt(ref['energy_j'], 1)} |",
              f"| Energy per generated token (J/token) | {_fmt(ref['j_per_token'], 3)} |",
              f"| Temperature at the end (°C) | {_fmt(ref['temp_end_c'], 1)} |"]
    sk = summary.get("soak")
    if sk:
        L += [f"| Generation drop under continuous load (%) | {_fmt(sk['drop_pct'], 1)} |",
              f"| Peak temperature under continuous load (°C) | {_fmt(sk['temp_max_c'], 1)} |"]
    rf = summary["roofline"]
    L += ["", "## Physical ceiling", "", "| Item | Value |", "| --- | --- |",
          f"| Measured bandwidth (GB/s) | {_fmt(rf['membw_gbps'])} |",
          f"| Weight bytes per token (GB) | {_fmt(rf['model_bytes'] / 1e9 if rf['model_bytes'] else None, 3)} |",
          f"| Theoretical ceiling (tok/s) | {_fmt(rf['ceiling_tps'], 1)} |",
          f"| Measured generation (tok/s, N={rf['tg_ref_n']}) | {_fmt(rf['tg_tps'], 2)} |",
          f"| % of the ceiling | {_fmt(rf['pct'], 0, ' %')} |",
          f"| Sanity check | {'—' if rf['sanity_ok'] is None else ('OK' if rf['sanity_ok'] else 'FAILS: check the model or the measurement')} |"]
    L += ["", "## Diagnosis", "", "| Signal | Value | Threshold | Triggered? | Consequence |",
          "| --- | --- | --- | --- | --- |"]
    for v in summary["diagnosis"]["verdicts"]:
        L.append(f"| {v['signal']} | {_fmt(v['value'], 1)} | {v['rule']} | {'yes' if v['hit'] else 'no'} | {v['then']} |")
    if summary["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in summary["warnings"]]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def build(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = _json(run_dir / "meta.json", {})
    card = _json(run_dir / "card.json", {})
    extras = _json(run_dir / "extras.json", {}) or {}
    reqs = []
    rp = run_dir / "results.jsonl"
    if rp.exists():
        reqs = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
    samples = read_samples(run_dir / "samples.csv")
    ext = external_power(run_dir, samples)
    if ext:
        extras["power_source"] = ext
    elif not (meta.get("plan") or {}).get("power", {}).get("source") in (None, "none"):
        extras.setdefault("power_source", {"source": (meta.get("plan") or {}).get("power", {}).get("source")})
    enrich(reqs, samples)
    groups = group(reqs)
    roof = roofline(groups, card, extras)
    soak = soak_summary(reqs)
    diag = diagnosis(groups, roof, soak, meta, reqs)

    warnings = []
    if meta.get("is_demo"):
        warnings.append("DEMO DATA: not measured on real hardware.")
    if meta.get("allow_placeholder"):
        warnings.append("S6/S7 measured with the placeholder reference prompt: not valid for the baseline.")
    if not any(s.get("power_w") is not None for s in samples):
        warnings.append("No power measurements (the Pi 4 has no PMIC): leave the meter log as "
                        "power_external.csv in the run folder and run report again.")
    elif ext and ext["samples_covered_pct"] < 90:
        warnings.append(f"The external meter only covers {ext['samples_covered_pct']} % of the run: "
                        "check the clock synchronization.")
    flagged = defaultdict(int)
    for r in reqs:
        if not r.get("warmup"):
            for f in r.get("flags", []):
                flagged[f] += 1
    for f, n in sorted(flagged.items()):
        warnings.append(f"{n} requests flagged «{f}».")
    errors = [r for r in reqs if r.get("error")]
    if errors:
        warnings.append(f"{len(errors)} requests failed (excluded).")

    summary = {"meta": meta, "card": card, "extras": extras, "groups": groups,
               "idle": idle_summary(samples), "load": load_summary(reqs), "roofline": roof,
               "soak": soak, "diagnosis": diag, "warnings": warnings}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    (run_dir / "baseline_tables.md").write_text(markdown(summary), encoding="utf-8")

    # bundle for the dashboard: subsampled samples and lightweight requests
    t0 = samples[0]["t_mono"] if samples else (reqs[0].get("t_send", 0) if reqs else 0)
    step = max(1, len(samples) // 6000)
    cols = ["t", "phase", "req_id", "cpu_total", "ram_used_mb", "rss_server_mb", "temp_c",
            "freq_mhz", "power_w"]
    srows = [[round(s["t_mono"] - t0, 2), s["phase"], s.get("req_id") or "",
              *[s.get(c) for c in cols[3:]]] for s in samples[::step]]
    lite = []
    for r in reqs:
        if r.get("error") or r.get("t_send") is None:
            continue
        lite.append({k: r.get(k) for k in ("id", "suite", "variant", "kv", "warmup", "p_actual",
                                           "n_actual", "ttft_s", "tg_tps_client", "e2e_s",
                                           "energy_j", "temp_end_c", "flags")} |
                    {"t0": round(r["t_send"] - t0, 3),
                     "t1": round((r.get("t_first") or r["t_send"]) - t0, 3),
                     "t2": round(r["t_end"] - t0, 3)})
    slim_card = {"system": card.get("system"), "gguf": card.get("gguf"),
                 "model_path": card.get("model_path"),
                 "backend": {k: (card.get("backend") or {}).get(k) for k in ("backend", "model", "version", "url")},
                 "details": ((card.get("backend") or {}).get("show") or {}).get("details"),
                 "model_info": {k: v for k, v in ((((card.get("backend") or {}).get("show") or {})
                                                   .get("model_info")) or {}).items()
                                if isinstance(v, (int, float, str)) and len(str(v)) < 80}}
    bundle = {"format": "estigia-bench/bundle@1",
              "meta": {k: meta.get(k) for k in ("run_id", "stage", "board", "note", "backend", "model",
                                                 "method_version", "method_hash", "started",
                                                 "finished", "status", "bench_version",
                                                 "bench_commit", "payload_commit", "is_demo",
                                                 "swap_max_mb", "idle_temp_c")},
              "inference": (meta.get("plan") or {}).get("inference"),
              "card": slim_card, "extras": extras, "groups": groups, "idle": summary["idle"],
              "load": summary["load"], "roofline": roof, "soak": soak, "diagnosis": diag,
              "warnings": warnings, "requests": lite,
              "samples": {"cols": cols, "rows": srows}}
    (run_dir / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, default=str))
    print(f"report: {run_dir / 'baseline_tables.md'}  ·  dashboard: {run_dir / 'bundle.json'}")
    return summary
