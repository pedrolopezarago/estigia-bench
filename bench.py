#!/usr/bin/env python3
"""estigia-bench: performance test bench for the AI payload.

Pluton UPV · Estigia mission · AI department · academic year 2026-27.

Quick usage (on the CM5):
    python bench.py card --model <tag>
    python bench.py run  --plan config.yaml --stage baseline --model <tag>
    python bench.py idle --out runs/<run_id> --label no_server --seconds 120
    python bench.py report runs/<run_id>

Every run leaves in runs/<run_id>/:
    meta.json      metadata (versions, model, parameters, method_hash)
    card.json      system and model card
    results.jsonl  one line per measured request
    samples.csv    system samples (CPU, RAM, temperature, power...) with phase
    extras.json    bandwidth, storage read, calibration
    summary.json, baseline_tables.md, bundle.json  (produced by `report`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml

import modelcard
import report
from backends import make_backend
from sysmon import SysMonitor, decode_throttled, read_temp, read_throttled

BENCH_VERSION = "0.2.0"
SYNTH_VERSION = "counting-v1"   # synthetic prompt generator (numeric sequence)
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def resolve(path: str, plan_dir: Path) -> Path:
    """Path relative to the plan; if it does not exist, relative to estigia-bench."""
    p = Path(path)
    if p.is_absolute() or (plan_dir / p).exists():
        return p if p.is_absolute() else plan_dir / p
    return HERE / p


def method_hash(plan: dict, plan_dir: Path) -> str:
    """Fingerprint of the METHOD (not of the model or the engine). If it changes, do not compare."""
    keys = ["method_version", "inference", "sampling", "power", "protocol", "suites",
            "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    m = {k: plan.get(k) for k in keys}
    pf = (plan.get("s7") or {}).get("prompt_file")
    m["prompt_file_sha"] = file_sha(resolve(pf, plan_dir)) if pf else None
    m["synth"] = SYNTH_VERSION
    blob = json.dumps(m, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit(path: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def counting_text(k: int) -> str:
    return ", ".join(str(i) for i in range(1, k + 1)) + ","


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, plan, backend, mon: SysMonitor, out: Path, plan_dir: Path):
        self.plan = plan
        self.backend = backend
        self.mon = mon
        self.out = out
        self.plan_dir = plan_dir
        self.proto = plan.get("protocol", {})
        self.results = (out / "results.jsonl").open("a", encoding="utf-8")
        self.n = 0
        self.idle_temp = None
        self.calib = None          # (a, b): tokens ≈ a + b·k
        self.extras: dict = {}
        self.loaded = False

    # -- protocol -------------------------------------------------------- #
    def cooldown(self):
        self.mon.set_phase("cooldown", "")
        time.sleep(float(self.proto.get("min_gap_s", 1.0)))
        ref = self.idle_temp
        t = read_temp()
        if ref is None or t is None:
            return
        target = ref + float(self.proto.get("cooldown_delta_c", 2.0))
        t0 = time.perf_counter()
        while t is not None and t > target and time.perf_counter() - t0 < float(
                self.proto.get("cooldown_max_s", 60)):
            time.sleep(1.0)
            t = read_temp()

    def ensure_loaded(self):
        if not self.loaded:
            log("loading model into memory...")
            self.mon.set_phase("load", "")
            self.backend.load()
            self.loaded = True
            self.mon.set_phase("gap", "")

    @staticmethod
    def nonce() -> str:
        return f"[{secrets.token_hex(4)}] "

    # -- synthetic prompt generator -------------------------------------- #
    def calibrate(self):
        if self.calib:
            return
        k1, k2 = 60, 360
        exact = self.backend.count_tokens(self.nonce() + counting_text(k1))
        if exact is not None:
            p1 = exact
            p2 = self.backend.count_tokens(self.nonce() + counting_text(k2))
        else:
            log("calibrating prompt length (2 requests)...")
            p1 = self.backend.generate(self.nonce() + counting_text(k1), 1, raw=True)["p_actual"]
            p2 = self.backend.generate(self.nonce() + counting_text(k2), 1, raw=True)["p_actual"]
        b = (p2 - p1) / (k2 - k1)
        a = p1 - b * k1
        self.calib = (a, b)
        self.extras["prompt_calibration"] = {"a_tokens": round(a, 2),
                                             "b_tokens_per_number": round(b, 4),
                                             "exact": exact is not None}
        log(f"calibration: tokens ≈ {a:.1f} + {b:.3f}·k")

    def synth_prompt(self, p_target: int) -> str:
        self.calibrate()
        a, b = self.calib
        k = max(1, round((p_target - a) / b))
        txt = self.nonce() + counting_text(k)
        # With an exact tokenizer (llama.cpp) it is fine-tuned.
        for _ in range(6):
            n = self.backend.count_tokens(txt)
            if n is None or n == p_target:
                break
            k = max(1, k + round((p_target - n) / b))
            txt = txt.split("] ", 1)[0] + "] " + counting_text(k)
        return txt

    # -- one measurement ------------------------------------------------- #
    def measure(self, suite: str, variant: str, prompt: str, n_predict: int, *, rep: int,
                warmup: bool, p_target=None, n_target=None, kv="cold", system=None,
                chat=False, raw=True, cache=True, ignore_eos=True, options=None,
                keep_text=False, cooldown=True, question=None) -> dict:
        if cooldown:
            self.cooldown()
        self.n += 1
        rid = f"{suite.split('_')[0]}-{self.n:05d}"
        temp0, thr0 = read_temp(), read_throttled()
        try:
            res = self.backend.generate(prompt, n_predict, system=system, chat=chat, raw=raw,
                                        cache=cache, ignore_eos=ignore_eos, options=options,
                                        phase_cb=lambda ph: self.mon.set_phase(ph, rid))
        except Exception as e:  # noqa: BLE001
            self.mon.set_phase("gap", "")
            log(f"  ERROR in {rid}: {e}")
            rec = {"id": rid, "suite": suite, "variant": variant, "rep": rep, "warmup": warmup,
                   "error": str(e)}
            self.results.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.results.flush()
            return rec
        self.mon.set_phase("gap", "")
        temp1, thr1 = read_temp(), read_throttled()

        flags = []
        if p_target and res.get("p_actual") and kv == "cold" and \
                abs(res["p_actual"] - p_target) > max(6, 0.1 * p_target):
            flags.append("prompt_length_out_of_tolerance")
        if n_target and res.get("n_actual") is not None and res["n_actual"] < 0.8 * n_target:
            flags.append("short_output")
        if not warmup and (res.get("load_backend_s") or 0) > 0.5:
            flags.append("model_reload")
        new_bits = set(decode_throttled(thr1)) - set(decode_throttled(thr0))
        if new_bits:
            flags.append("throttling:" + "|".join(sorted(new_bits)))
        if res.get("tg_tps_backend") and res.get("tg_tps_client"):
            if abs(res["tg_tps_backend"] - res["tg_tps_client"]) / res["tg_tps_backend"] > 0.05:
                flags.append("backend_and_client_differ_>5%")

        text = res.pop("text", "")
        rec = {"id": rid, "suite": suite, "variant": variant, "kv": kv, "rep": rep,
               "warmup": warmup, "p_target": p_target, "n_target": n_target,
               "question": question, **res, "temp_start_c": temp0, "temp_end_c": temp1,
               "throttled_start": thr0, "throttled_end": thr1, "flags": flags,
               "text": text if keep_text else text[:120]}
        self.results.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.results.flush()
        tag = "warmup" if warmup else f"rep {rep}"
        log(f"  {rid} {suite} {variant} {tag}: P={res.get('p_actual')} N={res.get('n_actual')} "
            f"TTFT={_f(res.get('ttft_s'))}s gen={_f(res.get('tg_tps_client'))} tok/s "
            f"T={_f(temp1)}°C {' '.join(flags)}")
        return rec

    def reps(self, suite_cfg: dict | None = None):
        sc = suite_cfg or {}
        return int(sc.get("warmup", self.proto.get("warmup", 1))), int(
            sc.get("reps", self.proto.get("reps", 5)))


def _f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #
def s0_system(ctx: Ctx):
    cfg = ctx.plan.get("s0", {})
    if cfg.get("membw_mb", 0):
        ctx.backend.unload()
        ctx.loaded = False
        time.sleep(2)
        ctx.mon.set_phase("membw", "")
        log(f"memory bandwidth ({cfg['membw_mb']} MB, {cfg.get('membw_procs', 4)} processes)...")
        bw = modelcard.membw(int(cfg["membw_mb"]), int(cfg.get("membw_procs", 4)))
        ctx.extras["membw"] = bw
        log(f"  {bw}")
    blob = ctx.extras.get("model_path")
    if cfg.get("storage_read") and blob:
        if modelcard.can_drop_caches():
            ctx.mon.set_phase("storage", "")
            ctx.extras["storage_read"] = modelcard.storage_read(blob)
            log(f"  storage read: {ctx.extras['storage_read']}")
        else:
            ctx.extras["storage_read"] = {"skipped": "run with sudo to measure the cold read"}
    ctx.mon.set_phase("gap", "")


def s1_idle(ctx: Ctx):
    secs = float(ctx.plan.get("s1", {}).get("seconds_per_state", 120))
    ctx.backend.unload()
    ctx.loaded = False
    time.sleep(5)
    log(f"idle: server without model ({secs:.0f} s)")
    ctx.mon.set_phase("idle:server_no_model", "")
    time.sleep(secs)
    ctx.idle_temp = read_temp()
    ctx.ensure_loaded()
    time.sleep(5)
    log(f"idle: model loaded ({secs:.0f} s)")
    ctx.mon.set_phase("idle:model_loaded", "")
    time.sleep(secs)
    ctx.mon.set_phase("gap", "")


def s2_load(ctx: Ctx):
    cfg = ctx.plan.get("s2", {})
    if ctx.backend.name != "ollama":
        log("S2: with llama-server the model is loaded when the server starts; measure it by hand.")
        return
    modes = []
    if cfg.get("cold", True):
        if modelcard.can_drop_caches():
            modes.append("cold")
        else:
            log("S2: cold load skipped (sudo is required to drop the page cache)")
    modes.append("warm")
    for mode in modes:
        for rep in range(int(cfg.get("reps", 3))):
            ctx.backend.unload()
            time.sleep(2)
            if mode == "cold":
                modelcard.drop_caches()
            ram0 = psutil.virtual_memory()
            ctx.n += 1
            rid = f"S2-{ctx.n:05d}"
            ctx.mon.set_phase("load", rid)
            r = ctx.backend.load()
            ctx.mon.set_phase("gap", "")
            time.sleep(1)
            ram1 = psutil.virtual_memory()
            rec = {"id": rid, "suite": "S2_load", "variant": mode, "rep": rep, "warmup": False,
                   **r, "ram_delta_mb": round(((ram1.total - ram1.available) -
                                               (ram0.total - ram0.available)) / 2**20, 1)}
            ctx.results.write(json.dumps(rec) + "\n")
            ctx.results.flush()
            log(f"  {mode} load rep {rep}: {_f(r.get('load_client_s'))} s, "
                f"+{rec['ram_delta_mb']} MB")
    ctx.loaded = True


def _sweep(ctx: Ctx, suite: str, points: list[tuple[int, int]], cfg: dict):
    ctx.ensure_loaded()
    w, reps = ctx.reps(cfg)
    for p, n in points:
        for i in range(w + reps):
            ctx.measure(suite, f"P{p}_N{n}", ctx.synth_prompt(p), n, rep=i - w, warmup=i < w,
                        p_target=p, n_target=n, kv="cold", raw=True, cache=False, ignore_eos=True)


def s3_prefill(ctx: Ctx):
    cfg = ctx.plan.get("s3", {})
    _sweep(ctx, "S3_prefill", [(p, 1) for p in cfg.get("prompt_tokens", [])], cfg)


def s4_generation(ctx: Ctx):
    cfg = ctx.plan.get("s4", {})
    p = int(cfg.get("prompt_tokens", 32))
    _sweep(ctx, "S4_generation", [(p, n) for n in cfg.get("output_tokens", [])], cfg)


def s5_matrix(ctx: Ctx):
    cfg = ctx.plan.get("s5", {})
    pts = [(p, n) for p in cfg.get("prompt_tokens", []) for n in cfg.get("output_tokens", [])]
    _sweep(ctx, "S5_matrix", pts, cfg)


def load_reference(ctx: Ctx, allow_placeholder: bool) -> dict | None:
    pf = ctx.plan.get("s7", {}).get("prompt_file", "prompts/reference_v1.json")
    path = resolve(pf, ctx.plan_dir)
    ref = json.loads(path.read_text(encoding="utf-8"))
    if "<<" in ref.get("system", "") and not allow_placeholder:
        log(f"S6/S7 SKIPPED: {path} still has the placeholder system prompt. "
            "Paste the real production prompt or use --allow-placeholder to try it out.")
        return None
    return ref


def s6_kv_cache(ctx: Ctx, allow_placeholder=False):
    cfg = ctx.plan.get("s6", {})
    ref = load_reference(ctx, allow_placeholder)
    if not ref:
        return
    ctx.ensure_loaded()
    w, reps = ctx.reps(cfg)
    # Same output length as production: that way the % of time in prefill is the real one.
    n = cfg.get("output_tokens", "production")
    if n == "production":
        v0 = (ctx.plan.get("s7", {}).get("variants") or [{"num_predict": 300}])[0]
        n = v0["num_predict"]
    n = int(n)
    qs = ref["questions"][: int(cfg.get("questions", 5))]
    system = ref["system"]
    for qi, q in enumerate(qs):
        for i in range(w + reps):
            # Cold: unique prefix at the start of the system prompt -> nothing reusable.
            ctx.measure("S6_kv_cache", "cold", q, n, rep=i - w, warmup=i < w, kv="cold",
                        system=ctx.nonce() + system, chat=True, cache=False, ignore_eos=False,
                        question=qi, keep_text=True)
        for i in range(w + reps):
            # Warm: an earlier request with the same system prompt leaves the prefix cached.
            other = qs[(qi + 1) % len(qs)]
            ctx.measure("S6_kv_cache", "primer", other, 1, rep=-1, warmup=True, kv="warm",
                        system=system, chat=True, cache=True, ignore_eos=False, question=qi,
                        cooldown=True)
            ctx.measure("S6_kv_cache", "warm", q, n, rep=i - w, warmup=i < w, kv="warm",
                        system=system, chat=True, cache=True, ignore_eos=False, question=qi,
                        cooldown=False, keep_text=True)


def s7_scenario(ctx: Ctx, allow_placeholder=False):
    cfg = ctx.plan.get("s7", {})
    ref = load_reference(ctx, allow_placeholder)
    if not ref:
        return
    ctx.ensure_loaded()
    w, reps = ctx.reps(cfg)
    variants = cfg.get("variants") or [{"name": "production", "num_predict": 300}]
    for v in variants:
        # Warmup: first question, discarded.
        for _ in range(w):
            ctx.measure("S7_scenario", v["name"], ref["questions"][0], int(v["num_predict"]),
                        rep=-1, warmup=True, kv="production", system=ref["system"], chat=True,
                        ignore_eos=False, options=v.get("options"))
        for rep in range(reps):
            for qi, q in enumerate(ref["questions"]):
                ctx.measure("S7_scenario", v["name"], q, int(v["num_predict"]), rep=rep,
                            warmup=False, kv="production", system=ref["system"], chat=True,
                            ignore_eos=False, options=v.get("options"), keep_text=True,
                            question=qi)


def s8_soak(ctx: Ctx):
    cfg = ctx.plan.get("s8", {})
    secs = float(cfg.get("seconds", 600))
    p, n = int(cfg.get("prompt_tokens", 64)), int(cfg.get("output_tokens", 128))
    ctx.ensure_loaded()
    log(f"thermal soak: {secs:.0f} s of continuous generation")
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < secs:
        ctx.measure("S8_soak", f"P{p}_N{n}", ctx.synth_prompt(p), n, rep=i, warmup=False,
                    p_target=p, n_target=n, raw=True, cache=False, ignore_eos=True,
                    cooldown=False)
        i += 1


def s9_threads(ctx: Ctx):
    cfg = ctx.plan.get("s9", {})
    if ctx.backend.name != "ollama":
        log("S9: with llama-server threads are fixed at startup (-t); repeat the run per value.")
        return
    w, reps = ctx.reps(cfg)
    p, n = int(cfg.get("prompt_tokens", 128)), int(cfg.get("output_tokens", 64))
    for t in cfg.get("num_thread", [1, 2, 3, 4]):
        for i in range(w + reps):  # the warmup absorbs the model reload
            ctx.measure("S9_threads", f"threads{t}", ctx.synth_prompt(p), n, rep=i - w,
                        warmup=i < w, p_target=p, n_target=n, raw=True, cache=False,
                        ignore_eos=True, options={"num_thread": int(t)})


SUITES = {"S0_system": s0_system, "S1_idle": s1_idle, "S2_load": s2_load,
          "S3_prefill": s3_prefill, "S4_generation": s4_generation, "S5_matrix": s5_matrix,
          "S6_kv_cache": s6_kv_cache, "S7_scenario": s7_scenario,
          "S8_soak": s8_soak, "S9_threads": s9_threads}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def build_card(plan: dict, backend, model: str, gguf_path: str | None = None) -> dict:
    info = backend.info()
    blob = gguf_path or (modelcard.ollama_blob_path(info.get("show", {}))
                         if backend.name == "ollama" else None)
    card = {"system": modelcard.system_card(blob), "backend": info, "model_path": blob}
    if blob:
        card["gguf"] = modelcard.gguf_histogram(blob)
    if backend.name == "ollama":
        card["loader_log"] = modelcard.loader_log()
    return card


def print_card(card: dict):
    s, b = card["system"], card["backend"]
    show = b.get("show", {}) if isinstance(b.get("show"), dict) else {}
    det = show.get("details", {})
    mi = show.get("model_info", {}) or {}
    print("\n=== System card ===")
    for k in ("board", "board_slug", "kernel", "cpu_model", "cpu_count", "governor", "freq_max_mhz",
              "ram_total_mb", "swap_total_mb", "pmic_ext5v_v", "throttled_at_start", "storage"):
        print(f"  {k:20s} {s.get(k)}")
    print("=== Model ===")
    print(f"  tag                  {b.get('model')}")
    print(f"  engine               {b.get('backend')} {b.get('version', '')}")
    print(f"  family               {det.get('family')}  parameters {mi.get('general.parameter_count', det.get('parameter_size'))}")
    print(f"  format / quant.      {det.get('format')} / {det.get('quantization_level')}")
    g = card.get("gguf") or {}
    if "types" in g:
        print(f"  file                 {g['path']} ({g['file_bytes'] / 1e9:.3f} GB, "
              f"{g['bits_per_weight']} bits/weight)")
        for t in g["types"]:
            print(f"    {t['type']:8s} {t['tensors']:4d} tensors  {t['bytes'] / 2**20:9.1f} MiB  "
                  f"{t['pct_bytes']:5.1f} %")
    elif g:
        print(f"  GGUF: {g.get('error')}")
    for line in card.get("loader_log", [])[-12:]:
        print(f"  log> {line}")


def cmd_card(args):
    plan = load_plan(args.plan)
    model = args.model or plan["backend"].get("model")
    backend = make_backend(plan, model)
    card = build_card(plan, backend, model, args.gguf)
    out = Path(args.out or Path(args.runs) / f"card_{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str))
    print_card(card)
    print(f"\nCard saved to {out / 'card.json'}")


def cmd_run(args):
    plan_path = Path(args.plan).resolve()
    plan = load_plan(str(plan_path))
    if args.backend_url:
        plan["backend"]["url"] = args.backend_url
    if args.backend:
        plan["backend"]["type"] = args.backend
    model = args.model or plan["backend"].get("model") or (Path(args.gguf).name if args.gguf else None)
    if not model:
        sys.exit("Give the model with --model (or --gguf with llama.cpp) or in config.yaml.")
    backend = make_backend(plan, model)
    suites = args.suites.split(",") if args.suites else plan.get("suites", list(SUITES))
    board = args.board or modelcard.detect_board()
    slug = lambda x: re.sub(r"[^A-Za-z0-9.]+", "-", str(x)).strip("-")[:48]
    run_id = f"{datetime.now():%Y%m%d-%H%M}_{board}_{backend.name}_{slug(model)}_{slug(args.stage)}"
    out = Path(args.runs) / run_id
    out.mkdir(parents=True, exist_ok=True)

    log(f"run {run_id} · engine {backend.name} · model {model}")
    card = build_card(plan, backend, model, args.gguf)
    (out / "card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str))
    mh = method_hash(plan, plan_path.parent)
    meta = {
        "run_id": run_id, "stage": args.stage, "board": board, "note": args.note,
        "bench_version": BENCH_VERSION,
        "bench_commit": git_commit(HERE), "payload_commit": args.payload_commit,
        "backend": backend.name, "backend_url": plan["backend"]["url"], "model": model,
        "method_version": plan.get("method_version"), "method_hash": mh, "plan": plan,
        "suites": suites, "started": datetime.now().isoformat(timespec="seconds"),
        "temp_start_c": read_temp(), "allow_placeholder": args.allow_placeholder,
        "is_demo": False,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))

    smp = plan.get("sampling", {})
    mon = SysMonitor(out / "samples.csv", interval=float(smp.get("interval_s", 0.5)),
                     power_cfg=plan.get("power", {}), proc_names=backend.proc_names,
                     throttle_every_s=float(smp.get("throttle_every_s", 2.0)))
    mon.start()
    ctx = Ctx(plan, backend, mon, out, plan_path.parent)
    ctx.extras["model_path"] = card.get("model_path")
    ctx.idle_temp = read_temp()
    status = "complete"
    try:
        for s in suites:
            fn = SUITES.get(s)
            if not fn:
                log(f"unknown suite: {s}")
                continue
            log(f"=== {s} ===")
            ctx.mon.set_phase("gap", "")
            try:
                if s in ("S6_kv_cache", "S7_scenario"):
                    fn(ctx, args.allow_placeholder)
                else:
                    fn(ctx)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001  a broken suite must not kill the run
                status = "with_errors"
                log(f"ERROR in {s}: {e!r}; continuing with the next suite")
                ctx.extras.setdefault("suite_errors", {})[s] = repr(e)
    except KeyboardInterrupt:
        status = "interrupted"
        log("interrupted: saving what has been measured so far")
    finally:
        mon.stop()
        ctx.results.close()
        meta.update({"finished": datetime.now().isoformat(timespec="seconds"), "status": status,
                     "swap_max_mb": round(mon.swap_max_mb, 1), "idle_temp_c": ctx.idle_temp,
                     "throttled_end": read_throttled()})
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        (out / "extras.json").write_text(json.dumps(ctx.extras, indent=2, ensure_ascii=False,
                                                    default=str))
    report.build(out)
    log(f"done: {out}")


def cmd_idle(args):
    out = Path(args.out)
    if not out.exists():
        sys.exit(f"{out} does not exist: point to the folder of an existing run.")
    plan = load_plan(args.plan)
    smp = plan.get("sampling", {})
    mon = SysMonitor(out / "samples.csv", interval=float(smp.get("interval_s", 0.5)),
                     power_cfg=plan.get("power", {}), proc_names=("ollama", "llama-server"))
    mon.set_phase(f"idle:{args.label}", "")
    log(f"idle '{args.label}' for {args.seconds} s -> {out / 'samples.csv'}")
    mon.start()
    try:
        time.sleep(args.seconds)
    finally:
        mon.set_phase("gap", "")
        time.sleep(float(smp.get("interval_s", 0.5)) * 2)
        mon.stop()
    log("done; run `report` again to include it")


def cmd_report(args):
    report.build(Path(args.run))


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", default=str(HERE / "config.yaml"),
                        help="test plan (config.yaml by default)")
    common.add_argument("--runs", default=str(HERE / "runs"), help="results folder")
    ap = argparse.ArgumentParser(description="estigia-bench · AI payload test bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("card", parents=[common], help="system and model card")
    c.add_argument("--model")
    c.add_argument("--gguf", help="path to the GGUF (for llama.cpp, or if not detected)")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_card)

    r = sub.add_parser("run", parents=[common], help="execute the test plan")
    r.add_argument("--model")
    r.add_argument("--stage", default="baseline",
                   help="stage of the course: baseline, post_llamacpp, post_arm_quant, ...")
    r.add_argument("--suites", help="comma-separated list, e.g. S3_prefill,S4_generation")
    r.add_argument("--backend", choices=["ollama", "llamacpp"])
    r.add_argument("--backend-url")
    r.add_argument("--gguf")
    r.add_argument("--payload-commit", help="commit of the payload code being measured")
    r.add_argument("--board", help="pi5, pi4, cm5... (detected by default)")
    r.add_argument("--note", default="")
    r.add_argument("--allow-placeholder", action="store_true",
                   help="allow S6/S7 with the placeholder reference prompt (testing only)")
    r.set_defaults(fn=cmd_run)

    i = sub.add_parser("idle", parents=[common], help="measure an idle state and add it to a run")
    i.add_argument("--out", required=True)
    i.add_argument("--label", required=True)
    i.add_argument("--seconds", type=float, default=120)
    i.set_defaults(fn=cmd_idle)

    p = sub.add_parser("report", parents=[common],
                       help="aggregate a run: tables, summary and dashboard bundle")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.plan = args.plan if Path(args.plan).exists() else str(HERE / args.plan)
    args.fn(args)


if __name__ == "__main__":
    main()
