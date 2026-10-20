#!/usr/bin/env python3
"""Generates DEMO runs with a virtual CM5, to show the dashboard before measuring.

    python tools/demo_run.py --out runs_demo

The figures are MADE UP (plausible for a ~1B model in Q4 on a CM5). Every run
is tagged is_demo = true and the dashboard says so. They go through the same
report.py as real runs, so they also exercise the report.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402
import report  # noqa: E402

GGUF_Q4KM = {"types": [{"type": "Q4_K", "tensors": 97, "bytes": 575_143_936, "params": 1_022_361_600, "pct_bytes": 71.21},
                       {"type": "Q6_K", "tensors": 17, "bytes": 217_419_776, "params": 212_385_792, "pct_bytes": 26.92},
                       {"type": "F32", "tensors": 33, "bytes": 15_130_752, "params": 1_067_008, "pct_bytes": 1.87}],
             "dominant": "Q4_K", "tensor_bytes": 807_694_464, "file_bytes": 807_694_464 + 6_912_000,
             "params": 1_235_814_400, "bits_per_weight": 5.229}
GGUF_Q40 = {"types": [{"type": "Q4_0", "tensors": 113, "bytes": 640_204_800, "params": 1_137_879_040, "pct_bytes": 86.27},
                      {"type": "Q6_K", "tensors": 1, "bytes": 86_528_000, "params": 84_520_960, "pct_bytes": 11.66},
                      {"type": "F32", "tensors": 33, "bytes": 15_355_904, "params": 13_414_400, "pct_bytes": 2.07}],
            "dominant": "Q4_0", "tensor_bytes": 742_088_704, "file_bytes": 742_088_704 + 6_912_000,
            "params": 1_235_814_400, "bits_per_weight": 4.804}

STAGES = {
    "pi5_baseline": dict(board="pi5", stage="baseline", backend="ollama", model="example:1b-q4_K_M",
                         quant="Q4_K_M", gguf=GGUF_Q4KM, version={"version": "0.12.3"}, pp=55.0, c0=0.12,
                         tg=11.6, ctx=0.00012, ovh=0.18, p_pre=6.4, p_dec=5.6, rss=1050,
                         load=(6.8, 1.6), membw=11.8, tamb=30.0, freq=2400,
                         board_name="Raspberry Pi 5 Model B Rev 1.0 (DEMO)", cpu="Cortex-A76",
                         note="Ollama with the production configuration"),
    "pi5_post_llamacpp": dict(board="pi5", stage="post_llamacpp", backend="llamacpp",
                              model="sha256-example-1b-q4_K_M.gguf", quant="Q4_K_M", gguf=GGUF_Q4KM,
                              version="b6650", pp=73.0, c0=0.06, tg=12.1, ctx=0.00011, ovh=0.05,
                              p_pre=6.7, p_dec=5.6, rss=1000, load=None, membw=11.8, tamb=30.0,
                              freq=2400, board_name="Raspberry Pi 5 Model B Rev 1.0 (DEMO)",
                              cpu="Cortex-A76", note="llama.cpp with the same GGUF as Ollama"),
    "pi4_baseline": dict(board="pi4", stage="baseline", backend="ollama", model="example:1b-q4_K_M",
                         quant="Q4_K_M", gguf=GGUF_Q4KM, version={"version": "0.12.3"}, pp=13.5, c0=0.30,
                         tg=4.7, ctx=0.0002, ovh=0.25, p_pre=6.0, p_dec=5.4, rss=1060,
                         load=(9.4, 2.3), membw=4.4, tamb=33.0, freq=1800,
                         board_name="Raspberry Pi 4 Model B Rev 1.5 (DEMO)", cpu="Cortex-A72",
                         note="Ollama with the production configuration"),
}
SYS_TOK = 420
Q_TOK = [14, 18, 15, 19, 16, 13, 16, 12, 17, 13]
N_NAT = [148, 62, 121, 187, 95, 133, 77, 54, 160, 110]


class VirtualCM5:
    def __init__(self, st, rng):
        self.st, self.rng = st, rng
        self.t, self.T, self.Tamb = 1000.0, st["tamb"] + 17.5, st["tamb"]
        self.rows, self.res, self.n = [], [], 0
        self.loaded = False
        self.soak_t = None

    def j(self, s=0.02):
        return self.rng.lognormvariate(0, s)

    def advance(self, dur, phase, rid="", power=3.1, cpu=4.0):
        end = self.t + dur
        while self.t < end:
            dt = min(0.5, end - self.t)
            Tss = self.Tamb + 6.0 * power
            self.T += (Tss - self.T) * dt / 75.0
            if self.soak_t is not None:
                self.Tamb = min(self.st["tamb"] + 11.0, self.Tamb + 0.018 * dt)
            ram = 410 + (980 if self.loaded else 0) + (30 if phase in ("prefill", "decode") else 0)
            rss = (self.st["rss"] if self.loaded else 60) + (25 if phase in ("prefill", "decode") else 0)
            cpus = [max(0, min(100, cpu / 4 * self.j(0.05))) for _ in range(4)]
            freq = self.st["freq"] if self.T < 80 else self.st["freq"] - 200
            self.rows.append([f"{self.t:.4f}", f"{1.79e9 + self.t:.3f}", phase, rid,
                              f"{sum(cpus):.1f}", *[f"{c:.1f}" for c in cpus], f"{ram * self.j(0.003):.1f}",
                              "0.0", f"{rss:.1f}", f"{self.T + self.rng.uniform(-0.3, 0.3):.1f}",
                              f"{freq:.0f}", f"{power * self.j(0.03):.3f}", "0x0"])
            self.t += dt

    def cooldown(self):
        idle_T = self.st["tamb"] + 17.5
        self.advance(1.0, "cooldown", power=3.1)
        waited = 0
        while self.T > idle_T + 2 and waited < 60:
            self.advance(1.0, "cooldown", power=3.1)
            waited += 1

    def request(self, suite, variant, p_eval, n, *, rep, warmup, p_target=None, n_target=None,
                kv="cold", question=None, cooldown=True, p_total=None):
        st = self.st
        if cooldown:
            self.cooldown()
        self.n += 1
        rid = f"{suite.split('_')[0]}-{self.n:05d}"
        tg_eff = st["tg"] / (1 + st["ctx"] * (p_total or p_eval)) * self.j()
        if self.soak_t is not None:
            tg_eff *= 1 - 0.065 * min(1.0, (self.t - self.soak_t) / 600)
        t_pre = (p_eval / st["pp"] + st["c0"] + 2e-7 * p_eval ** 2) * self.j()
        ttft = st["ovh"] + t_pre
        T0 = self.T
        t_send = self.t
        self.advance(ttft, "prefill", rid, power=st["p_pre"], cpu=395)
        t_first = self.t
        dec = (n - 1) / tg_eff
        self.advance(dec, "decode", rid, power=st["p_dec"], cpu=380)
        t_end = self.t
        itl = 1000 / tg_eff
        rec = {"id": rid, "suite": suite, "variant": variant, "kv": kv, "rep": rep, "warmup": warmup,
               "p_target": p_target, "n_target": n_target, "question": question,
               "t_send": t_send, "t_end": t_end, "chunks": n, "t_first": t_first, "ttft_s": ttft,
               "e2e_s": t_end - t_send, "tg_tps_client": tg_eff if n > 1 else None,
               "itl_p50_ms": itl if n > 1 else None, "itl_p95_ms": itl * 1.12 if n > 1 else None,
               "p_actual": p_eval, "n_actual": n, "prompt_eval_s": t_pre, "eval_s": n / tg_eff,
               "load_backend_s": 0.0, "pp_tps_backend": p_eval / t_pre,
               "tg_tps_backend": tg_eff * 0.995, "done_reason": "length",
               "temp_start_c": round(T0, 1), "temp_end_c": round(self.T, 1),
               "throttled_start": "0x0", "throttled_end": "0x0", "flags": [],
               "text": " ".join(["word"] * int(n / 1.55)) if suite in ("S6_kv_cache", "S7_scenario") else ""}
        self.res.append(rec)
        self.advance(0.3, "gap", power=3.2)


def simulate(stage: str, out: Path, seed: int):
    st = STAGES[stage]
    rng = random.Random(seed)
    plan = bench.load_plan(str(ROOT / "config.yaml"))
    v = VirtualCM5(st, rng)
    # Idle with no server (manual idle command) and S0
    v.advance(120, "idle:no_server", power=2.85, cpu=3)
    v.advance(25, "membw", power=6.1, cpu=398)
    v.advance(10, "gap", power=3.0)
    # S1
    v.advance(120, "idle:server_no_model", power=2.95, cpu=4)
    v.loaded = True
    v.advance(4, "load", power=4.5, cpu=110)
    v.advance(120, "idle:model_loaded", power=3.1, cpu=5)
    # S2
    if st["load"]:
        for mode, base in (("cold", st["load"][0]), ("warm", st["load"][1])):
            for rep in range(3):
                v.n += 1
                d = base * v.j(0.04)
                v.advance(d, "load", f"S2-{v.n:05d}", power=4.4, cpu=105)
                v.res.append({"id": f"S2-{v.n:05d}", "suite": "S2_load", "variant": mode, "rep": rep,
                              "warmup": False, "load_client_s": d, "load_backend_s": d * 0.96,
                              "ram_delta_mb": 980 + rng.uniform(-6, 6)})
                v.advance(3, "gap", power=3.0)
    # S3, S4, S5
    W, R = 1, 5
    for p in plan["s3"]["prompt_tokens"]:
        for i in range(W + R):
            v.request("S3_prefill", f"P{p}_N1", p + rng.choice([-1, 0, 0, 1]), 1, rep=i - W,
                      warmup=i < W, p_target=p, n_target=1)
    for n in plan["s4"]["output_tokens"]:
        for i in range(W + R):
            v.request("S4_generation", f"P32_N{n}", 32, n, rep=i - W, warmup=i < W, p_target=32,
                      n_target=n)
    for p in plan["s5"]["prompt_tokens"]:
        for n in plan["s5"]["output_tokens"]:
            for i in range(1 + 3):
                v.request("S5_matrix", f"P{p}_N{n}", p, n, rep=i - 1, warmup=i < 1, p_target=p,
                          n_target=n)
    # S6
    for qi in range(5):
        cold = SYS_TOK + Q_TOK[qi] + 15
        for i in range(1 + 3):
            v.request("S6_kv_cache", "cold", cold, N_NAT[qi], rep=i - 1, warmup=i < 1, kv="cold",
                      question=qi)
        for i in range(1 + 3):
            v.request("S6_kv_cache", "primer", Q_TOK[(qi + 1) % 10] + 15, 1, rep=-1, warmup=True,
                      kv="warm", question=qi, p_total=cold)
            v.request("S6_kv_cache", "warm", Q_TOK[qi] + 15, N_NAT[qi], rep=i - 1,
                      warmup=i < 1, kv="warm", question=qi, cooldown=False, p_total=cold)
    # S7
    v.request("S7_scenario", "production", SYS_TOK + Q_TOK[0] + 15, N_NAT[0], rep=-1, warmup=True,
              kv="production", question=0)
    for rep in range(2):
        for qi in range(10):
            v.request("S7_scenario", "production", Q_TOK[qi] + 15, N_NAT[qi], rep=rep, warmup=False,
                      kv="production", question=qi, p_total=SYS_TOK + Q_TOK[qi] + 15)
    # S8
    v.soak_t = v.t
    i = 0
    while v.t - v.soak_t < 600:
        v.request("S8_soak", "P64_N128", 64, 128, rep=i, warmup=False, p_target=64,
                  n_target=128, cooldown=False)
        i += 1
    v.advance(5, "gap", power=3.1)

    run_id = f"DEMO_{st['board']}_{st['backend']}_{st['stage']}"
    d = out / run_id
    d.mkdir(parents=True, exist_ok=True)
    hdr = (["t_mono", "t_wall", "phase", "req_id", "cpu_total"] + [f"cpu{i}" for i in range(4)] +
           ["ram_used_mb", "swap_used_mb", "rss_server_mb", "temp_c", "freq_mhz", "power_w", "throttled"])
    with (d / "samples.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(v.rows)
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in v.res) + "\n")
    card = {"system": {"board": st["board_name"], "board_slug": st["board"],
                       "kernel": "6.12.47+rpt-rpi-2712" if st["board"] == "pi5" else "6.12.47+rpt-rpi-v8",
                       "cpu_model": st["cpu"], "cpu_count": 4,
                       "governor": "ondemand", "freq_max_mhz": st["freq"], "ram_total_mb": 8063,
                       "swap_total_mb": 512, "pmic_ext5v_v": 5.08 if st["board"] == "pi5" else None,
                       "throttled_at_start": {"hex": "0x0", "bits": []},
                       "storage": {"mount": "/dev/mmcblk0p2 ext4", "kind": "emmc/sd", "free_gb": 21.4}},
            "backend": {"backend": st["backend"], "model": st["model"], "version": st["version"],
                        "show": {"details": {"format": "gguf", "family": "llama",
                                             "parameter_size": "1.2B",
                                             "quantization_level": st["quant"]},
                                 "model_info": {"general.architecture": "llama",
                                                "general.parameter_count": 1235814400}}},
            "model_path": f"/models/{st['model']}", "gguf": {**st["gguf"], "path": "(demo)"}}
    (d / "card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    (d / "extras.json").write_text(json.dumps({
        "membw": {"method": "numpy sum float64, 4 processes x 64 MB", "gbps_best": st["membw"], "gbps_median": round(st["membw"] * 0.97, 2)},
        "storage_read": {"mbps": 168.0, "seconds": 4.8} if st["load"] else {},
        "prompt_calibration": {"a_tokens": 6.2, "b_tokens_per_number": 2.0}}, indent=2))
    meta = {"run_id": run_id, "stage": st["stage"], "board": st["board"], "note": st["note"], "bench_version": bench.BENCH_VERSION,
            "backend": st["backend"], "model": st["model"], "method_version": plan["method_version"],
            "method_hash": bench.method_hash(plan, ROOT), "plan": plan,
            "started": {"pi5_baseline": "2026-10-20T10:00:00", "pi4_baseline": "2026-10-21T09:15:00"}.get(stage, "2026-11-04T11:00:00"),
            "status": "complete", "is_demo": True, "swap_max_mb": 0.0, "idle_temp_c": 47.5}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    report.build(d)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs_demo"))
    a = ap.parse_args()
    for i, s in enumerate(STAGES):
        simulate(s, Path(a.out), seed=7 + i)
