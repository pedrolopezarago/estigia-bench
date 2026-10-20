"""estigia-bench system monitor.

Samples in a separate thread, at a fixed interval, and writes one row per
sample to samples.csv. Every row carries the current phase (idle:*, prefill,
decode, gap, load, cooldown...) and the id of the request in flight, so that
any metric can later be averaged per phase.

Power:
  * pmic  -> `vcgencmd pmic_read_adc` (CM5 / Pi 5). Sums V x I of every rail.
             It is an ESTIMATE: it excludes losses and peripherals. Calibrate
             against an external meter (calib_a, calib_b in config.yaml).
  * hwmon -> sysfs file of an INA219/INA226 (power1_input, in µW).
  * none  -> no power measurement.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
CPUFREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
_PMIC_RE = re.compile(r"^\s*(\S+)_([AV])\s+(?:current|volt)\(\d+\)=([\d.]+)[AV]\s*$")


def _read_float(p: Path, div: float = 1.0) -> Optional[float]:
    try:
        return float(p.read_text().strip()) / div
    except Exception:  # noqa: BLE001
        return None


def _vcgencmd(*args: str) -> Optional[str]:
    if not shutil.which("vcgencmd"):
        return None
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True,
                              timeout=2).stdout
    except Exception:  # noqa: BLE001
        return None


def read_throttled() -> Optional[str]:
    out = _vcgencmd("get_throttled")
    if out and "=" in out:
        return out.strip().split("=", 1)[1]
    return None


def decode_throttled(hexstr: Optional[str]) -> list[str]:
    """Translate the bits of get_throttled into text."""
    if not hexstr:
        return []
    v = int(hexstr, 16)
    names = {0: "under-voltage now", 1: "frequency capped now", 2: "throttling now",
             3: "soft temperature limit now", 16: "under-voltage occurred",
             17: "frequency cap occurred", 18: "throttling occurred",
             19: "soft temperature limit occurred"}
    return [txt for bit, txt in names.items() if v & (1 << bit)]


def read_pmic_power() -> Optional[float]:
    out = _vcgencmd("pmic_read_adc")
    if not out:
        return None
    rails: dict[str, dict[str, float]] = {}
    for line in out.splitlines():
        m = _PMIC_RE.match(line)
        if m:
            rails.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    total = sum(r["A"] * r["V"] for r in rails.values() if "A" in r and "V" in r)
    return total if rails else None


def read_pmic_input_voltage() -> Optional[float]:
    out = _vcgencmd("pmic_read_adc", "EXT5V_V")
    if out:
        m = re.search(r"=([\d.]+)V", out)
        if m:
            return float(m.group(1))
    return None


class PowerReader:
    def __init__(self, cfg: dict):
        self.source = (cfg or {}).get("source", "pmic")
        self.path = Path(cfg["hwmon_path"]) if cfg and cfg.get("hwmon_path") else None
        self.a = float((cfg or {}).get("calib_a", 1.0))
        self.b = float((cfg or {}).get("calib_b", 0.0))

    def read(self) -> Optional[float]:
        raw = None
        if self.source == "pmic":
            raw = read_pmic_power()
        elif self.source == "hwmon" and self.path:
            raw = _read_float(self.path, 1e6)
        if raw is None:
            return None
        return self.a * raw + self.b


def read_temp() -> Optional[float]:
    return _read_float(THERMAL, 1000.0)


class SysMonitor(threading.Thread):
    def __init__(self, csv_path: Path, interval: float = 0.5, power_cfg: dict | None = None,
                 proc_names: tuple[str, ...] = (), throttle_every_s: float = 2.0,
                 clock_offset: float = 0.0):
        super().__init__(daemon=True)
        self.csv_path = Path(csv_path)
        self.interval = interval
        self.power = PowerReader(power_cfg or {})
        self.proc_names = proc_names
        self.throttle_every_s = throttle_every_s
        self.phase = "start"
        self.req_id = ""
        self._halt = threading.Event()
        self._procs: list[psutil.Process] = []
        self._procs_t = 0.0
        self._last_thr: Optional[str] = None
        self._last_thr_t = 0.0
        self.ncpu = psutil.cpu_count(logical=True) or 1
        self.swap_max_mb = 0.0

    # Labels set by the test runner.
    def set_phase(self, phase: str, req_id: str | None = None):
        self.phase = phase
        if req_id is not None:
            self.req_id = req_id

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def _server_rss_mb(self) -> Optional[float]:
        now = time.monotonic()
        if now - self._procs_t > 5 or not self._procs:
            self._procs = []
            for p in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = (p.info["name"] or "").lower()
                    cmd = " ".join(p.info["cmdline"] or []).lower()
                    if any(n in name or n in cmd for n in self.proc_names):
                        if "bench.py" not in cmd:
                            self._procs.append(p)
                except Exception:  # noqa: BLE001
                    continue
            self._procs_t = now
        if not self._procs:
            return None
        total = 0
        for p in list(self._procs):
            try:
                total += p.memory_info().rss
            except Exception:  # noqa: BLE001
                self._procs.remove(p)
        return total / 2**20

    def _row(self) -> list:
        t_mono = time.perf_counter()
        cpus = psutil.cpu_percent(percpu=True)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        swap_mb = sw.used / 2**20
        self.swap_max_mb = max(self.swap_max_mb, swap_mb)
        freq = _read_float(CPUFREQ, 1000.0)
        if freq is None:
            try:
                freq = psutil.cpu_freq().current
            except Exception:  # noqa: BLE001
                freq = None
        if t_mono - self._last_thr_t >= self.throttle_every_s:
            self._last_thr = read_throttled()
            self._last_thr_t = t_mono
        rss = self._server_rss_mb()
        return [f"{t_mono:.4f}", f"{time.time():.3f}", self.phase, self.req_id,
                f"{sum(cpus):.1f}", *[f"{c:.1f}" for c in cpus],
                f"{(vm.total - vm.available) / 2**20:.1f}", f"{swap_mb:.1f}",
                "" if rss is None else f"{rss:.1f}",
                _fmt(read_temp()), _fmt(freq), _fmt(self.power.read(), 3),
                self._last_thr or ""]

    def header(self) -> list[str]:
        return (["t_mono", "t_wall", "phase", "req_id", "cpu_total"] +
                [f"cpu{i}" for i in range(self.ncpu)] +
                ["ram_used_mb", "swap_used_mb", "rss_server_mb", "temp_c", "freq_mhz",
                 "power_w", "throttled"])

    def run(self):
        new = not self.csv_path.exists()
        psutil.cpu_percent(percpu=True)  # first call: initializes the counters
        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.header())
            nxt = time.perf_counter()
            while not self._halt.is_set():
                nxt += self.interval
                try:
                    w.writerow(self._row())
                    f.flush()
                except Exception as e:  # noqa: BLE001
                    print(f"[sysmon] sample lost: {e}")
                delay = nxt - time.perf_counter()
                if delay > 0:
                    self._halt.wait(delay)
                else:
                    nxt = time.perf_counter()


def _fmt(v, nd=1):
    return "" if v is None else f"{v:.{nd}f}"
