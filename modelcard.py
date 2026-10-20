"""System and model card (baseline sections 2 and 3).

  * system_card()      hardware, OS, governor, config.txt, storage.
  * ollama_blob_path() path of the GGUF that Ollama uses for a given tag.
  * gguf_histogram()   type of every GGUF tensor -> effective quantization.
  * loader_log()       llama.cpp loader lines from the Ollama journal.
  * membw()            memory read bandwidth (multi-process).
  * storage_read()     sequential read of the model file (cache dropped).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import psutil


def _sh(cmd: list[str], timeout=10) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or r.stderr.strip()
    except Exception:  # noqa: BLE001
        return None


def _read(p: str) -> Optional[str]:
    try:
        return Path(p).read_text(errors="ignore").strip("\x00\n ")
    except Exception:  # noqa: BLE001
        return None


def detect_board() -> str:
    """pi5 | pi4 | cm5 | cm4 | other (from /proc/device-tree/model)."""
    m = (_read("/proc/device-tree/model") or "").lower()
    for key, slug in (("compute module 5", "cm5"), ("compute module 4", "cm4"),
                      ("raspberry pi 5", "pi5"), ("raspberry pi 4", "pi4")):
        if key in m:
            return slug
    return "other"


def system_card(model_path: Optional[str] = None) -> dict:
    vm = psutil.virtual_memory()
    card = {
        "board": _read("/proc/device-tree/model"),
        "board_slug": detect_board(),
        "hostname": platform.node(),
        "os": _read("/etc/os-release"),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_model": None,
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "freq_max_mhz": None,
        "ram_total_mb": round(vm.total / 2**20),
        "ram_available_mb": round(vm.available / 2**20),
        "swap_total_mb": round(psutil.swap_memory().total / 2**20),
        "vc_config": None,
        "pmic_ext5v_v": None,
        "throttled_at_start": None,
        "storage": None,
    }
    fmax = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    card["freq_max_mhz"] = int(fmax) / 1000 if fmax and fmax.isdigit() else None
    lscpu = _sh(["lscpu"])
    if lscpu:
        m = re.search(r"Model name:\s*(.+)", lscpu)
        card["cpu_model"] = m.group(1).strip() if m else None
    cfg = _sh(["vcgencmd", "get_config", "int"])
    if cfg:
        keep = ("arm_freq", "over_voltage", "arm_boost", "gpu_mem", "total_mem", "sdram_freq")
        card["vc_config"] = {k: v for k, v in (l.split("=", 1) for l in cfg.splitlines() if "=" in l)
                             if k in keep}
    try:
        from sysmon import read_pmic_input_voltage, read_throttled, decode_throttled
        card["pmic_ext5v_v"] = read_pmic_input_voltage()
        thr = read_throttled()
        card["throttled_at_start"] = {"hex": thr, "bits": decode_throttled(thr)}
    except Exception:  # noqa: BLE001
        pass
    if model_path:
        card["storage"] = storage_info(model_path)
    return card


def storage_info(path: str) -> dict:
    out = {"path": path}
    src = _sh(["findmnt", "-n", "-o", "SOURCE,FSTYPE", "--target", path])
    out["mount"] = src
    try:
        du = shutil.disk_usage(path)
        out["free_gb"] = round(du.free / 1e9, 2)
        out["total_gb"] = round(du.total / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    if src:
        dev = src.split()[0]
        out["kind"] = ("nvme" if "nvme" in dev else "emmc/sd" if "mmcblk" in dev else
                       "usb/sata" if dev.startswith("/dev/sd") else dev)
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def ollama_blob_path(show: dict) -> Optional[str]:
    """Extract the GGUF path from the FROM line of the Modelfile in /api/show."""
    mf = (show or {}).get("modelfile", "")
    for line in mf.splitlines():
        if line.startswith("FROM "):
            p = line[5:].strip()
            if p.startswith("/") and os.path.exists(p):
                return p
    return None


def gguf_histogram(path: str) -> dict:
    """Tensor types by count and by bytes. Requires `pip install gguf`."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return {"error": "the gguf package is missing (pip install gguf)"}
    try:
        r = GGUFReader(path)
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not read the GGUF: {e}"}
    hist: dict[str, dict] = defaultdict(lambda: {"tensors": 0, "bytes": 0, "params": 0})
    for t in r.tensors:
        name = t.tensor_type.name
        hist[name]["tensors"] += 1
        hist[name]["bytes"] += int(t.n_bytes)
        hist[name]["params"] += int(t.n_elements)
    total_b = sum(h["bytes"] for h in hist.values()) or 1
    total_p = sum(h["params"] for h in hist.values()) or 1
    rows = sorted(({"type": k, **v, "pct_bytes": round(100 * v["bytes"] / total_b, 2)}
                   for k, v in hist.items()), key=lambda x: -x["bytes"])
    meta = {}
    for key in ("general.architecture", "general.name", "general.file_type",
                "general.quantization_version", "general.size_label"):
        f = r.fields.get(key)
        if f is not None:
            try:
                v = f.parts[f.data[0]]
                meta[key] = v.tobytes().decode() if v.dtype.kind == "u" and v.dtype.itemsize == 1 \
                    else v.tolist()[0]
            except Exception:  # noqa: BLE001
                pass
    return {"path": path, "file_bytes": os.path.getsize(path), "tensor_bytes": total_b,
            "params": total_p, "bits_per_weight": round(8 * total_b / total_p, 3),
            "types": rows, "dominant": rows[0]["type"] if rows else None, "meta": meta}


def loader_log(lines=600) -> list[str]:
    """Loader lines (tensor types, ARM repacking) from the Ollama journal."""
    out = _sh(["journalctl", "-u", "ollama", "--no-pager", "-n", str(lines)], timeout=15)
    if not out:
        return []
    pat = re.compile(r"(llama_model_loader: - type|repack|CPU_REPACK|file type|model size|"
                     r"n_threads|system_info|KV self size|llama_kv_cache)", re.I)
    return [l[-220:] for l in out.splitlines() if pat.search(l)][-60:]


# --------------------------------------------------------------------------- #
# Memory read bandwidth, multi-process so it does not depend on the GIL
# --------------------------------------------------------------------------- #
def _bw_worker(mb, reps, barrier, q):
    import numpy as np
    a = np.ones(int(mb * 2**20 / 8), dtype=np.float64)
    a.sum()
    times = []
    for _ in range(reps):
        barrier.wait()
        t0 = time.perf_counter()
        a.sum()
        times.append(time.perf_counter() - t0)
    q.put(times)


def membw(total_mb: int = 256, procs: int = 4, reps: int = 7) -> dict:
    """Sequential read with N processes. Returns the best repetition (GB/s)."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        return {"error": "numpy is missing"}
    per = max(16, total_mb // procs)
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    barrier, q = ctx.Barrier(procs), ctx.Queue()
    ps = [ctx.Process(target=_bw_worker, args=(per, reps, barrier, q)) for _ in range(procs)]
    for p in ps:
        p.start()
    res = [q.get(timeout=300) for _ in ps]
    for p in ps:
        p.join()
    # In each repetition the slowest process wins (they all start together).
    per_rep = [max(r[i] for r in res) for i in range(reps)]
    bytes_total = per * 2**20 * procs
    gbps = [bytes_total / t / 1e9 for t in per_rep[1:]]  # drop the first one
    gbps.sort()
    return {"method": f"numpy sum float64, {procs} processes x {per} MB",
            "gbps_best": round(gbps[-1], 2),
            "gbps_median": round(gbps[len(gbps) // 2], 2), "reps": reps - 1}


def can_drop_caches() -> bool:
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False


def drop_caches() -> bool:
    try:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def storage_read(path: str, chunk_mb: int = 4) -> dict:
    if not drop_caches():
        return {"skipped": "sudo is required to drop the page cache"}
    size = os.path.getsize(path)
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while f.read(chunk_mb * 2**20):
            pass
    dt = time.perf_counter() - t0
    return {"bytes": size, "seconds": round(dt, 3), "mbps": round(size / dt / 1e6, 1)}
