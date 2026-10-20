"""estigia-bench inference clients.

Ollama and llama.cpp (llama-server) return the same result dictionary, so that
the baseline (Ollama) and phase 1 (llama.cpp) are measured with the same
method. Two families of timings are recorded:

  * client: measured here with time.perf_counter() over the stream. They do
    not depend on how each engine counts its own internal time -> official
    figure for comparing engines.
  * backend: the ones reported by the server itself (prompt_eval_*, eval_*,
    timings). Used as a cross-check.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

import requests

PhaseCb = Optional[Callable[[str], None]]


def _now() -> float:
    return time.perf_counter()


def _ns_to_s(v) -> Optional[float]:
    return None if v is None else v / 1e9


def _client_stats(t_send: float, token_times: list[float], t_end: float) -> dict:
    """Client-side metrics derived from the timestamp of every chunk."""
    out = {"t_send": t_send, "t_end": t_end, "chunks": len(token_times)}
    if token_times:
        t_first = token_times[0]
        out["t_first"] = t_first
        out["ttft_s"] = t_first - t_send
        out["e2e_s"] = token_times[-1] - t_send
        if len(token_times) > 1:
            span = token_times[-1] - t_first
            out["tg_tps_client"] = (len(token_times) - 1) / span if span > 0 else None
            itl = sorted((b - a) * 1000 for a, b in zip(token_times, token_times[1:]))
            out["itl_p50_ms"] = itl[len(itl) // 2]
            out["itl_p95_ms"] = itl[min(len(itl) - 1, int(round(0.95 * (len(itl) - 1))))]
    else:
        out["t_first"] = None
        out["ttft_s"] = None
        out["e2e_s"] = t_end - t_send
    return out


class Backend:
    name = "base"
    # Process names whose RSS the monitor adds up as "the inference process".
    proc_names: tuple[str, ...] = ()

    def info(self) -> dict:
        raise NotImplementedError

    def generate(self, prompt: str, n_predict: int, *, system: str | None = None,
                 chat: bool = False, raw: bool = True, cache: bool = True,
                 ignore_eos: bool = True, options: dict | None = None,
                 phase_cb: PhaseCb = None) -> dict:
        raise NotImplementedError

    def count_tokens(self, text: str) -> Optional[int]:
        return None

    def load(self) -> dict:
        return {}

    def unload(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
class OllamaBackend(Backend):
    name = "ollama"
    proc_names = ("ollama",)

    def __init__(self, url: str, model: str, base_options: dict, keep_alive="30m",
                 timeout: float = 900):
        self.url = url.rstrip("/")
        self.model = model
        self.base_options = {k: v for k, v in base_options.items() if v is not None}
        self.keep_alive = keep_alive
        self.timeout = timeout

    # -- information ------------------------------------------------------ #
    def _get(self, path):
        r = requests.get(self.url + path, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body, timeout=60):
        r = requests.post(self.url + path, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def info(self) -> dict:
        out = {"backend": "ollama", "url": self.url, "model": self.model}
        for key, fn in (("version", lambda: self._get("/api/version")),
                        ("show", lambda: self._post("/api/show", {"model": self.model})),
                        ("tags", lambda: self._get("/api/tags")),
                        ("ps", lambda: self._get("/api/ps"))):
            try:
                out[key] = fn()
            except Exception as e:  # noqa: BLE001
                out[key] = {"error": str(e)}
        # Without the full tensor list (it is huge): read from the GGUF instead.
        if isinstance(out.get("show"), dict):
            out["show"].pop("tensors", None)
            out["show"].pop("license", None)
        return out

    # -- load / unload --------------------------------------------------- #
    def unload(self) -> None:
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=120)
        except Exception:  # noqa: BLE001
            pass
        # Wait until /api/ps stops listing it.
        for _ in range(60):
            try:
                ps = self._get("/api/ps").get("models", [])
                if not any(m.get("name") == self.model or m.get("model") == self.model for m in ps):
                    return
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.5)

    def load(self) -> dict:
        t0 = _now()
        body = {"model": self.model, "keep_alive": self.keep_alive,
                "options": self.base_options}
        res = self._post("/api/generate", body, timeout=self.timeout)
        t1 = _now()
        return {"load_client_s": t1 - t0, "load_backend_s": _ns_to_s(res.get("load_duration"))}

    # -- generation ------------------------------------------------------ #
    def generate(self, prompt, n_predict, *, system=None, chat=False, raw=True, cache=True,
                 ignore_eos=True, options=None, phase_cb=None) -> dict:
        opts = dict(self.base_options)
        opts.update(options or {})
        opts["num_predict"] = n_predict
        if chat:
            path = "/api/chat"
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            body = {"model": self.model, "messages": msgs, "stream": True,
                    "options": opts, "keep_alive": self.keep_alive}
        else:
            path = "/api/generate"
            body = {"model": self.model, "prompt": prompt, "stream": True, "raw": raw,
                    "options": opts, "keep_alive": self.keep_alive}
            if system and not raw:
                body["system"] = system
        # Ollama exposes neither "ignore_eos" nor a way to disable the prefix
        # cache: cold-cache control is done with a unique prefix (see bench.py).

        token_times, parts, final = [], [], {}
        if phase_cb:
            phase_cb("prefill")
        t_send = _now()
        with requests.post(self.url + path, json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                piece = msg.get("response") if not chat else (msg.get("message") or {}).get("content", "")
                if piece:
                    if not token_times and phase_cb:
                        phase_cb("decode")
                    token_times.append(_now())
                    parts.append(piece)
                if msg.get("done"):
                    final = msg
                    break
        t_end = _now()
        if phase_cb:
            phase_cb("gap")

        res = _client_stats(t_send, token_times, t_end)
        p_n = final.get("prompt_eval_count")
        p_s = _ns_to_s(final.get("prompt_eval_duration"))
        g_n = final.get("eval_count")
        g_s = _ns_to_s(final.get("eval_duration"))
        res.update({
            "text": "".join(parts),
            "p_actual": p_n,
            "n_actual": g_n,
            "prompt_eval_s": p_s,
            "eval_s": g_s,
            "load_backend_s": _ns_to_s(final.get("load_duration")),
            "total_backend_s": _ns_to_s(final.get("total_duration")),
            "pp_tps_backend": (p_n / p_s) if p_n and p_s else None,
            "tg_tps_backend": (g_n / g_s) if g_n and g_s else None,
            "done_reason": final.get("done_reason"),
        })
        return res


# --------------------------------------------------------------------------- #
# llama.cpp (llama-server)
# --------------------------------------------------------------------------- #
class LlamaCppBackend(Backend):
    name = "llamacpp"
    proc_names = ("llama-server", "llama_server")

    def __init__(self, url: str, model: str, base_options: dict, timeout: float = 900, **_):
        self.url = url.rstrip("/")
        self.model = model  # informational: the server already has its model loaded
        self.base_options = {k: v for k, v in base_options.items() if v is not None}
        self.timeout = timeout

    def info(self) -> dict:
        out = {"backend": "llamacpp", "url": self.url, "model": self.model}
        for key, path in (("props", "/props"), ("models", "/v1/models"), ("health", "/health")):
            try:
                r = requests.get(self.url + path, timeout=30)
                out[key] = r.json()
            except Exception as e:  # noqa: BLE001
                out[key] = {"error": str(e)}
        return out

    def count_tokens(self, text: str) -> Optional[int]:
        try:
            r = requests.post(self.url + "/tokenize", json={"content": text, "add_special": True},
                              timeout=60)
            return len(r.json().get("tokens", []))
        except Exception:  # noqa: BLE001
            return None

    def _apply_template(self, system, prompt) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        try:
            r = requests.post(self.url + "/apply-template", json={"messages": msgs}, timeout=30)
            r.raise_for_status()
            return r.json()["prompt"]
        except Exception:  # noqa: BLE001
            return (system + "\n\n" if system else "") + prompt

    def generate(self, prompt, n_predict, *, system=None, chat=False, raw=True, cache=True,
                 ignore_eos=True, options=None, phase_cb=None) -> dict:
        text_in = self._apply_template(system, prompt) if chat else prompt
        body = {"prompt": text_in, "n_predict": n_predict, "stream": True,
                "cache_prompt": cache, "ignore_eos": bool(ignore_eos and not chat)}
        o = dict(self.base_options)
        o.update(options or {})
        for k_src, k_dst in (("temperature", "temperature"), ("seed", "seed")):
            if k_src in o:
                body[k_dst] = o[k_src]

        token_times, parts, final = [], [], {}
        if phase_cb:
            phase_cb("prefill")
        t_send = _now()
        with requests.post(self.url + "/completion", json=body, stream=True,
                           timeout=self.timeout) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line or not raw_line.startswith(b"data:"):
                    continue
                msg = json.loads(raw_line[5:].strip())
                piece = msg.get("content", "")
                if piece:
                    if not token_times and phase_cb:
                        phase_cb("decode")
                    token_times.append(_now())
                    parts.append(piece)
                if msg.get("stop"):
                    final = msg
                    break
        t_end = _now()
        if phase_cb:
            phase_cb("gap")

        res = _client_stats(t_send, token_times, t_end)
        tm = final.get("timings", {}) or {}
        p_n, p_ms = tm.get("prompt_n"), tm.get("prompt_ms")
        g_n, g_ms = tm.get("predicted_n"), tm.get("predicted_ms")
        res.update({
            "text": "".join(parts),
            "p_actual": final.get("tokens_evaluated", p_n),
            "p_evaluated": p_n,
            "p_cached": tm.get("cache_n", final.get("tokens_cached")),
            "n_actual": g_n,
            "prompt_eval_s": p_ms / 1000 if p_ms else None,
            "eval_s": g_ms / 1000 if g_ms else None,
            "load_backend_s": 0.0,
            "pp_tps_backend": tm.get("prompt_per_second"),
            "tg_tps_backend": tm.get("predicted_per_second"),
            "done_reason": final.get("stop_type"),
        })
        return res


def make_backend(cfg: dict, model: str) -> Backend:
    b = cfg["backend"]
    inf = cfg.get("inference", {})
    base = {"temperature": inf.get("temperature", 0), "seed": inf.get("seed", 42),
            "num_ctx": inf.get("num_ctx"), "num_thread": inf.get("num_thread")}
    if b["type"] == "ollama":
        return OllamaBackend(b["url"], model, base, keep_alive=inf.get("keep_alive", "30m"))
    if b["type"] == "llamacpp":
        base.pop("num_ctx"), base.pop("num_thread")  # fixed when llama-server starts
        return LlamaCppBackend(b["url"], model, base)
    raise ValueError(f"unknown backend: {b['type']}")
