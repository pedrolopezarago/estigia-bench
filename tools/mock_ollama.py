#!/usr/bin/env python3
"""Server that mimics the Ollama API, to test estigia-bench without the CM5.

    python tools/mock_ollama.py --port 11434 --pp 2000 --tg 200

It measures nothing real: it is there to check that the bench works end to
end (streaming, metrics, prefix cache, load/unload).
"""
import argparse
import json
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOK = re.compile(r"\d+|\w+|[^\w\s]")
STATE = {"loaded": False, "cache": [], "lock": threading.Lock()}
ARGS = None
MODEL = "mock:1b-q4_K_M"


def toks(text):
    return ["<s>"] + TOK.findall(text)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # like Ollama: streaming with Transfer-Encoding: chunked

    def log_message(self, *a):
        pass

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/version":
            return self._json({"version": "0.0.0-mock"})
        if self.path == "/api/tags":
            return self._json({"models": [{"name": MODEL, "model": MODEL, "size": 807_694_464,
                                           "digest": "0" * 64}]})
        if self.path == "/api/ps":
            return self._json({"models": [{"name": MODEL, "model": MODEL, "size": 1_100_000_000,
                                           "context_length": 4096}] if STATE["loaded"] else []})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/show":
            return self._json({
                "modelfile": f"FROM {ARGS.gguf}\n" if ARGS.gguf else "FROM /nonexistent\n",
                "details": {"format": "gguf", "family": "llama", "parameter_size": "1.2B",
                            "quantization_level": "Q4_K_M"},
                "model_info": {"general.architecture": "llama", "general.parameter_count": 1235814400,
                               "llama.context_length": 131072, "llama.block_count": 16}})
        if self.path in ("/api/generate", "/api/chat"):
            return self.gen(body, chat=self.path == "/api/chat")
        self._json({"error": "not found"}, 404)

    def gen(self, body, chat):
        if chat:
            text = "\n".join(m["content"] for m in body.get("messages", []))
        else:
            text = body.get("prompt", "")
            if not body.get("raw") and body.get("system"):
                text = body["system"] + "\n" + text
        load_ns = 0
        with STATE["lock"]:
            if not text:
                if body.get("keep_alive") == 0:
                    STATE["loaded"] = False
                    STATE["cache"] = []
                    return self._json({"done": True, "done_reason": "unload"})
                if not STATE["loaded"]:
                    time.sleep(ARGS.load)
                    load_ns = int(ARGS.load * 1e9)
                    STATE["loaded"] = True
                return self._json({"done": True, "load_duration": load_ns})
            if not STATE["loaded"]:
                time.sleep(ARGS.load)
                load_ns = int(ARGS.load * 1e9)
                STATE["loaded"] = True
            t = toks(text)
            common = 0
            for a, b in zip(STATE["cache"], t):
                if a != b:
                    break
                common += 1
            evaluated = max(1, len(t) - common)
            STATE["cache"] = t

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            t0 = time.perf_counter()
            time.sleep(evaluated / ARGS.pp)
            p_ns = int((time.perf_counter() - t0) * 1e9)
            n = int((body.get("options") or {}).get("num_predict", 64))
            nums = re.findall(r"\d+", text)
            nxt = int(nums[-1]) + 1 if nums and not chat else 1
            limit = n if not chat else min(n, 40)
            t1 = time.perf_counter()
            for i in range(limit):
                time.sleep(1 / ARGS.tg)
                piece = f" {nxt + i}," if not chat else (" word" if i % 2 else " other")
                msg = {"model": MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                       "done": False}
                if chat:
                    msg["message"] = {"role": "assistant", "content": piece}
                else:
                    msg["response"] = piece
                self._chunk((json.dumps(msg) + "\n").encode())
            e_ns = int((time.perf_counter() - t1) * 1e9)
            final = {"model": MODEL, "done": True, "done_reason": "length" if limit == n else "stop",
                     "total_duration": p_ns + e_ns + load_ns, "load_duration": load_ns,
                     "prompt_eval_count": evaluated, "prompt_eval_duration": p_ns,
                     "eval_count": limit, "eval_duration": e_ns}
            if chat:
                final["message"] = {"role": "assistant", "content": ""}
            else:
                final["response"] = ""
            self._chunk((json.dumps(final) + "\n").encode())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--pp", type=float, default=2000, help="simulated prompt tokens/s")
    ap.add_argument("--tg", type=float, default=200, help="simulated generation tokens/s")
    ap.add_argument("--load", type=float, default=0.5, help="simulated load seconds")
    ap.add_argument("--gguf", help="path to a real GGUF to exercise the histogram")
    ARGS = ap.parse_args()
    print(f"Ollama mock on :{ARGS.port} (pp {ARGS.pp} tok/s, tg {ARGS.tg} tok/s)")
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
