# estigia-bench

Reproducible performance bench for a small LLM served on a Raspberry Pi, with per-phase latency, resource and energy measurements.

Built for the AI payload of the **Estigia** mission (Pluton UPV, academic year 2026-27), but nothing in it is mission-specific: it measures any model served by **Ollama** or **llama.cpp** on a **Raspberry Pi 5, Pi 4 or CM5**, and ships an offline dashboard (`dashboard.html`) to compare runs.

What it gives you, with a fixed method so two runs are actually comparable:

* prompt processing and token generation timed **separately**, client-side and backend-side;
* CPU, RAM, RSS, temperature, frequency and throttling sampled every 0.5 s and tagged by phase;
* energy per response and per generated token, integrated from PMIC or external-meter power;
* a measured memory-bandwidth roofline, so you know how much headroom software can still buy;
* a `method_hash` that changes whenever the method changes — the dashboard refuses to compare across it.

## Install

Python 3.10 or newer. On each board:

```bash
git clone https://github.com/pedrolopezarago/estigia-bench.git
cd estigia-bench
python3 -m venv ~/estigia-venv && source ~/estigia-venv/bin/activate
pip install -r requirements.txt
```

Before the first real session, paste your **actual** system prompt and ten representative questions into `prompts/reference_v1.json`. While the placeholder text is still there, S6 and S7 are skipped (use `--allow-placeholder` to exercise them anyway).

## Usage

```bash
# 1. System and model card (prints the path of the GGUF that Ollama uses)
python bench.py card --model <ollama_tag>

# 2. Check that everything works (~10 min; not comparable)
python bench.py run --plan config_quick.yaml --stage smoke --model <tag>

# 3. Baseline with Ollama (full plan, ~1.5 h on a Pi 5)
python bench.py run --stage baseline --model <tag>

# 4. Idle with no inference server (appended to the previous run)
sudo systemctl stop ollama
python bench.py idle --out runs/<run_id> --label no_server --seconds 120
sudo systemctl start ollama
python bench.py report runs/<run_id>

# 5. First comparison: llama.cpp with THE SAME GGUF
sudo systemctl stop ollama
llama-server -m <gguf_path> -c 4096 -t 4 --port 8080
python bench.py run --backend llamacpp --backend-url http://127.0.0.1:8080 \
    --gguf <gguf_path> --stage post_llamacpp
```

`sudo` on step 3 adds the cold load and the storage read (it drops the page cache). Without `sudo`, those two measurements are skipped and everything else works the same.

Each run lands in `runs/<date-time>_<board>_<engine>_<model>_<stage>/`. The board is detected automatically; force it with `--board pi5`.

## What to compare, and how

Change **one axis per run**: engine (same GGUF), board (same engine and GGUF), base model (same board, engine and quantization type) or one optimization. The dashboard detects which axis changed and warns you if several changed at once, or if the `method_hash` does not match.

Across different models, compare **full response time (s)** and **words/s**: tok/s depends on the tokenizer.

## Power

* **Pi 5 / CM5:** estimated from the PMIC (`vcgencmd pmic_read_adc`). Calibrate once against a USB-C meter and set `calib_a` and `calib_b` in `config.yaml`.
* **Pi 4:** has no PMIC. Log with an external USB-C meter and drop the CSV into the run folder as `power_external.csv`, with columns `t_wall` (epoch or ISO 8601) and `power_w` (or `voltage_v` and `current_a`). Re-run `report`.
* **Comparing energy across boards:** use the same external meter on both, with clocks synchronized over NTP.
* INA219/INA226 sensor with an hwmon driver: `power.source: hwmon` plus the path to `power1_input`.

## Test suites

| Suite | What it measures |
| --- | --- |
| `S0_system` | memory bandwidth, storage read |
| `S1_idle` | idle power in each server state |
| `S2_load` | cold and warm model load time, RAM added |
| `S3_prefill` | prompt processing speed and TTFT versus prompt length |
| `S4_generation` | generation speed versus output length |
| `S5_matrix` | prompt × output matrix (end to end, energy, TTFT) |
| `S6_kv_cache` | cold versus warm KV cache on the reference prompt |
| `S7_scenario` | the production scenario exactly as it runs |
| `S8_soak` | 10 min of continuous generation: thermal drop |
| `S9_threads` | optional: scaling with the number of threads |

## Files in each run

| File | Contents |
| --- | --- |
| `meta.json` | versions, board, engine, model, parameters, `method_hash` |
| `card.json` | system and model card, GGUF tensor types |
| `results.jsonl` | one line per request (client and backend timings, flags) |
| `samples.csv` | CPU, RAM, temperature, frequency, throttling and power every 0.5 s, with the phase |
| `extras.json` | memory bandwidth, storage read, calibration |
| `summary.json` | everything aggregated (median, quartiles, n) |
| `baseline_tables.md` | tables ready for the baseline document |
| `bundle.json` | the file the dashboard loads |

## Dashboard

Open `dashboard.html` in any browser (it works offline, with no build step and no server) and drop in up to four `bundle.json` files. The first one is the reference. Files are read in the browser and never uploaded anywhere. It opens with embedded demo data — click **Remove demo** to clear it.

## Try it without a board

```bash
python tools/mock_ollama.py --port 11434 &          # simulated Ollama
python bench.py run --plan config_quick.yaml --model mock:1b --allow-placeholder
python tools/demo_run.py --out runs_demo            # demo runs for the dashboard
python tools/build_dashboard.py runs_demo/*/bundle.json   # re-embed them into dashboard.html
```

## Layout

| File | Role |
| --- | --- |
| `bench.py` | CLI and suites S0–S9 |
| `backends.py` | Ollama and llama-server clients with client-side timings |
| `sysmon.py` | system and power monitor |
| `modelcard.py` | card, GGUF histogram, bandwidth |
| `report.py` | aggregation, energy per phase, diagnosis, tables and bundle |
| `config.yaml` / `config_quick.yaml` | full plan (method v1) and quick plan |
| `tools/` | simulated Ollama, demo runs, dashboard builder |

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
