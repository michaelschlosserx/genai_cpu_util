#!/usr/bin/env python3
"""Measure CPU usage while a Neat GenAIServer serves one or more models at once.

Writes cpu.csv (samples) and marks.csv (phase boundaries) per run, then plots
timeline.png and cores.png.
"""
import argparse
import base64
import json
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from genai_cpu_util import charts


def synthetic_png(size):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            rows += bytes((x * 255 // (size - 1), y * 255 // (size - 1), 128))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))


def request_content(model_dir, args):
    """A text prompt, or a text+image prompt if the model has vision. Detected
    from the model's own config, so VLMs and LLMs can be mixed freely."""
    config_path = Path(model_dir) / "devkit" / "vlm_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    # vision detection as in has_vision_capability() (GenAIInternal.cpp)
    if not (config.get("vm_cfg") is not None
            and config.get("mm_cfg") is not None
            and isinstance(config.get("vision_model_name"), str)
            and config["vision_model_name"] != ""):
        return args.text_prompt
    size = (config.get("vm_cfg") or {}).get("image_size")
    if isinstance(size, list):  # some models give [w, h], others a bare int
        size = size[0]
    png = synthetic_png(size if isinstance(size, int) and size > 0 else args.image_size)
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    return [{"type": "text", "text": args.image_prompt},
            {"type": "image_url", "image_url": {"url": uri}}]


def base_url(args):
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    return f"http://{host}:{args.port}"


def wait_until_listening(args):
    while True:
        try:
            with urllib.request.urlopen(f"{base_url(args)}/v1/models", timeout=5):
                return
        except OSError:
            time.sleep(0.5)


def chat(job):
    """One model's request loop. Each model loops independently so all of them
    stay busy for the whole phase -- firing once each leaves the faster model
    idle while the others are still decoding, which is not 'simultaneously'."""
    name, content, args = job
    payload = {"model": name,
               "messages": [{"role": "user", "content": content}],
               "max_tokens": args.max_tokens}
    # built once, outside the loop: re-serialising the base64 image every
    # iteration would burn client CPU inside the measured window
    request = urllib.request.Request(
        f"{base_url(args)}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    for _ in range(args.prompts):
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.load(response)
    return name, body["choices"][0]["message"]["content"]


def one_run(out, args):
    import pyneat as neat  # only on the DevKit, so --help works off it too

    out.mkdir(parents=True, exist_ok=True)
    marks = []

    def mark(tag):
        marks.append((time.time(), tag))

    # a crashed run leaves its sampler alive; a second one truncating the same
    # cpu.csv while the first keeps writing at its old offset fills the file
    # with NULs, so clear stragglers before starting
    subprocess.run(["pkill", "-f", "[g]enai_cpu_util/sampler.py"])
    sampler = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("sampler.py")),
         str(out / "cpu.csv"), str(args.interval)])

    print(f"idle baseline {args.idle_pre}s...", flush=True)
    mark("idle_pre")
    time.sleep(args.idle_pre)

    mark("prep")
    jobs = [(Path(d).resolve().name, request_content(d, args), args)
            for d in args.models]

    print("initializing...", flush=True)
    init_started = time.monotonic()
    mark("construct")
    options = neat.genai.GenAIServerOptions()
    options.host, options.port = args.host, args.port
    server = neat.genai.GenAIServer(options)
    for model_dir, (name, _, _) in zip(args.models, jobs):
        mark(f"load_{name}")
        server.add_model(model_dir, name)  # loads the model
    mark("warmup_bind")
    server.start()  # warms up every model synchronously, then listens
    mark("wait_listen")
    wait_until_listening(args)
    print(f"initialization done in {time.monotonic() - init_started:.1f}s", flush=True)

    print(f"loaded idle {args.idle_loaded}s...", flush=True)
    mark("idle_loaded")
    time.sleep(args.idle_loaded)

    print(f"{len(jobs)} models x {args.prompts} prompts, looping...", flush=True)
    mark("inference")
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(chat, jobs))

    mark("drain")
    time.sleep(args.drain)

    mark("shutdown")
    server.stop()
    mark("end")
    sampler.terminate()

    (out / "marks.csv").write_text("".join(f"{t:.3f},{tag}\n" for t, tag in marks))

    for name, text in results:
        print(f"\n--- {name} ---\n{text}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure CPU usage while a Neat GenAIServer serves one or "
                    "more models at once, split into initialization, inference "
                    "and peak. VLM or LLM is detected per model, so any mix "
                    "works and nothing needs to be declared.",
        epilog="""examples:
  %(prog)s /models/Qwen3-4B
  %(prog)s /models/Qwen2.5-VL-7B /models/Qwen3-4B --repeat 3 --prompts 20
  %(prog)s /models/A /models/B --name overnight --max-tokens 512

output goes to reports/<name>/, per run directory:
  cpu.csv    one row per sample: timestamp, /proc/stat totals, per-core busy/total
  marks.csv  phase boundaries: timestamp, phase name

timeline.png and cores.png are plotted at the end of the run.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("models", nargs="+", metavar="MODEL_DIR",
                        help="one or more model directories to serve together")
    parser.add_argument("--name", metavar="NAME",
                        help="name of this report, created inside --reports-dir "
                             "(default: a timestamp). With --repeat it gets one "
                             "run1..runN subdirectory per run")

    run = parser.add_argument_group("what to measure")
    run.add_argument("--repeat", type=int, default=1, metavar="N",
                     help="reload and rerun N times -- the only way to get spread "
                          "on the initialization numbers (default: %(default)s)")
    run.add_argument("--prompts", type=int, default=1, metavar="M",
                     help="requests per model per run, looped so every model stays "
                          "busy -- use this for inference spread, it is far cheaper "
                          "than --repeat (default: %(default)s)")
    run.add_argument("--max-tokens", type=int, default=128, metavar="T",
                     help="tokens to generate per request (default: %(default)s)")

    timing = parser.add_argument_group("phase durations, seconds")
    timing.add_argument("--idle-pre", type=float, default=5, metavar="S",
                        help="idle baseline before anything is loaded; every other "
                             "number is only meaningful against it "
                             "(default: %(default)s)")
    timing.add_argument("--idle-loaded", type=float, default=5, metavar="S",
                        help="models resident and serving but no traffic -- the "
                             "standing cost of keeping them up (default: %(default)s)")
    timing.add_argument("--drain", type=float, default=5, metavar="S",
                        help="settle time after the last response, to catch work "
                             "leaking past it (default: %(default)s)")
    timing.add_argument("--interval", type=float, default=0.1, metavar="S",
                        help="CPU sampling interval; below ~0.05 the 10 ms clock "
                             "tick dominates (default: %(default)s)")

    server = parser.add_argument_group("server")
    server.add_argument("--host", default="0.0.0.0", help="bind address (default: %(default)s)")
    server.add_argument("--port", type=int, default=9998, help="bind port (default: %(default)s)")
    server.add_argument("--timeout", type=float, default=600, metavar="S",
                        help="per-request timeout (default: %(default)s)")
    server.add_argument("--text-prompt", default="What is the tallest mountain in the world?",
                        help="prompt sent to models without vision")
    server.add_argument("--image-prompt", default="Describe this image in one paragraph.",
                        help="prompt sent alongside the image to models with vision")
    server.add_argument("--image-size", type=int, default=448, metavar="PX",
                        help="fallback synthetic image size when the model config "
                             "does not state one (default: %(default)s)")

    host = parser.add_argument_group("SiMa DevKit specifics")
    host.add_argument("--restart-service", default="simaai-appcomplex.service",
                      metavar="UNIT",
                      help="systemd unit restarted after each run, because the MLA "
                           "stack leaks buffers and the next run can then fail to "
                           "allocate; pass an empty string to skip "
                           "(default: %(default)s)")
    host.add_argument("--restart-settle", type=float, default=10, metavar="S",
                      help="wait after that restart before the next run "
                           "(default: %(default)s)")

    parser.add_argument("--reports-dir", type=Path, default=Path("reports"),
                        metavar="DIR",
                        help="root the report is written under; an absolute "
                             "outdir ignores it (default: %(default)s)")
    return parser.parse_args()


def report_dir(name, reports_dir):
    """Reports collect under one root, unless an absolute path says otherwise.
    An unnamed report gets a timestamp, which also sorts chronologically."""
    name = Path(name or time.strftime("%Y%m%d-%H%M%S"))
    return name if name.is_absolute() else reports_dir / name


def print_config(out, args):
    print("\n".join([
        "config",
        f"  report      {out}",
        f"  models      {', '.join(Path(m).name for m in args.models)}",
        f"  workload    {args.repeat} run(s) x {args.prompts} prompt(s) per model,"
        f" {args.max_tokens} max tokens",
        f"  phases      idle_pre {args.idle_pre:g}s, idle_loaded"
        f" {args.idle_loaded:g}s, drain {args.drain:g}s",
        f"  sampling    every {args.interval:g}s",
        f"  server      {args.host}:{args.port}",
    ]), flush=True)


def main():
    args = parse_args()
    out = report_dir(args.name, args.reports_dir)
    print_config(out, args)
    for i in range(1, args.repeat + 1):
        run_dir = out / f"run{i}" if args.repeat > 1 else out
        print(f"\n===== run {i}/{args.repeat} -> {run_dir}", flush=True)
        one_run(run_dir, args)
        if args.restart_service:
            subprocess.run(["sudo", "systemctl", "restart", args.restart_service])
            time.sleep(args.restart_settle)

    charts.render(out)


if __name__ == "__main__":
    main()
