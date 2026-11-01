"""Run experiment grids on one GPU with several experiments in parallel, skip finished ones, and (on cloud
machines) push finished results to a `results/<branch>` branch after every experiment.

    python scripts/run_grid.py configs/p2/vit.yaml --parallel 4 --branch vit
    python scripts/run_grid.py configs/p1/cnn.yaml --only p1_r50,p1_convnext_t --parallel 2
    python scripts/run_grid.py configs/jobs/p3_da_pipeline.yaml --branch p3-da   # `commands:` file, sequential

Small models (ResNet-18, ViT-S at batch 32) cannot saturate an A100/L4 alone: the input pipeline and
Python overhead are the bottleneck. Running N experiments side by side fills the GPU without changing
any experiment's hyper-parameters (batch size, steps, LR stay as in the config).

Pushing needs GH_TOKEN in the environment; the token is sent as an HTTP header and never printed.
"""
import argparse
import base64
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def experiments_of(config):
    spec = yaml.safe_load(Path(config).read_text())
    if "experiments" in spec:
        return [(config, e["name"]) for e in spec["experiments"]]
    return [(config, spec["name"])]


def done(name):
    return (ROOT / "experiments" / name / "metrics.json").exists()


class Pusher:
    def __init__(self, branch):
        token = os.environ.get("GH_TOKEN", "").strip()
        self.branch = f"results/{branch}" if branch else None
        self.auth = ("AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{token}".encode()).decode()) if token else None
        self.token = token

    def git(self, *args):
        cmd = ["git"] + (["-c", f"http.https://github.com/.extraheader={self.auth}"] if self.auth else []) + list(args)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out = (r.stdout + r.stderr)
        for secret in filter(None, (self.token, self.auth)):
            out = out.replace(secret, "***")
        return r.returncode, out

    def push(self, message):
        if not (self.branch and self.auth):
            return
        self.git("add", *[d for d in ("experiments", "submissions") if (ROOT / d).exists()])
        code, _ = self.git("commit", "-q", "-m", message)
        if code == 0:
            code, out = self.git("push", "-q", "-f", "origin", f"HEAD:{self.branch}")
            print(f"  pushed to {self.branch}" if code == 0 else f"  push failed: {out}", flush=True)
        else:
            print("  nothing new to commit", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--branch", help="push results to results/<branch> (needs GH_TOKEN)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra arguments for scripts/train.py, e.g. --test")
    args = ap.parse_args()

    pusher = Pusher(args.branch)
    if args.branch and not pusher.auth:
        print("WARNING: --branch given but GH_TOKEN is not set: results stay on this machine only", flush=True)
    if pusher.branch and pusher.auth:
        pusher.git("config", "user.email", "runner@cloud")
        pusher.git("config", "user.name", "cloud-runner")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    # A `commands:` file is a sequential pipeline (each command depends on the previous one).
    for config in args.configs:
        spec = yaml.safe_load(Path(config).read_text())
        if "commands" in spec:
            for i, cmd in enumerate(spec["commands"]):
                print(f"[{i + 1}/{len(spec['commands'])}] {cmd}", flush=True)
                with open(logs / f"{Path(config).stem}_{i}.log", "w") as f:
                    rc = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
                print("  ok" if rc == 0 else f"  FAILED (exit {rc}), see logs/{Path(config).stem}_{i}.log", flush=True)
                pusher.push(f"{config}: step {i + 1} ({'ok' if rc == 0 else 'failed'})")
                if rc:
                    sys.exit(rc)
    queue = [(c, n) for c in args.configs if "commands" not in yaml.safe_load(Path(c).read_text())
             for c, n in experiments_of(c)]
    if args.only:
        keep = set(args.only.split(","))
        queue = [q for q in queue if q[1] in keep]
    skipped = [n for _, n in queue if done(n)]
    queue = [q for q in queue if not done(q[1])]
    if skipped:
        print("already done, skipped:", ", ".join(skipped), flush=True)
    workers = max(2, (os.cpu_count() or 4) // max(args.parallel, 1) - 1)
    print(f"{len(queue)} experiments, {args.parallel} in parallel, {workers} data workers each", flush=True)

    running = {}
    while queue or running:
        while queue and len(running) < args.parallel:
            config, name = queue.pop(0)
            log = open(logs / f"{name}.log", "w")
            cmd = [sys.executable, "scripts/train.py", config, "--only", name, "--set", f"num_workers={workers}", *args.extra]
            running[name] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"start {name}", flush=True)
        time.sleep(10)
        for name, (proc, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            tail = [l for l in (logs / f"{name}.log").read_text().splitlines() if "] CV" in l]
            status = tail[-1].split("] ", 1)[1] if tail and proc.returncode == 0 else f"FAILED (exit {proc.returncode}), see logs/{name}.log"
            print(f"done  {name}: {status}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            pusher.push(f"{name}: {status}")


if __name__ == "__main__":
    main()
