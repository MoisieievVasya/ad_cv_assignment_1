"""Build the mandatory results table and the compute log from experiments/*/metrics.json.

    uv run python scripts/make_report_tables.py

Writes reports/results_table.md (CV mean ± std over the 5 site folds, OOF EMD, Kaggle public/private
LB for experiments that were submitted, from reports/leaderboard.csv) and reports/compute_log.csv
(device, GPU-hours, peak memory per experiment, totals per device).
"""
import json

import pandas as pd

from soilgs import paths

PARTS = {"0": "Part 0 — baselines", "1": "Part 1 — CNN", "2": "Part 2 — ViT", "3": "Part 3 — self-supervised",
         "6": "Part 6 — data efficiency", "C": "Block C — colour normalisation / cross-device", "S": "Sanity checks", "final": "Final models (fold checkpoints + test)"}


def main():
    rows = []
    for m in sorted(paths.EXPERIMENTS.glob("*/metrics.json")):
        d = json.loads(m.read_text())
        if d["name"].startswith(("smoke", "cka_")):
            continue
        c = d.get("compute", {})
        part = str(d.get("part"))
        if d["name"].startswith("rep_"):
            part = "rep"
        rows.append({"part": part, "experiment": d["name"], "cv_mean": d["cv_mean"], "cv_std": d["cv_std"],
                     "oof_emd": d["oof_emd"], "device": c.get("device", ""), "gpu_hours": c.get("gpu_hours", 0.0),
                     "hours": c.get("hours", 0.0), "peak_mem_gb": c.get("peak_mem_gb")})
    df = pd.DataFrame(rows)
    lb_path = paths.ROOT / "reports" / "leaderboard.csv"
    lb = pd.read_csv(lb_path) if lb_path.exists() else pd.DataFrame(columns=["submission", "public_lb", "private_lb"])
    lb["experiment"] = lb.submission.str.replace(".csv", "", regex=False)
    df = df.merge(lb[["experiment", "public_lb", "private_lb"]], on="experiment", how="left")

    (paths.ROOT / "reports").mkdir(exist_ok=True)
    out = ["# Results table", "",
           "CV = mean ± std of the competition EMD over the 5 site-grouped folds (`data/folds.csv`, same for every",
           "experiment); OOF = EMD of the pooled out-of-fold sample curves. Lower is better. Public / private LB only",
           "for experiments that were submitted (`reports/leaderboard.csv`). Seed noise ≈ 0.2–0.7 (replicates `rep_*`).",
           "Mean-curve baseline 93.4, uniform curve 102.", ""]
    order = list(PARTS) + ["rep"]
    titles = {**PARTS, "rep": "Seed replicates (noise floor)"}
    for p in order:
        t = df[df.part == p].sort_values("cv_mean")
        if t.empty:
            continue
        out += [f"## {titles[p]}", "", "| experiment | CV | OOF | public LB | private LB | GPU-h | device |", "|---|---|---|---|---|---|---|"]
        for r in t.itertuples():
            lbp = "" if pd.isna(r.public_lb) else f"{r.public_lb:.2f}"
            lbq = "" if pd.isna(r.private_lb) else f"{r.private_lb:.2f}"
            out.append(f"| {r.experiment} | {r.cv_mean:.2f} ± {r.cv_std:.2f} | {r.oof_emd:.2f} | {lbp} | {lbq} | "
                       f"{r.gpu_hours:.3f} | {r.device} |")
        out.append("")
    subs = lb[~lb.experiment.isin(df.experiment)]
    if len(subs):
        out += ["## Other submissions", "", "| submission | public LB | private LB |", "|---|---|---|"]
        out += [f"| {r.submission} | {r.public_lb:.2f} | {'' if pd.isna(r.private_lb) else r.private_lb} |" for r in subs.itertuples()]
    (paths.ROOT / "reports" / "results_table.md").write_text("\n".join(out) + "\n")

    log = df[["part", "experiment", "device", "gpu_hours", "hours", "peak_mem_gb"]].sort_values(["part", "experiment"])
    log.to_csv(paths.ROOT / "reports" / "compute_log.csv", index=False)
    totals = log.groupby("device")[["gpu_hours", "hours"]].sum().round(2)
    totals.to_csv(paths.ROOT / "reports" / "compute_totals.csv")
    print(totals.to_string())
    print(f"{len(df)} experiments -> reports/results_table.md, reports/compute_log.csv")


if __name__ == "__main__":
    main()
