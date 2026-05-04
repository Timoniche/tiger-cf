"""
Diploma analytics: aggregate TensorBoard runs from MastersDiploma/* into
summary tables and plots.

Usage (from repo root):
    uv run --with tensorboard,pandas,matplotlib,tabulate \
        python3 ai/vk_exps/diploma_analytics.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from tensorboard.backend.event_processing import event_accumulator


DATASETS = {
    "Beauty": "MastersDiploma/beauty_tensorboard_logs",
    "VK-small": "MastersDiploma/vk_small_tensorboard_logs",
    "Yambda": "MastersDiploma/yambda_tensorboard_logs",
}

VARIANT_ORDER = ["baseline", "tuned, 0.07", "logQ"]

METRICS = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20"]
SEGMENTS = ["overall", "cold", "warm", "hot"]


def variant_from_dirname(name: str) -> str:
    n = name.lower()
    if "logq" in n:
        return "logQ"
    if "tuned_0_07" in n or "tuned_0.07" in n:
        return "tuned, 0.07"
    if "baseline" in n:
        return "baseline"
    return name


@dataclass
class RunResult:
    dataset: str
    variant: str
    run_dir: str
    # metric -> {segment -> max value}
    metrics: dict[str, dict[str, float]]


def parse_event_file(path: str) -> dict[str, float]:
    ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, float] = {}
    for tag in ea.Tags().get("scalars", []):
        if not tag.startswith("eval/"):
            continue
        pts = ea.Scalars(tag)
        if not pts:
            continue
        out[tag[len("eval/"):]] = max(p.value for p in pts)
    return out


def parse_run(dataset: str, run_dir: str) -> RunResult:
    event_files = sorted(
        glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    )
    merged: dict[str, float] = {}
    for ef in event_files:
        for k, v in parse_event_file(ef).items():
            if k not in merged or v > merged[k]:
                merged[k] = v

    metrics: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    for raw_tag, value in merged.items():
        m = re.match(r"^(ndcg@\d+|recall@\d+)(?:_(cold|warm|hot))?$", raw_tag)
        if not m:
            continue
        metric, seg = m.group(1), m.group(2) or "overall"
        if metric in metrics:
            metrics[metric][seg] = value

    return RunResult(
        dataset=dataset,
        variant=variant_from_dirname(os.path.basename(run_dir)),
        run_dir=run_dir,
        metrics=metrics,
    )


def collect_runs() -> list[RunResult]:
    runs: list[RunResult] = []
    for dataset, root in DATASETS.items():
        if not os.path.isdir(root):
            print(f"[warn] dataset dir missing: {root}")
            continue
        for entry in sorted(os.listdir(root)):
            run_dir = os.path.join(root, entry)
            if not os.path.isdir(run_dir):
                continue
            runs.append(parse_run(dataset, run_dir))
    return runs


def build_summary_frame(runs: list[RunResult], segment: str = "overall") -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {"dataset": r.dataset, "variant": r.variant}
        for m in METRICS:
            row[m] = r.metrics.get(m, {}).get(segment, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=list(DATASETS), ordered=True)
    return df.sort_values(["dataset", "variant"]).reset_index(drop=True)


def build_cold_uplift_frame(runs: list[RunResult]) -> pd.DataFrame:
    """Cold-segment uplift over baseline, in absolute and % terms."""
    cold_df = build_summary_frame(runs, segment="cold")
    rows = []
    for dataset, sub in cold_df.groupby("dataset", observed=True):
        base = sub[sub["variant"] == "baseline"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, r in sub.iterrows():
            if r["variant"] == "baseline":
                continue
            for m in METRICS:
                bv = base_row[m]
                if pd.isna(bv) or pd.isna(r[m]):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": r["variant"],
                        "metric": m,
                        "baseline": bv,
                        "value": r[m],
                        "abs_uplift": r[m] - bv,
                        "rel_uplift_%": (r[m] - bv) / bv * 100.0 if bv else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def fmt_summary_block(df: pd.DataFrame, segment_label: str) -> str:
    out = [f"### {segment_label}\n"]
    for dataset, sub in df.groupby("dataset", observed=True):
        out.append(f"**{dataset}**")
        view = sub[["variant", *METRICS]].copy()
        for m in METRICS:
            view[m] = view[m].map(lambda v: f"{v:.4g}" if pd.notna(v) else "-")
        out.append(tabulate(view, headers="keys", tablefmt="github", showindex=False))
        out.append("")
    return "\n".join(out)


def fmt_cold_uplift(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no cold-segment data_"
    out = ["### Cold-item improvements over baseline\n"]
    for dataset, sub in df.groupby("dataset", observed=True):
        out.append(f"**{dataset}**")
        view = sub.copy()
        view["baseline"] = view["baseline"].map(lambda v: f"{v:.4g}")
        view["value"] = view["value"].map(lambda v: f"{v:.4g}")
        view["abs_uplift"] = view["abs_uplift"].map(lambda v: f"{v:+.4g}")
        view["rel_uplift_%"] = view["rel_uplift_%"].map(lambda v: f"{v:+.2f}%")
        out.append(
            tabulate(
                view[["variant", "metric", "baseline", "value", "abs_uplift", "rel_uplift_%"]],
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )
        out.append("")
    return "\n".join(out)


def plot_summary(df: pd.DataFrame, segment_label: str, out_path: str) -> None:
    datasets = list(df["dataset"].cat.categories)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.25
    x = np.arange(len(METRICS))
    palette = {"baseline": "#9e9e9e", "tuned, 0.07": "#1f77b4", "logQ": "#d62728"}
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        variants_present = [v for v in VARIANT_ORDER if v in sub["variant"].astype(str).tolist()]
        for i, var in enumerate(variants_present):
            row = sub[sub["variant"] == var]
            if row.empty:
                continue
            vals = [row.iloc[0][m] for m in METRICS]
            ax.bar(x + (i - 1) * width, vals, width, label=var, color=palette.get(var, None))
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=30, ha="right")
        ax.set_title(f"{ds} — {segment_label}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle(f"Eval metrics ({segment_label}) — max over training", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cold_uplift(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique(), key=lambda d: list(DATASETS).index(d))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.4
    x = np.arange(len(METRICS))
    palette = {"tuned, 0.07": "#1f77b4", "logQ": "#d62728"}
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        variants = [v for v in VARIANT_ORDER if v in sub["variant"].unique() and v != "baseline"]
        for i, var in enumerate(variants):
            sv = sub[sub["variant"] == var].set_index("metric").reindex(METRICS)
            vals = sv["rel_uplift_%"].to_numpy(dtype=float)
            ax.bar(x + (i - 0.5) * width, vals, width, label=var, color=palette.get(var, None))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=30, ha="right")
        ax.set_ylabel("Relative uplift over baseline, %")
        ax.set_title(f"{ds} — cold items")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Cold-item relative uplift vs baseline", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="ai/vk_exps/diploma_analytics_out",
        help="where to put CSVs and PNGs",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = collect_runs()
    if not runs:
        raise SystemExit("no runs found under MastersDiploma/*_tensorboard_logs")

    print(f"Found {len(runs)} runs:")
    for r in runs:
        print(f"  - {r.dataset:9s} | {r.variant:12s} | {os.path.basename(r.run_dir)}")
    print()

    overall = build_summary_frame(runs, "overall")
    cold = build_summary_frame(runs, "cold")
    warm = build_summary_frame(runs, "warm")
    hot = build_summary_frame(runs, "hot")
    cold_uplift = build_cold_uplift_frame(runs)

    overall.to_csv(os.path.join(args.out_dir, "summary_overall.csv"), index=False)
    cold.to_csv(os.path.join(args.out_dir, "summary_cold.csv"), index=False)
    warm.to_csv(os.path.join(args.out_dir, "summary_warm.csv"), index=False)
    hot.to_csv(os.path.join(args.out_dir, "summary_hot.csv"), index=False)
    cold_uplift.to_csv(os.path.join(args.out_dir, "cold_uplift_vs_baseline.csv"), index=False)

    print(fmt_summary_block(overall, "Overall"))
    print(fmt_summary_block(cold, "Cold items"))
    print(fmt_summary_block(warm, "Warm items"))
    print(fmt_summary_block(hot, "Hot items"))
    print(fmt_cold_uplift(cold_uplift))

    plot_summary(overall, "overall", os.path.join(args.out_dir, "metrics_overall.png"))
    plot_summary(cold, "cold", os.path.join(args.out_dir, "metrics_cold.png"))
    plot_summary(warm, "warm", os.path.join(args.out_dir, "metrics_warm.png"))
    plot_summary(hot, "hot", os.path.join(args.out_dir, "metrics_hot.png"))
    plot_cold_uplift(cold_uplift, os.path.join(args.out_dir, "cold_uplift.png"))

    md_path = os.path.join(args.out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("# Diploma TensorBoard summary\n\n")
        f.write(fmt_summary_block(overall, "Overall") + "\n")
        f.write(fmt_summary_block(cold, "Cold items") + "\n")
        f.write(fmt_summary_block(warm, "Warm items") + "\n")
        f.write(fmt_summary_block(hot, "Hot items") + "\n")
        f.write(fmt_cold_uplift(cold_uplift) + "\n")
    print(f"\nReport written to {md_path}")
    print(f"Plots and CSVs in {args.out_dir}/")


if __name__ == "__main__":
    main()
