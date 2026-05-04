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
from dataclasses import dataclass, field

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
    # raw tag (e.g. "ndcg@20_cold") -> list of (step, value)
    points: dict[str, list[tuple[int, float]]]
    # metric -> {segment -> max value within truncation window}, filled later
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    max_step: int = 0
    cutoff_step: int = 0


def parse_event_file(path: str) -> dict[str, list[tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        if not tag.startswith("eval/"):
            continue
        pts = ea.Scalars(tag)
        if not pts:
            continue
        out[tag[len("eval/"):]] = [(int(p.step), float(p.value)) for p in pts]
    return out


def parse_run(dataset: str, run_dir: str) -> RunResult:
    event_files = sorted(
        glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    )
    merged: dict[str, list[tuple[int, float]]] = {}
    for ef in event_files:
        for k, pts in parse_event_file(ef).items():
            merged.setdefault(k, []).extend(pts)

    max_step = max((s for pts in merged.values() for s, _ in pts), default=0)
    return RunResult(
        dataset=dataset,
        variant=variant_from_dirname(os.path.basename(run_dir)),
        run_dir=run_dir,
        points=merged,
        max_step=max_step,
    )


def fill_metrics(runs: list[RunResult]) -> None:
    """Equalize per dataset to the min training length, then take max per metric."""
    by_dataset: dict[str, list[RunResult]] = {}
    for r in runs:
        by_dataset.setdefault(r.dataset, []).append(r)

    for dataset, rs in by_dataset.items():
        cutoff = min(r.max_step for r in rs)
        print(
            f"[truncate] {dataset}: per-run max_step="
            + ", ".join(f"{r.variant}={r.max_step}" for r in rs)
            + f" -> cutoff={cutoff}"
        )
        for r in rs:
            r.cutoff_step = cutoff
            metrics: dict[str, dict[str, float]] = {m: {} for m in METRICS}
            for raw_tag, pts in r.points.items():
                m = re.match(r"^(ndcg@\d+|recall@\d+)(?:_(cold|warm|hot))?$", raw_tag)
                if not m:
                    continue
                metric, seg = m.group(1), m.group(2) or "overall"
                if metric not in metrics:
                    continue
                vals = [v for s, v in pts if s <= cutoff]
                if vals:
                    metrics[metric][seg] = max(vals)
            r.metrics = metrics


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
    fill_metrics(runs)
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


def build_baseline_uplift_frame(runs: list[RunResult], segment: str) -> pd.DataFrame:
    """Per-segment uplift over baseline for every non-baseline variant."""
    seg_df = build_summary_frame(runs, segment=segment)
    rows = []
    for dataset, sub in seg_df.groupby("dataset", observed=True):
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


def build_pair_uplift_frame(
    runs: list[RunResult], segment: str, base: str, target: str
) -> pd.DataFrame:
    """Per-dataset uplift of `target` variant over `base` variant for `segment`."""
    seg_df = build_summary_frame(runs, segment=segment)
    rows = []
    for dataset, sub in seg_df.groupby("dataset", observed=True):
        base_row = sub[sub["variant"] == base]
        tgt_row = sub[sub["variant"] == target]
        if base_row.empty or tgt_row.empty:
            continue
        b, t = base_row.iloc[0], tgt_row.iloc[0]
        for m in METRICS:
            bv, tv = b[m], t[m]
            if pd.isna(bv) or pd.isna(tv):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "metric": m,
                    "base": bv,
                    "value": tv,
                    "abs_uplift": tv - bv,
                    "rel_uplift_%": (tv - bv) / bv * 100.0 if bv else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fmt_summary_block(df: pd.DataFrame, segment_label: str) -> str:
    out = [f"### {segment_label}\n"]
    for dataset, sub in df.groupby("dataset", observed=True):
        sub = sub.reset_index(drop=True)
        out.append(f"**{dataset}**")
        view = sub[["variant", *METRICS]].copy()
        for m in METRICS:
            vals = sub[m].to_numpy(dtype=float)
            ranked = sorted(
                [(i, v) for i, v in enumerate(vals) if not np.isnan(v)],
                key=lambda x: -x[1],
            )
            best_idx = ranked[0][0] if len(ranked) >= 1 else None
            second_idx = ranked[1][0] if len(ranked) >= 2 else None
            cells = []
            for i, v in enumerate(vals):
                if pd.isna(v):
                    cells.append("-")
                elif i == best_idx:
                    cells.append(f"**{v:.4g}**")
                elif i == second_idx:
                    cells.append(f"<u>{v:.4g}</u>")
                else:
                    cells.append(f"{v:.4g}")
            view[m] = cells
        out.append(tabulate(view, headers="keys", tablefmt="github", showindex=False))
        out.append("")
    return "\n".join(out)


def fmt_baseline_uplift(df: pd.DataFrame, segment_label: str) -> str:
    title = f"### Improvements over baseline — {segment_label}\n"
    if df.empty:
        return title + "_no data_\n"
    out = [title]
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


def fmt_pair_uplift(df: pd.DataFrame, base: str, target: str, segment_label: str) -> str:
    title = f"### {target} vs {base} — {segment_label}\n"
    if df.empty:
        return title + "_no data_\n"
    out = [title]
    for dataset, sub in df.groupby("dataset", observed=True):
        out.append(f"**{dataset}**")
        view = sub.copy()
        view["base"] = view["base"].map(lambda v: f"{v:.4g}")
        view["value"] = view["value"].map(lambda v: f"{v:.4g}")
        view["abs_uplift"] = view["abs_uplift"].map(lambda v: f"{v:+.4g}")
        view["rel_uplift_%"] = view["rel_uplift_%"].map(lambda v: f"{v:+.2f}%")
        view = view.rename(columns={"base": base, "value": target})
        out.append(
            tabulate(
                view[["metric", base, target, "abs_uplift", "rel_uplift_%"]],
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )
        out.append("")
    return "\n".join(out)


def plot_pair_uplift(
    df: pd.DataFrame, base: str, target: str, segment_label: str, out_path: str
) -> None:
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique(), key=lambda d: list(DATASETS).index(d))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    x = np.arange(len(METRICS))
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].set_index("metric").reindex(METRICS)
        vals = sub["rel_uplift_%"].to_numpy(dtype=float)
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        ax.bar(x, vals, 0.6, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=30, ha="right")
        ax.set_ylabel(f"Relative uplift of {target} over {base}, %")
        ax.set_title(f"{ds} — {segment_label}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(f"{target} vs {base} — {segment_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


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
    fig.suptitle(
        f"Eval metrics ({segment_label}) — max within common min_step window",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_uplift(df: pd.DataFrame, segment_label: str, out_path: str) -> None:
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
        ax.set_title(f"{ds} — {segment_label}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle(f"Relative uplift vs baseline — {segment_label}", fontsize=12)
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
        print(
            f"  - {r.dataset:9s} | {r.variant:12s} | "
            f"max_step={r.max_step:>6d} cutoff={r.cutoff_step:>6d} | "
            f"{os.path.basename(r.run_dir)}"
        )
    print()

    seg_labels = [
        ("overall", "Overall"),
        ("cold", "Cold items"),
        ("warm", "Warm items"),
        ("hot", "Hot items"),
    ]

    summaries = {seg: build_summary_frame(runs, seg) for seg in SEGMENTS}
    baseline_uplift = {seg: build_baseline_uplift_frame(runs, seg) for seg in SEGMENTS}
    logq_vs_tuned = {
        seg: build_pair_uplift_frame(runs, seg, base="tuned, 0.07", target="logQ")
        for seg in SEGMENTS
    }

    for seg, df in summaries.items():
        df.to_csv(os.path.join(args.out_dir, f"summary_{seg}.csv"), index=False)
    for seg, df in baseline_uplift.items():
        df.to_csv(os.path.join(args.out_dir, f"uplift_vs_baseline_{seg}.csv"), index=False)
    for seg, df in logq_vs_tuned.items():
        df.to_csv(os.path.join(args.out_dir, f"logq_vs_tuned_{seg}.csv"), index=False)

    for seg, label in seg_labels:
        print(fmt_summary_block(summaries[seg], label))
    for seg, label in seg_labels:
        print(fmt_baseline_uplift(baseline_uplift[seg], label))
    for seg, label in seg_labels:
        print(fmt_pair_uplift(logq_vs_tuned[seg], "tuned, 0.07", "logQ", label))

    for seg in SEGMENTS:
        plot_summary(summaries[seg], seg, os.path.join(args.out_dir, f"metrics_{seg}.png"))
        plot_baseline_uplift(
            baseline_uplift[seg], seg,
            os.path.join(args.out_dir, f"uplift_vs_baseline_{seg}.png"),
        )
        plot_pair_uplift(
            logq_vs_tuned[seg], "tuned, 0.07", "logQ", seg,
            os.path.join(args.out_dir, f"logq_vs_tuned_{seg}.png"),
        )

    md_path = os.path.join(args.out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("# Diploma TensorBoard summary\n\n")
        for seg, label in seg_labels:
            f.write(fmt_summary_block(summaries[seg], label) + "\n")
            f.write(f"![{label}](metrics_{seg}.png)\n\n")
        f.write("## Uplift vs baseline — per segment\n\n")
        for seg, label in seg_labels:
            f.write(fmt_baseline_uplift(baseline_uplift[seg], label) + "\n")
            f.write(f"![Uplift vs baseline — {label}](uplift_vs_baseline_{seg}.png)\n\n")
        f.write("## logQ vs tuned, 0.07 — uplift per segment\n\n")
        for seg, label in seg_labels:
            f.write(fmt_pair_uplift(logq_vs_tuned[seg], "tuned, 0.07", "logQ", label) + "\n")
            f.write(f"![logQ vs tuned — {label}](logq_vs_tuned_{seg}.png)\n\n")
    print(f"\nReport written to {md_path}")
    print(f"Plots and CSVs in {args.out_dir}/")


if __name__ == "__main__":
    main()
