"""Additional comparison plots for lambda sweep results.

Creates:
1) Grouped bar plot across lambdas for OT methods (+ KL/BC baseline lines)
2) Seed-level strip plot at each OT method's best lambda
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "bc": "BC",
    "forward_kl": "Forward KL",
    "reverse_kl": "Reverse KL",
    "wasserstein": "Wasserstein",
    "partial_ot": "Partial OT",
    "unbalanced_ot": "Unbalanced OT",
}

OT_METHODS = ["wasserstein", "partial_ot", "unbalanced_ot"]
BASELINES = ["bc", "forward_kl", "reverse_kl"]


def load_summary(path: str):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                {
                    "method": r["method"],
                    "lambda": float(r["lambda"]),
                    "mean": float(r["mean_reward_mean"]),
                    "std": float(r["mean_reward_std"]),
                    "stderr": float(r["mean_reward_stderr"]),
                }
            )
    return rows


def load_raw(path: str):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                {
                    "seed": int(r["seed"]),
                    "method": r["method"],
                    "lambda": float(r["lambda"]),
                    "reward": float(r["mean_reward"]),
                }
            )
    return rows


def get_best_lambda(summary_rows, method: str) -> float:
    arr = [r for r in summary_rows if r["method"] == method]
    best = max(arr, key=lambda x: x["mean"])
    return best["lambda"]


def plot_grouped_bar(summary_rows, out_path: str, title: str):
    lambdas = sorted({r["lambda"] for r in summary_rows if r["method"] in OT_METHODS})
    x = np.arange(len(lambdas))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {
        "wasserstein": "#1f77b4",
        "partial_ot": "#2ca02c",
        "unbalanced_ot": "#ff7f0e",
    }

    for i, method in enumerate(OT_METHODS):
        rows = sorted([r for r in summary_rows if r["method"] == method], key=lambda z: z["lambda"])
        means = [r["mean"] for r in rows]
        errs = [r["stderr"] for r in rows]
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            color=colors[method],
            alpha=0.9,
            label=METHOD_LABELS[method],
        )

    base_styles = {
        "bc": ("--", "#666666"),
        "forward_kl": ("-.", "#111111"),
        "reverse_kl": (":", "#444444"),
    }
    for base in BASELINES:
        row = next((r for r in summary_rows if r["method"] == base), None)
        if row is None:
            continue
        ls, c = base_styles[base]
        ax.axhline(row["mean"], linestyle=ls, color=c, linewidth=2.0, label=f"{METHOD_LABELS[base]} baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in lambdas])
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Mean reward (higher is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_best_seed_strip(summary_rows, raw_rows, out_path: str, title: str):
    best_by_method = {m: get_best_lambda(summary_rows, m) for m in OT_METHODS}

    categories = BASELINES + OT_METHODS
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Seed points
    for i, method in enumerate(categories):
        if method in OT_METHODS:
            target_lambda = best_by_method[method]
            samples = [r["reward"] for r in raw_rows if r["method"] == method and abs(r["lambda"] - target_lambda) < 1e-12]
        else:
            samples = [r["reward"] for r in raw_rows if r["method"] == method]

        # Jitter for visibility
        jitter = np.linspace(-0.08, 0.08, max(len(samples), 1))
        for j, y in enumerate(samples):
            ax.scatter(i + jitter[j], y, c="tab:blue", s=28, alpha=0.8)

    # Mean ± std markers from summary
    for i, method in enumerate(categories):
        if method in OT_METHODS:
            lam = best_by_method[method]
            row = next(r for r in summary_rows if r["method"] == method and abs(r["lambda"] - lam) < 1e-12)
        else:
            row = next(r for r in summary_rows if r["method"] == method)
        ax.errorbar(i, row["mean"], yerr=row["std"], fmt="o", color="black", capsize=4, markersize=6)

    labels = []
    for method in categories:
        if method in OT_METHODS:
            labels.append(f"{METHOD_LABELS[method]}\n(best λ={best_by_method[method]:g})")
        else:
            labels.append(f"{METHOD_LABELS[method]}\n(baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean reward (seed-wise)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create extra lambda comparison plots.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--title-tag", default="")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    summary_rows = load_summary(args.summary_csv)
    raw_rows = load_raw(args.raw_csv)

    title = "Lambda Sweep Comparison"
    if args.title_tag:
        title = f"{title} ({args.title_tag})"

    grouped_out = f"results/{args.output_prefix}_grouped_bar.png"
    strip_out = f"results/{args.output_prefix}_best_seed_strip.png"

    plot_grouped_bar(summary_rows, grouped_out, title)
    plot_best_seed_strip(summary_rows, raw_rows, strip_out, title + " - Best Lambda Seed Distribution")


if __name__ == "__main__":
    main()
