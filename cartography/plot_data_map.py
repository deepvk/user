## https://github.com/alisawuffles/wanli/blob/main/cartography/plot_data_map.py

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_data_map(
    metrics_df: pd.DataFrame,
    title: str,
):
    # Set style.
    sns.set(style="whitegrid", font_scale=1.6, context="paper")
    print(f"Plotting figure for {title}")

    # Normalize correctness to a value between 0 and 1.
    metrics_df = metrics_df.assign(corr_frac=lambda d: d.correctness / d.correctness.max())
    metrics_df["correct."] = [f"{x:.1f}" for x in metrics_df["corr_frac"]]

    main_metric = "variability"
    other_metric = "confidence"
    hue_order = ["1.0", "0.8", "0.6", "0.4", "0.2", "0.0"]
    plt.figure(figsize=(8, 6))
    plot = sns.jointplot(
        x=main_metric,
        y=other_metric,
        kind="hist",
        data=metrics_df,
        hue="correct.",
        hue_order=hue_order,
        joint_kws={"fill": True, "hue_order": hue_order},
        marginal_kws={"fill": False, "hue_order": hue_order},
        height=8,
        bins=150,
    )

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda text, xyc, bbc: plot.ax_joint.annotate(
        text,
        xy=xyc,
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        rotation=350,
        bbox=bb(bbc),
    )
    func_annotate("ambiguous", xyc=(0.9, 0.5), bbc="black")
    func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc="r")
    func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc="b")

    plot.ax_joint.set_title(f"{title} Data Map", fontsize=17)

    plot.figure.tight_layout()
    filename = f"figures/{title}_datamap.png"
    plot.figure.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")


def main(td_metrics_path: str, title: str):
    metrics_df = pd.read_json(f"{td_metrics_path}/td_metrics.jsonl", lines=True)
    plot_data_map(metrics_df, title=title)


if __name__ == "__main__":
    main()
