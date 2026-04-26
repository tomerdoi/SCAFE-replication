"""Generate ensemble comparison bar chart for the paper (per-fold means)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"

datasets = {
    "Merged": {
        "Best ML\n(Vote)":         {"ROC-AUC": 0.932, "PR-AUC": 0.589, "F1": 0.611},
        "UniXcoder":               {"ROC-AUC": 0.953, "PR-AUC": 0.645, "F1": 0.679},
        "Weighted\navg":           {"ROC-AUC": 0.959, "PR-AUC": 0.662, "F1": 0.683},
        "Stack:\nSCAFE+UX":       {"ROC-AUC": 0.952, "PR-AUC": 0.550, "F1": 0.688},
    },
    "DAppSCAN": {
        "Best ML\n(RF)":           {"ROC-AUC": 0.907, "PR-AUC": 0.371, "F1": 0.425},
        "UniXcoder":               {"ROC-AUC": 0.948, "PR-AUC": 0.691, "F1": 0.653},
        "Weighted\navg":           {"ROC-AUC": 0.967, "PR-AUC": 0.730, "F1": 0.675},
        "Stack:\nSCAFE+UX":       {"ROC-AUC": 0.967, "PR-AUC": 0.706, "F1": 0.672},
    },
}

metrics = ["ROC-AUC", "PR-AUC", "F1"]
colors = {
    0: "#5B9BD5",
    1: "#ED7D31",
    2: "#70AD47",
    3: "#FFC000",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax_idx, (ds_name, configs) in enumerate(datasets.items()):
    ax = axes[ax_idx]
    config_names = list(configs.keys())
    n_configs = len(config_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.linspace(-(n_configs-1)*width/2, (n_configs-1)*width/2, n_configs)

    for i, (cfg_name, vals) in enumerate(configs.items()):
        bar_vals = [vals[m] for m in metrics]
        bars = ax.bar(x + offsets[i], bar_vals, width, label=cfg_name,
                      color=colors[i], edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(ds_name, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           framealpha=0.9, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT / "fig_ensemble_comparison.pdf", bbox_inches="tight", dpi=300)
print(f"Saved: {OUT / 'fig_ensemble_comparison.pdf'}")
