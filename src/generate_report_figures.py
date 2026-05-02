"""
generate_report_figures.py
--------------------------
Generates 6 publication-quality figures for the Tennis Serve Biomechanics Report.
Run with: C:/Users/gonza/anaconda3/python.exe src/generate_report_figures.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from math import pi

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
})

PLAYER_COLORS = {
    "Djokovic": "#1565C0",
    "Federer":  "#2E7D32",
    "Amateur":  "#E65100",
}
PLAYER_KEYS   = ["djokovic", "federer", "amateur"]
PLAYER_LABELS = ["Djokovic", "Federer", "Amateur"]

OUT_DIR = "outputs/report_figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_serves():
    dfs = {}
    for player in PLAYER_KEYS:
        path  = f"outputs/{player}_serves"
        files = sorted(
            [f for f in os.listdir(path) if f.endswith(".csv")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        frames = []
        for f in files:
            df = pd.read_csv(os.path.join(path, f))
            df["serve_id"] = f.replace(".csv", "")
            frames.append(df)
        dfs[player] = pd.concat(frames, ignore_index=True)
    return dfs


def impact_row(df):
    """Row with highest score per serve (the impact frame)."""
    return df.loc[df.groupby("serve_id")["score"].idxmax()].reset_index(drop=True)


# ── Figure 1: Radar / Spider chart ───────────────────────────────────────────

def fig_radar(dfs, summary_df):
    metrics     = ["Elbow Angle", "Knee Angle", "Shoulder Angle"]
    metric_keys = ["elbow_angle", "knee_angle", "shoulder_angle"]

    # Use mean of per-serve maximums
    player_means = {}
    for pk, pl in zip(PLAYER_KEYS, PLAYER_LABELS):
        df_imp = impact_row(dfs[pk])
        player_means[pl] = [df_imp[mk].mean() for mk in metric_keys]

    # Normalise to 0-1 across all players per metric
    all_vals = np.array([player_means[pl] for pl in PLAYER_LABELS])
    mins     = all_vals.min(axis=0)
    maxs     = all_vals.max(axis=0)
    safe_range = np.where(maxs - mins == 0, 1, maxs - mins)

    norm = {pl: (np.array(player_means[pl]) - mins) / safe_range
            for pl in PLAYER_LABELS}

    N      = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11, fontweight="bold")
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8, color="grey")
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    for pl in PLAYER_LABELS:
        values = norm[pl].tolist() + [norm[pl][0]]
        color  = PLAYER_COLORS[pl]
        ax.plot(angles, values, color=color, linewidth=2.5, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.15)


    legend_handles = [
        mpatches.Patch(facecolor=PLAYER_COLORS[pl], label=pl, alpha=0.7)
        for pl in PLAYER_LABELS
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), frameon=True)

    # Subtitle with actual means
    subtitle_lines = []
    for pl in PLAYER_LABELS:
        e, k, s = [f"{v:.0f}°" for v in player_means[pl]]
        subtitle_lines.append(f"{pl}: Elbow {e} | Knee {k} | Shoulder {s}")
    fig.text(0.5, 0.02, "\n".join(subtitle_lines),
             ha="center", fontsize=8, color="grey", style="italic")

    ax.set_title("Multi-Metric Biomechanical Profile\n(Normalised Peak Angles at Impact)",
                 fontsize=13, fontweight="bold", pad=20)

    path = os.path.join(OUT_DIR, "fig1_radar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Figure 2: Violin + Box + Strip plots ─────────────────────────────────────

def fig_violin_box(dfs):
    metric_keys   = ["elbow_angle", "knee_angle", "shoulder_angle"]
    metric_labels = ["Elbow Angle (°)", "Knee Angle (°)", "Shoulder Angle (°)"]
    metric_ranges = [(130, 185), (120, 190), (90, 185)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    fig.suptitle("Distribution of Peak Angles at Serve Impact\n(All Serves per Player)",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, mk, ml, (ylo, yhi) in zip(axes, metric_keys, metric_labels, metric_ranges):
        plot_data = []
        for pk, pl in zip(PLAYER_KEYS, PLAYER_LABELS):
            df_imp = impact_row(dfs[pk])
            vals   = df_imp[mk].values
            for v in vals:
                plot_data.append({"Player": pl, "Value": v})
        pdf = pd.DataFrame(plot_data)

        # Violin
        vp = sns.violinplot(
            data=pdf, x="Player", y="Value", ax=ax,
            palette=PLAYER_COLORS, inner=None, cut=0, linewidth=1.2, alpha=0.4,
            order=PLAYER_LABELS,
        )
        for patch in ax.collections:
            patch.set_alpha(0.35)

        # Box on top
        sns.boxplot(
            data=pdf, x="Player", y="Value", ax=ax,
            palette=PLAYER_COLORS, width=0.18, fliersize=0,
            linewidth=1.5, order=PLAYER_LABELS,
            boxprops={"zorder": 3}, medianprops={"color": "white", "linewidth": 2.5, "zorder": 4},
        )

        # Strip (individual points)
        sns.stripplot(
            data=pdf, x="Player", y="Value", ax=ax,
            palette=PLAYER_COLORS, size=5, alpha=0.8, jitter=True,
            order=PLAYER_LABELS, zorder=5,
        )

        # Annotate mean ± std
        for i, pl in enumerate(PLAYER_LABELS):
            vals = pdf.loc[pdf["Player"] == pl, "Value"]
            ax.annotate(
                f"μ={vals.mean():.1f}°\nσ={vals.std():.1f}°",
                xy=(i, yhi - (yhi - ylo) * 0.08),
                ha="center", va="top", fontsize=8.5,
                color=PLAYER_COLORS[pl], fontweight="bold",
            )

        ax.set_title(ml, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ml)
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_violin_box.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Figure 3: Average Serve Motion Profile ────────────────────────────────────

def fig_motion_profile(dfs):
    metric_keys   = ["elbow_angle", "knee_angle", "wrist_height", "shoulder_angle"]
    metric_labels = ["Elbow Angle (°)", "Knee Angle (°)", "Wrist Height (norm.)", "Shoulder Angle (°)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Average Serve Motion Profile\n(Mean ± 1 SD across all detected serves, aligned to impact at frame 25)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        for pk, pl in zip(PLAYER_KEYS, PLAYER_LABELS):
            df = dfs[pk].copy()
            # Pivot: rows = relative_frame, cols = serve_id
            pivot = df.pivot_table(index="relative_frame", columns="serve_id", values=mk)
            mean  = pivot.mean(axis=1)
            std   = pivot.std(axis=1)

            color = PLAYER_COLORS[pl]
            n     = pivot.shape[1]
            ax.plot(mean.index.to_numpy(), mean.values, color=color, linewidth=2.2,
                    label=f"{pl} (n={n})")
            ax.fill_between(mean.index.to_numpy(),
                            mean.values - std.values,
                            mean.values + std.values,
                            color=color, alpha=0.12)

        # Mark impact reference line (relative_frame == 25)
        ax.axvline(x=25, color="black", linestyle="--", linewidth=1,
                   alpha=0.6, label="Impact ref.")
        ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.set_xlabel("Relative Frame")
        ax.set_ylabel(ml)
        ax.legend(loc="best", fontsize=9)
        ax.grid(linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_motion_profile.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Figure 4: Elbow vs Knee Scatter at Impact ────────────────────────────────

def fig_scatter_elbow_knee(dfs):
    fig, ax = plt.subplots(figsize=(8, 6))

    for pk, pl in zip(PLAYER_KEYS, PLAYER_LABELS):
        df_imp = impact_row(dfs[pk])
        color  = PLAYER_COLORS[pl]
        ax.scatter(
            df_imp["elbow_angle"], df_imp["knee_angle"],
            c=color, s=70, alpha=0.75, edgecolors="white",
            linewidths=0.8, label=pl, zorder=3,
        )
        # Confidence ellipse (mean ± 1 SD rectangle approximation via ellipse)
        xm, ym = df_imp["elbow_angle"].mean(), df_imp["knee_angle"].mean()
        xs, ys = df_imp["elbow_angle"].std(),  df_imp["knee_angle"].std()
        ellipse = mpatches.Ellipse(
            (xm, ym), 2 * xs, 2 * ys,
            angle=0, linewidth=1.5,
            edgecolor=color, facecolor=color, alpha=0.12, zorder=2,
        )
        ax.add_patch(ellipse)
        ax.annotate(
            f"  {pl}\n  μ=({xm:.0f}°, {ym:.0f}°)",
            xy=(xm, ym), fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xlabel("Elbow Angle at Impact (°)", fontsize=11)
    ax.set_ylabel("Knee Angle at Impact (°)", fontsize=11)
    ax.set_title("Elbow vs. Knee Angle at Serve Impact\n(Ellipses show ±1 SD per player)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(linestyle="--", alpha=0.4)

    path = os.path.join(OUT_DIR, "fig4_scatter.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Figure 5: Wrist Height Trajectory ────────────────────────────────────────

def fig_wrist_trajectory(dfs):
    fig, ax = plt.subplots(figsize=(10, 5))

    for pk, pl in zip(PLAYER_KEYS, PLAYER_LABELS):
        df    = dfs[pk].copy()
        pivot = df.pivot_table(index="relative_frame", columns="serve_id",
                               values="wrist_height")
        mean  = pivot.mean(axis=1)
        std   = pivot.std(axis=1)
        color = PLAYER_COLORS[pl]
        n     = pivot.shape[1]

        ax.plot(mean.index.to_numpy(), mean.values, color=color, linewidth=2.5,
                label=f"{pl} (n={n})")
        ax.fill_between(mean.index.to_numpy(),
                        mean.values - std.values,
                        mean.values + std.values,
                        color=color, alpha=0.15)

    ax.axvline(x=25, color="black", linestyle="--", linewidth=1.2,
               alpha=0.65, label="Impact reference")
    ax.invert_yaxis()
    ax.set_xlabel("Relative Frame", fontsize=11)
    ax.set_ylabel("Wrist Height (normalised, lower = higher in image)", fontsize=10)
    ax.set_title("Average Wrist Height Trajectory During Serve\n(Mean ± 1 SD — lower Y value = wrist higher in frame)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    # Annotate regions
    ax.axvspan(0, 10, alpha=0.04, color="blue")
    ax.axvspan(10, 25, alpha=0.04, color="orange")
    ax.axvspan(25, 36, alpha=0.04, color="red")
    ax.text(5,  ax.get_ylim()[0], "Prep.", ha="center", fontsize=8, color="steelblue",   style="italic")
    ax.text(17, ax.get_ylim()[0], "Toss",  ha="center", fontsize=8, color="darkorange", style="italic")
    ax.text(30, ax.get_ylim()[0], "Follow", ha="center", fontsize=8, color="firebrick",  style="italic")

    path = os.path.join(OUT_DIR, "fig5_wrist_trajectory.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Figure 6: Summary Heatmap (z-score normalised) ───────────────────────────

def fig_heatmap(dfs):
    metric_keys   = ["elbow_angle", "knee_angle", "shoulder_angle", "wrist_height", "score"]
    metric_labels = ["Elbow Angle", "Knee Angle", "Shoulder Angle", "Wrist Height", "Impact Score"]

    # Build a matrix: rows = players, cols = metrics (mean of impact rows)
    matrix = {}
    for pk, pl in zip(PLAYER_KEYS, PLAYER_LABELS):
        df_imp  = impact_row(dfs[pk])
        matrix[pl] = [df_imp[mk].mean() for mk in metric_keys]

    df_heat = pd.DataFrame(matrix, index=metric_labels).T

    # Z-score normalise across players (per column)
    df_z = (df_heat - df_heat.mean()) / df_heat.std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("Player Biomechanics Summary at Impact", fontsize=14, fontweight="bold")

    # Left: raw values
    sns.heatmap(
        df_heat, ax=ax1, annot=True, fmt=".1f",
        cmap="YlOrRd", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Raw Value"},
        annot_kws={"size": 10},
    )
    ax1.set_title("Raw Mean Values at Impact", fontsize=11, fontweight="bold")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=25, labelsize=9)
    ax1.tick_params(axis="y", rotation=0,  labelsize=10)

    # Right: z-score
    sns.heatmap(
        df_z, ax=ax2, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Z-score"},
        annot_kws={"size": 10},
    )
    ax2.set_title("Z-Score Normalised (relative to group mean)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=25, labelsize=9)
    ax2.tick_params(axis="y", rotation=0,  labelsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Figure 7: Consistency Bar Chart ──────────────────────────────────────────

def fig_consistency(dfs):
    metric_keys   = ["elbow_angle", "knee_angle", "shoulder_angle"]
    metric_labels = ["Elbow", "Knee", "Shoulder"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x      = np.arange(len(metric_labels))
    width  = 0.25
    offset = [-1, 0, 1]

    for i, (pk, pl) in enumerate(zip(PLAYER_KEYS, PLAYER_LABELS)):
        df_imp = impact_row(dfs[pk])
        means  = [df_imp[mk].mean() for mk in metric_keys]
        stds   = [df_imp[mk].std()  for mk in metric_keys]
        color  = PLAYER_COLORS[pl]

        bars = ax.bar(
            x + offset[i] * width, means,
            width=width * 0.9, label=pl,
            color=color, alpha=0.8,
            yerr=stds, capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
        )
        # Value labels
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{m:.0f}°", ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Peak Angle at Impact (°)", fontsize=11)
    ax.set_title("Mean Peak Joint Angles at Serve Impact (Error bars = ±1 SD)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.set_ylim(0, 210)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = os.path.join(OUT_DIR, "fig7_bar_consistency.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading serve data...")
    dfs = load_all_serves()
    for pk in PLAYER_KEYS:
        n_serves = dfs[pk]["serve_id"].nunique()
        print(f"  {pk}: {n_serves} serves")

    summary_df = pd.read_csv("outputs/summary_statistics.csv")

    print("\nGenerating figures...")
    paths = []
    paths.append(fig_radar(dfs, summary_df))
    paths.append(fig_violin_box(dfs))
    paths.append(fig_motion_profile(dfs))
    paths.append(fig_scatter_elbow_knee(dfs))
    paths.append(fig_wrist_trajectory(dfs))
    paths.append(fig_heatmap(dfs))
    paths.append(fig_consistency(dfs))

    print("\nAll figures saved:")
    for p in paths:
        print(f"  {p}")
