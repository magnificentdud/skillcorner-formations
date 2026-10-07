"""
Validation and sanity check figures for the shape pipeline.

1. Validation: compare our width and team length (with GK) against the values
   SkillCorner publishes in phases_of_play at each phase start and end.
2. Sanity check: mean width, depth, compactness and line height per phase pairing,
   attacking team vs defending team, pooled over all matches.

Usage (from the repo root, after src/shape_pipeline.py):
    python src/validate_and_plot.py
Reads data/processed/shape_features/ and data/raw/*/phases_of_play.csv,
writes summary CSVs to data/processed/ and figures to figures/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def validation_table(feat, phases):
    teams = feat.groupby("match_id")["team_id"].unique()
    phases = phases.copy()
    phases["opp_id"] = [
        [t for t in teams[m] if t != tp][0]
        for m, tp in zip(phases["match_id"], phases["team_in_possession_id"])
    ]
    k = feat.set_index(["match_id", "frame", "team_id"])
    rows = []
    for side, tcol, pre in [("in possession", "team_in_possession_id", "team_in_possession"),
                            ("out of possession", "opp_id", "team_out_of_possession")]:
        for when in ["start", "end"]:
            idx = pd.MultiIndex.from_arrays(
                [phases["match_id"], phases[f"frame_{when}"], phases[tcol]])
            mine = k.reindex(idx)
            for ours, theirs in [("width", "width"), ("depth_incl_gk", "length")]:
                a = mine[ours].to_numpy()
                b = phases[f"{pre}_{theirs}_{when}"].to_numpy()
                m = ~np.isnan(a) & ~np.isnan(b)
                err = np.abs(a[m] - b[m])
                rows.append({
                    "team": side, "phase_boundary": when, "feature": ours,
                    "n": int(m.sum()), "pearson_r": np.corrcoef(a[m], b[m])[0, 1],
                    "median_abs_err_m": np.median(err),
                    "pct_within_0.5m": (err <= 0.5).mean() * 100,
                })
    # Rows for the scatter: defending team at phase start
    idx = pd.MultiIndex.from_arrays(
        [phases["match_id"], phases["frame_start"], phases["opp_id"]])
    return pd.DataFrame(rows), k.reindex(idx), phases


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def sanity_figure(feat, out_path):
    pairs = [("build_up", "high_block", "Build up\nvs high block"),
             ("create", "medium_block", "Create\nvs medium block"),
             ("finish", "low_block", "Finish\nvs low block")]
    means = feat.dropna(subset=["phase"]).groupby("phase")[
        ["width", "depth", "compactness", "line_height"]].mean()
    metrics = [("width", "Width (m)"), ("depth", "Depth (m)"),
               ("compactness", "Compactness (m)"),
               ("line_height", "Defensive line height (m from own goal)")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharey=True)
    ys = np.arange(len(pairs))[::-1]
    for ax, (col, title) in zip(axes.flat, metrics):
        style(ax)
        for y, (att, dfn, _) in zip(ys, pairs):
            a, d = means.loc[att, col], means.loc[dfn, col]
            ax.plot([a, d], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter(a, y, s=70, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
            ax.scatter(d, y, s=70, color=ORANGE, edgecolor="white", linewidth=2, zorder=3)
            close = abs(a - d) < 1.5
            ax.annotate(f"{a:.0f}", (a, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK2)
            ax.annotate(f"{d:.0f}", (d, y), xytext=(0, -9 if close else 9),
                        textcoords="offset points", ha="center",
                        va="top" if close else "bottom", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
        ax.set_yticks(ys, [p[2] for p in pairs], fontsize=9, color=INK2)
        ax.set_ylim(-0.6, len(pairs) - 0.3)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c,
                          markeredgecolor="white") for c in (BLUE, ORANGE)]
    fig.legend(handles, ["Team in possession", "Team out of possession"],
               loc="upper right", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.98, 0.99), labelcolor=INK2)
    n_matches = feat["match_id"].nunique()
    fig.suptitle(f"Team shape by phase of play, outfield players, "
                 f"{n_matches} SkillCorner open data matches",
                 x=0.02, ha="left", fontsize=11, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def validation_figure(mine, phases, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, ours, theirs, title in [
        (axes[0], "width", "team_out_of_possession_width_start", "Width (m)"),
        (axes[1], "depth_incl_gk", "team_out_of_possession_length_start", "Length incl. GK (m)"),
    ]:
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        a, b = mine[ours].to_numpy(), phases[theirs].to_numpy()
        m = ~np.isnan(a) & ~np.isnan(b)
        lo, hi = min(a[m].min(), b[m].min()), max(a[m].max(), b[m].max())
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle="--", zorder=1)
        ax.scatter(b[m], a[m], s=4, color=BLUE, alpha=0.25, linewidth=0, zorder=2)
        r = np.corrcoef(a[m], b[m])[0, 1]
        ax.set_title(f"{title}   r = {r:.3f}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("SkillCorner phases_of_play value", fontsize=9, color=INK2)
        ax.set_ylabel("Our pipeline", fontsize=9, color=INK2)
    fig.suptitle("Pipeline vs SkillCorner published values (defending team, phase starts)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


REPO = Path(__file__).resolve().parent.parent


def main():
    out = REPO / "data" / "processed"
    figs = REPO / "figures"
    figs.mkdir(exist_ok=True)
    feat = pd.concat(
        (pd.read_parquet(p) for p in sorted((out / "shape_features").glob("*.parquet"))),
        ignore_index=True)
    feat["phase"] = feat["phase"].astype("object")
    feat["team_name"] = feat["team_name"].astype("object")
    phases = pd.concat(
        (pd.read_csv(p) for p in sorted((REPO / "data" / "raw").glob("*/*_phases_of_play.csv"))),
        ignore_index=True)

    table, mine, phases = validation_table(feat, phases)
    table.round(3).to_csv(out / "validation_vs_skillcorner.csv", index=False)
    print(table.round(3).to_string(index=False))

    summary = (feat.dropna(subset=["phase"])
               .groupby(["team_name", "phase"])[["width", "depth", "compactness", "line_height"]]
               .mean().round(2))
    summary["seconds"] = (feat.dropna(subset=["phase"])
                          .groupby(["team_name", "phase"]).size() / 10).round(0)
    summary.to_csv(out / "shape_by_team_and_phase.csv")

    sanity_figure(feat, figs / "fig_shape_by_phase.png")
    validation_figure(mine, phases, figs / "fig_validation.png")
    print("Figures written.")


if __name__ == "__main__":
    main()
