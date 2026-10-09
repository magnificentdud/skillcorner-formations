"""
Average formation with and without the ball for a team in one half of a match.

Each slot is labelled with the official position of the player who fills it
most, and an arrow shows how that slot moves from out of possession (orange)
to in possession (blue).

Usage (repo root):
    python analysis/formation_plot.py overview          # every team match, small multiples
    python analysis/formation_plot.py MATCH_ID TEAM_ID  # one large figure
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#c9c8c2"


def meta(mid):
    m = json.loads((REPO / f"data/raw/{mid}/{mid}_match.json").read_text())
    roles = {p["id"]: p["player_role"]["acronym"] for p in m["players"]}
    names = {p["id"]: p["short_name"] for p in m["players"]}
    teams = {m["home_team"]["id"]: m["home_team"]["short_name"],
             m["away_team"]["id"]: m["away_team"]["short_name"]}
    return roles, names, teams


def slot_means(mid, team, period=1):
    r = pd.read_parquet(REPO / f"data/processed/roles/{mid}.parquet")
    r = r[(r.team_id == team) & (r.period == period)]
    s = r.groupby(["state", "role_player"])[["x", "y"]].mean().reset_index()
    own = (r.player_id == r.role_player).groupby([r.state, r.role_player]).mean().rename("own")
    return s.merge(own.reset_index(), on=["state", "role_player"])


def draw_pitch(ax, lw=0.8):
    kw = dict(color=LINE, lw=lw, zorder=0)
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], **kw)
    ax.plot([0, 0], [-34, 34], **kw)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        x0 = 52.5 * s
        ax.plot([x0, x0 - 16.5 * s, x0 - 16.5 * s, x0], [-20.16, -20.16, 20.16, 20.16], **kw)
        ax.plot([x0, x0 - 5.5 * s, x0 - 5.5 * s, x0], [-9.16, -9.16, 9.16, 9.16], **kw)
    ax.set_xlim(-54, 54); ax.set_ylim(-35.5, 35.5)
    ax.set_aspect("equal"); ax.axis("off")


def draw_team(ax, s, roles, size=60, labels=True, fs=7):
    inn = s[s.state == "in"].set_index("role_player")
    out = s[s.state == "out"].set_index("role_player")
    for pid in inn.index.intersection(out.index):
        ax.annotate("", xy=(inn.x[pid], inn.y[pid]), xytext=(out.x[pid], out.y[pid]),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                    shrinkA=4, shrinkB=4, mutation_scale=7), zorder=2)
    ax.scatter(out.x, out.y, s=size, color=ORANGE, edgecolor="white", lw=1.2, zorder=3)
    ax.scatter(inn.x, inn.y, s=size, color=BLUE, edgecolor="white", lw=1.2, zorder=3)
    if labels:
        pts = pd.concat([inn[["x", "y"]], out[["x", "y"]]]).to_numpy()
        for pid, row in inn.iterrows():
            # label above the dot unless another dot sits just above it
            above = ((abs(pts[:, 0] - row.x) < 5) & (pts[:, 1] - row.y > 0.5)
                     & (pts[:, 1] - row.y < 6)).any()
            dy, va = (-7, "top") if above else (6, "bottom")
            ax.annotate(roles.get(pid, "?"), (row.x, row.y), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va, fontsize=fs, color=INK)


def overview():
    files = sorted((REPO / "data/processed/roles").glob("*.parquet"))
    pairs = []
    for f in files:
        mid = int(f.stem)
        _, _, teams = meta(mid)
        pairs += [(mid, t, n) for t, n in teams.items()]
    fig, axes = plt.subplots(8, 5, figsize=(15, 16))
    for ax, (mid, t, n) in zip(axes.flat, pairs):
        roles, _, _ = meta(mid)
        draw_pitch(ax, lw=0.5)
        draw_team(ax, slot_means(mid, t), roles, size=18, fs=5)
        ax.set_title(f"{n}  {mid}  ({t})", fontsize=7, color=INK2)
    fig.tight_layout()
    out = sys.argv[2] if len(sys.argv) > 2 else REPO / "figures/formations_overview.png"
    fig.savefig(out, dpi=110, facecolor="white")
    print(out)


def single(mid, team):
    roles, names, teams = meta(mid)
    s = slot_means(mid, team)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    draw_pitch(ax)
    draw_team(ax, s, roles, size=110, fs=8)
    ax.annotate("", xy=(30, -38.5), xytext=(-30, -38.5), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.text(0, -40.5, "attacking direction", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.scatter([], [], s=60, color=ORANGE, label="Out of possession")
    ax.scatter([], [], s=60, color=BLUE, label="In possession")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{teams[team]}: average shape without and with the ball (first half)",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = REPO / f"figures/formation_{mid}_{team}.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    if __name__ == "__main__":
    matplotlib.use("Agg")
    if sys.argv[1] == "overview":
        overview()
    else:
        single(int(sys.argv[1]), int(sys.argv[2]))
