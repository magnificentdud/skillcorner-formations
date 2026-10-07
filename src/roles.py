"""
Role assignment: give every outfield player a formation slot in every frame.

Method (a simplified version of Bialkowski et al., 2014, "Large-scale analysis
of soccer matches using spatiotemporal tracking data"):

  For each team, half and possession state separately:
  1. Express each outfield player's position relative to the team centroid,
     so the slots describe the formation, not where the block is on the pitch.
  2. Start with 10 slot templates = the average relative positions of the 10
     players who appear most in that half and state.
  3. In every frame, assign the 10 players to the 10 slots one to one, minimizing
     total squared distance (Hungarian algorithm).
  4. Move each template to the average position of everyone assigned to it.
  5. Repeat 3 and 4 until assignments stop changing.

Because slots are reassigned every frame, substitutions and position swaps are
handled automatically: a player who swaps wings changes slot.

In possession and out of possession are fitted separately because teams use
different shapes. Slots are linked across the two states (and halves) through
the player who occupies them most, see link_slots().

Usage (repo root):
    python src/roles.py                 # all matches, writes data/processed/roles/
    python src/roles.py 1874553         # one match
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "processed" / "roles"
STEP = 2          # use every 2nd frame (5 Hz) to keep the prototype fast
MAX_ITER = 30


# ---------- load ----------

def load_match(match_dir: Path):
    """One row per outfield player per frame, normalized so each team attacks +x."""
    mid = match_dir.name
    meta = json.loads((match_dir / f"{mid}_match.json").read_text())
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}
    gk = {p["id"] for p in meta["players"] if p["player_role"]["acronym"] == "GK"}
    home_side = {i + 1: s for i, s in enumerate(meta["home_team_side"])}

    rows = []
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl") as f:
        for line in f:
            d = json.loads(line)
            fr, period = d["frame"], d["period"]
            if period is None or fr % STEP or not d["player_data"]:
                continue
            group = d["possession"]["group"]
            poss = home_id if group == "home team" else away_id if group == "away team" else 0
            bx, by = d["ball_data"]["x"], d["ball_data"]["y"]
            for p in d["player_data"]:
                pid = p["player_id"]
                if pid in gk or pid not in team_of:
                    continue
                rows.append((fr, period, team_of[pid], pid, p["x"], p["y"],
                             p["is_detected"], poss, bx, by))
    df = pd.DataFrame(rows, columns=["frame", "period", "team_id", "player_id", "x", "y",
                                     "detected", "poss_team", "ball_x", "ball_y"])
    home_ltr = df["period"].map(lambda p: home_side[p] == "left_to_right")
    attacks_ltr = np.where(df["team_id"] == home_id, home_ltr, ~home_ltr)
    sign = np.where(attacks_ltr, 1.0, -1.0)
    for c in ["x", "y", "ball_x", "ball_y"]:
        df[c] = df[c] * sign
    df["state"] = np.where(df["poss_team"] == 0, "none",
                           np.where(df["poss_team"] == df["team_id"], "in", "out"))
    df["detected"] = df["detected"].fillna(False).astype(bool)
    return df.drop(columns="poss_team"), meta


# ---------- assign ----------

def assign_group(g: pd.DataFrame):
    """Fit slots for one team, half and state. Returns slot per row and templates."""
    # keep frames with exactly 10 outfield players (drops red card periods)
    g = g[g.groupby("frame")["player_id"].transform("size") == 10]
    g = g.sort_values(["frame", "player_id"])
    n = len(g) // 10
    P = g[["x", "y"]].to_numpy().reshape(n, 10, 2)
    R = P - P.mean(axis=1, keepdims=True)                 # relative to centroid
    ids = g["player_id"].to_numpy().reshape(n, 10)

    top = g["player_id"].value_counts().index[:10]
    rel = pd.DataFrame(R.reshape(-1, 2), columns=["rx", "ry"]).assign(pid=ids.ravel())
    T = rel.groupby("pid")[["rx", "ry"]].mean().loc[top].to_numpy()

    slot = np.zeros((n, 10), dtype=np.int8)
    prev = None
    for it in range(MAX_ITER):
        cost = ((R[:, :, None, :] - T[None, None, :, :]) ** 2).sum(-1)   # n x player x slot
        for i in range(n):
            _, cols = linear_sum_assignment(cost[i])
            slot[i] = cols
        T = np.array([R[slot == k].mean(axis=0) for k in range(10)])
        if prev is not None and (prev == slot).mean() > 0.999:
            break
        prev = slot.copy()
    out = g.copy()
    out["slot"] = slot.ravel()
    out["rx"], out["ry"] = R[:, :, 0].ravel(), R[:, :, 1].ravel()
    return out, T, it + 1


def link_slots(df: pd.DataFrame):
    """
    Give slots ids that are consistent across halves and possession states within
    a match: each (team, period, state) slot is mapped to the player who fills it
    most, and that player's starting identity becomes the slot's id.
    Returns df with 'role_player' = the player whose slot this is.
    """
    own = (df.groupby(["team_id", "period", "state", "slot", "player_id"]).size()
             .rename("n").reset_index())
    own = own.sort_values("n", ascending=False)
    # greedy one to one: each slot gets its most frequent player not already taken
    rows = []
    for key, d in own.groupby(["team_id", "period", "state"]):
        taken_p, taken_s = set(), set()
        for r in d.itertuples():
            if r.slot in taken_s or r.player_id in taken_p:
                continue
            taken_s.add(r.slot); taken_p.add(r.player_id)
            rows.append((*key, r.slot, r.player_id))
    m = pd.DataFrame(rows, columns=["team_id", "period", "state", "slot", "role_player"])
    return df.merge(m, on=["team_id", "period", "state", "slot"], how="left")


def process(match_dir: Path):
    df, meta = load_match(match_dir)
    parts, info = [], []
    for (team, period, state), g in df[df.state != "none"].groupby(["team_id", "period", "state"]):
        a, T, iters = assign_group(g)
        parts.append(a)
        info.append((team, period, state, iters, len(a) // 10))
    out = link_slots(pd.concat(parts, ignore_index=True))
    out.insert(0, "match_id", int(match_dir.name))
    return out, pd.DataFrame(info, columns=["team_id", "period", "state", "iterations", "frames"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:] or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    for mid in ids:
        out, info = process(RAW / mid)
        for c in ["x", "y", "rx", "ry", "ball_x", "ball_y"]:
            out[c] = out[c].round(2).astype("float32")
        out.to_parquet(OUT / f"{mid}.parquet", index=False, compression="zstd")
        print(f"{mid}: {len(out):,} player frames, iterations "
              f"{info.iterations.min()} to {info.iterations.max()}")


if __name__ == "__main__":
    main()
