"""
Team shape extraction pipeline for SkillCorner tracking data.

For every frame and both teams, computes four shape features on outfield players:
  width        max y minus min y (m)
  depth        max x minus min x (m)
  compactness  mean distance from each player to the team centroid (m)
  line_height  distance of the deepest outfield player from the team's own goal line (m)

Coordinates are normalized per team so that every team always attacks toward +x.
Frames are labeled with possession and with the phase of play from SkillCorner's
phases_of_play file (team in possession gets the in possession phase, the other team
gets the out of possession phase).

Usage (from the repo root, after src/download_data.py):
    python src/shape_pipeline.py
Writes one parquet per match to data/processed/shape_features/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- 1. Load ----------

def load_match_meta(match_dir: Path):
    mid = match_dir.name
    meta = json.loads((match_dir / f"{mid}_match.json").read_text())
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    players = pd.DataFrame(
        {
            "player_id": p["id"],
            "team_id": p["team_id"],
            "is_gk": p["player_role"]["acronym"] == "GK",
        }
        for p in meta["players"]
    )
    return meta, home_id, away_id, players


def load_tracking(match_dir: Path) -> pd.DataFrame:
    """Flatten tracking jsonl into one row per player per frame."""
    mid = match_dir.name
    rows = []
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl") as f:
        for line in f:
            d = json.loads(line)
            period = d["period"]
            if period is None or not d["player_data"]:
                continue
            frame = d["frame"]
            group = d["possession"]["group"]
            for p in d["player_data"]:
                rows.append((frame, period, group, p["player_id"], p["x"], p["y"]))
    return pd.DataFrame(
        rows, columns=["frame", "period", "possession_group", "player_id", "x", "y"]
    )


# ---------- 2. Clean ----------

def clean(track, meta, home_id, away_id, players):
    df = track.merge(players, on="player_id", how="left")
    df["is_gk"] = df["is_gk"].fillna(False).astype(bool)
    # Goalkeepers are kept here only for the depth_incl_gk validation column;
    # shape_features() drops them before computing the four features.

    # Direction fix: home_team_side lists the home team's direction per period.
    # Normalize so each team attacks toward +x. Flip both x and y (a 180 degree
    # rotation) so a team's left flank stays on the same side.
    home_side = {i + 1: s for i, s in enumerate(meta["home_team_side"])}
    home_ltr = df["period"].map(lambda p: home_side[p] == "left_to_right")
    is_home = df["team_id"] == home_id
    attacks_ltr = np.where(is_home, home_ltr, ~home_ltr)
    sign = np.where(attacks_ltr, 1.0, -1.0)
    df["x_n"] = df["x"] * sign
    df["y_n"] = df["y"] * sign

    # Possession tag from tracking: which team has the ball this frame
    poss_team = df["possession_group"].map({"home team": home_id, "away team": away_id})
    df["in_possession"] = np.where(poss_team.isna(), np.nan, poss_team == df["team_id"])
    return df


# ---------- 3. Shape features ----------

def shape_features(df, half_length):
    # Team length including the goalkeeper, which is how SkillCorner defines
    # length in phases_of_play. Used only to validate against their numbers.
    gx = df.groupby(["frame", "team_id"])["x_n"]
    depth_gk = (gx.max() - gx.min()).rename("depth_incl_gk")

    # Drop goalkeepers, keep outfield only
    df = df[~df["is_gk"]].copy()
    g = df.groupby(["frame", "team_id"])
    out = g.agg(
        period=("period", "first"),
        in_possession=("in_possession", "first"),
        n_players=("x_n", "size"),
        x_min=("x_n", "min"),
        x_max=("x_n", "max"),
        y_min=("y_n", "min"),
        y_max=("y_n", "max"),
        cx=("x_n", "mean"),
        cy=("y_n", "mean"),
    ).reset_index()

    out["width"] = out["y_max"] - out["y_min"]
    out["depth"] = out["x_max"] - out["x_min"]
    out["line_height"] = out["x_min"] + half_length  # meters from own goal line

    df = df.merge(out[["frame", "team_id", "cx", "cy"]], on=["frame", "team_id"])
    df["d"] = np.hypot(df["x_n"] - df["cx"], df["y_n"] - df["cy"])
    comp = df.groupby(["frame", "team_id"])["d"].mean().rename("compactness")
    out = out.merge(comp.reset_index(), on=["frame", "team_id"])
    out = out.merge(depth_gk.reset_index(), on=["frame", "team_id"], how="left")
    out = out.rename(columns={"cx": "centroid_x"})
    return out[
        ["frame", "period", "team_id", "in_possession", "n_players",
         "width", "depth", "compactness", "line_height", "centroid_x", "depth_incl_gk"]
    ]


# ---------- Phase labels ----------

def attach_phases(feat, phases):
    """Label each team-frame with its phase of play (frames outside phases stay NaN)."""
    phases = phases.reset_index(drop=True)
    starts = phases["frame_start"].to_numpy()
    ends = phases["frame_end"].to_numpy()
    idx = np.searchsorted(starts, feat["frame"].to_numpy(), side="right") - 1
    valid = (idx >= 0) & (feat["frame"].to_numpy() < ends[np.clip(idx, 0, None)])
    idx = np.where(valid, idx, -1)

    ph = phases.iloc[np.clip(idx, 0, None)]
    in_poss_team = ph["team_in_possession_id"].to_numpy()
    team_has_ball = feat["team_id"].to_numpy() == in_poss_team
    feat["phase_index"] = np.where(valid, ph["index"].to_numpy(), np.nan)
    feat["phase_team_in_possession"] = np.where(valid, team_has_ball, np.nan)
    feat["phase"] = np.where(
        ~valid, None,
        np.where(team_has_ball,
                 ph["team_in_possession_phase_type"].to_numpy(),
                 ph["team_out_of_possession_phase_type"].to_numpy()),
    )
    return feat


def process_match(match_dir: Path):
    meta, home_id, away_id, players = load_match_meta(match_dir)
    track = load_tracking(match_dir)
    df = clean(track, meta, home_id, away_id, players)
    feat = shape_features(df, meta["pitch_length"] / 2)
    phases = pd.read_csv(match_dir / f"{match_dir.name}_phases_of_play.csv")
    feat = attach_phases(feat, phases)
    feat.insert(0, "match_id", int(match_dir.name))
    names = {home_id: meta["home_team"]["short_name"], away_id: meta["away_team"]["short_name"]}
    feat.insert(3, "team_name", feat["team_id"].map(names))
    feat["is_home"] = feat["team_id"] == home_id
    return feat, phases


REPO = Path(__file__).resolve().parent.parent


def compact(feat):
    """Shrink the table for storage: float32 and 2 decimals (1 cm precision)."""
    for c in ["width", "depth", "compactness", "line_height", "centroid_x", "depth_incl_gk"]:
        feat[c] = feat[c].round(2).astype("float32")
    for c in ["in_possession", "phase_team_in_possession"]:
        feat[c] = feat[c].astype("float32")
    feat["phase_index"] = feat["phase_index"].astype("float32")
    feat["phase"] = feat["phase"].astype("category")
    feat["team_name"] = feat["team_name"].astype("category")
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=REPO / "data" / "raw")
    ap.add_argument("--out", default=REPO / "data" / "processed" / "shape_features")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for match_dir in sorted(Path(args.data).iterdir()):
        if not match_dir.is_dir():
            continue
        if not (match_dir / f"{match_dir.name}_tracking_extrapolated.jsonl").exists():
            print(f"{match_dir.name}: tracking missing, run src/download_data.py first")
            continue
        feat, _ = process_match(match_dir)
        compact(feat).to_parquet(out / f"{match_dir.name}.parquet", index=False,
                                 compression="zstd")
        total += len(feat)
        print(f"{match_dir.name}: {len(feat):,} team-frames, "
              f"{feat['phase'].notna().mean():.0%} inside a phase")
    print(f"Saved {total:,} rows to {out}/")


if __name__ == "__main__":
    main()
