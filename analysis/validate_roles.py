"""
Checks for the role assignment.

1. Official positions: does each starter's slot sit where their listed position
   says it should? Tested as pairwise order within each team, half and state:
     front to back  defender < midfielder < forward (by slot depth)
     left to right  a left sided role sits left of a right sided role
2. Rotation: how often a player is in their own slot, by position group.
3. Camera: are estimated (off camera) positions tighter and more 'in slot'
   than positions the camera actually saw?
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LEVEL = {**dict.fromkeys(["CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"], 0),
         **dict.fromkeys(["DM", "LDM", "RDM", "LM", "RM", "AM"], 1),
         **dict.fromkeys(["CF", "LW", "RW", "LF", "RF"], 2)}
SIDE = {**dict.fromkeys(["LCB", "LB", "LWB", "LDM", "LM", "LW", "LF"], 1),
        **dict.fromkeys(["RCB", "RB", "RWB", "RDM", "RM", "RW", "RF"], -1)}
GROUP = {**dict.fromkeys(["CB", "LCB", "RCB"], "Centre back"),
         **dict.fromkeys(["LB", "RB", "LWB", "RWB"], "Full back"),
         **dict.fromkeys(["DM", "LDM", "RDM"], "Defensive mid"),
         **dict.fromkeys(["LM", "RM", "AM"], "Wide or attacking mid"),
         **dict.fromkeys(["LW", "RW", "LF", "RF"], "Winger"),
         "CF": "Centre forward"}


def official_roles():
    rows = []
    for p in (REPO / "data/raw").glob("*/*_match.json"):
        m = json.loads(p.read_text())
        for pl in m["players"]:
            rows.append((m["id"], pl["id"], pl["player_role"]["acronym"]))
    return pd.DataFrame(rows, columns=["match_id", "player_id", "role"])


def main():
    r = pd.concat(pd.read_parquet(p) for p in sorted((REPO / "data/processed/roles").glob("*.parquet")))
    off = official_roles()
    r = r.merge(off, on=["match_id", "player_id"], how="left")
    r["own"] = r.player_id == r.role_player

    # slot templates, labelled by the slot's main player
    tmpl = (r.groupby(["match_id", "team_id", "period", "state", "role_player"])[["rx", "ry"]]
              .mean().reset_index()
              .merge(off.rename(columns={"player_id": "role_player"}), on=["match_id", "role_player"]))
    tmpl = tmpl[tmpl.role.isin(LEVEL)]

    # ---- 1. pairwise order checks ----
    depth_ok, depth_n, side_ok, side_n = 0, 0, 0, 0
    by_state = {}
    for key, g in tmpl.groupby(["match_id", "team_id", "period", "state"]):
        st = key[3]
        for a, b in combinations(g.itertuples(), 2):
            if LEVEL[a.role] != LEVEL[b.role]:
                lo, hi = (a, b) if LEVEL[a.role] < LEVEL[b.role] else (b, a)
                ok = lo.rx < hi.rx
                depth_ok += ok; depth_n += 1
                by_state.setdefault((st, "depth"), []).append(ok)
            sa, sb = SIDE.get(a.role, 0), SIDE.get(b.role, 0)
            if sa * sb == -1:
                le, ri = (a, b) if sa == 1 else (b, a)
                ok = le.ry > ri.ry
                side_ok += ok; side_n += 1
                by_state.setdefault((st, "side"), []).append(ok)
    print(f"Front to back order correct: {depth_ok / depth_n:.1%} of {depth_n:,} pairs")
    print(f"Left to right order correct: {side_ok / side_n:.1%} of {side_n:,} pairs")
    for k, v in sorted(by_state.items()):
        print(f"   {k[0]:>3} {k[1]:<5} {np.mean(v):.1%}")

    # ---- 2. rotation by position group (starters, detected or not) ----
    st = r[r.role.isin(GROUP)].copy()
    st["group"] = st.role.map(GROUP)
    rot = st.groupby(["group", "state"]).own.mean().unstack().round(3)
    print("\nShare of time in own slot, by official position")
    print(rot.loc[["Centre back", "Full back", "Defensive mid", "Wide or attacking mid",
                   "Winger", "Centre forward"]].to_string())

    # ---- 3. camera check ----
    print(f"\nPlayer frames seen by the camera: {r.detected.mean():.1%}")
    t = r.groupby(["match_id", "team_id", "period", "state", "slot"])[["rx", "ry"]].transform("mean")
    r["dist_to_slot"] = np.hypot(r.rx - t.rx, r.ry - t.ry)
    cam = r.groupby(["state", "detected"]).agg(in_own_slot=("own", "mean"),
                                                dist_to_slot_m=("dist_to_slot", "median"))
    print(cam.round(3).to_string())
    # same comparison within each player, so it is not driven by who is off camera
    pp = (r.groupby(["match_id", "player_id", "state", "detected"])
            .agg(own=("own", "mean"), dist=("dist_to_slot", "median"), n=("own", "size"))
            .reset_index())
    pp = pp[pp.n >= 200].pivot_table(index=["match_id", "player_id", "state"],
                                     columns="detected", values=["own", "dist"]).dropna()
    diff_own = (pp[("own", False)] - pp[("own", True)])
    diff_d = (pp[("dist", False)] - pp[("dist", True)])
    print(f"Within player, estimated minus seen: in own slot {diff_own.mean():+.3f} "
          f"(estimated higher for {(diff_own > 0).mean():.0%} of players), "
          f"distance to slot {diff_d.mean():+.2f} m")

    # off camera players are far from the ball, and players far from the ball hold
    # position more anyway, so compare within player AND within ball distance band
    b = r.dropna(subset=["ball_x"]).copy()
    b["band"] = pd.cut(np.hypot(b.x - b.ball_x, b.y - b.ball_y), [0, 10, 20, 30, 40, 60, 120])
    bb = (b.groupby(["match_id", "player_id", "state", "band", "detected"], observed=True)
            .own.agg(["mean", "size"]).reset_index())
    bb = bb[bb["size"] >= 50].pivot_table(index=["match_id", "player_id", "state", "band"],
                                          columns="detected", values="mean", observed=True).dropna()
    print(f"Within player and ball distance band, estimated minus seen: "
          f"{(bb[False] - bb[True]).mean():+.3f} (n = {len(bb):,})")
    band = b.groupby(["state", "band"], observed=True).own.mean().unstack(0).round(3)
    print("\nShare of time in own slot by distance from the ball (m)")
    print(band.to_string())
    band.to_csv(REPO / "data/processed/roles_by_ball_distance.csv")

    out = REPO / "data/processed"
    rot.to_csv(out / "roles_rotation_by_position.csv")
    cam.round(4).to_csv(out / "roles_camera_check.csv")


if __name__ == "__main__":
    main()
