import argparse
import bisect
import copy
import json
import math
import os
import pickle
import ast
from collections import defaultdict, deque, Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError: 
    HAVE_LGB = False

CFG = dict(
    elo_init=1500.0,
    hfa=65.0,  
    k_club=20.0,
    k_player=1.2,      
    player_decay=0.002,
    player_cap=120.0,
    form_n=10, 
    usual_n=10,
    min_starters=11,
    coverage_min=0.90,
    min_games_season=300,
    ema_alpha=0.2,
    comp_prior_n=50.0,
    val_seasons=1,
    test_seasons=1,
)

META = ["game_id", "date", "season", "competition_id", "home", "away",
        "home_goals", "away_goals", "gd", "y", "has_lineups", "split"]

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return []
    try:
        return [str(p) for p in list(x) if p is not None and str(p) != "nan"]
    except TypeError:
        return []


def prepare(df, cfg=CFG):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    for c in ["home_starters", "home_substitutes", "away_starters", "away_substitutes"]:
        d[c] = d[c].map(_as_list)
    d["has_goals"] = d["home_club_goals"].notna() & d["away_club_goals"].notna()
    m = cfg["min_starters"]
    d["has_lineups"] = (d["home_starters"].map(len) >= m) & (d["away_starters"].map(len) >= m)
    return d.sort_values(["date", "game_id"]).reset_index(drop=True)


def coverage_report(d, cfg=CFG):
    """Share of scored games with both starting lineups, per season and per competition-season."""
    g = d[d["has_goals"]]
    by_season = (g.groupby("season")
                 .agg(n_games=("game_id", "size"), lineup_share=("has_lineups", "mean"),
                      first=("date", "min"), last=("date", "max"))
                 .sort_values("first"))
    by_season["good"] = ((by_season["lineup_share"] >= cfg["coverage_min"])
                         & (by_season["n_games"] >= cfg["min_games_season"]))
    by_comp = (g.groupby(["season", "competition_id"])
               .agg(n_games=("game_id", "size"), lineup_share=("has_lineups", "mean"))
               .reset_index())
    return by_season, by_comp

def _exp(diff):
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov(gd):
    return math.log(abs(gd) + 1.0) + 1.0


def pair_features(h, a, comp):
    """Feature row (home minus away where sensible) from two club snapshots."""
    f = {}
    f["elo_diff"] = h["elo_p"] - a["elo_p"]
    f["elo_mean"] = 0.5 * (h["elo_p"] + a["elo_p"])
    f["joint_club_diff"] = h["elo_j"] - a["elo_j"]
    f["joint_diff"] = (h["elo_j"] + h["adj"]) - (a["elo_j"] + a["adj"])
    f["lineup_adj_diff"] = h["adj"] - a["adj"]
    f["lineup_vs_usual_h"] = h["adj"] - h["adj_usual"]
    f["lineup_vs_usual_a"] = a["adj"] - a["adj_usual"]
    f["lineup_vs_usual_diff"] = f["lineup_vs_usual_h"] - f["lineup_vs_usual_a"]
    f["missing_h"] = h["missing"]
    f["missing_a"] = a["missing"]
    f["missing_diff"] = h["missing"] - a["missing"]
    f["unseen_h"] = h["unseen"]
    f["unseen_a"] = a["unseen"]
    f["bench_diff"] = h["bench"] - a["bench"]
    f["ppg_diff"] = h["ppg"] - a["ppg"]
    f["gdf_diff"] = h["gdf"] - a["gdf"]
    f["rest_diff"] = h["rest"] - a["rest"]
    f["n_min"] = min(h["n"], a["n"])
    f["comp_home_rate"] = comp[0]
    f["comp_draw_rate"] = comp[1]
    f["comp_goals_pg"] = comp[2]
    return f


class _State:
    """All time-evolving quantities. Snapshots are taken BEFORE a date's matches are applied."""

    def __init__(self, cfg):
        self.c = cfg
        self.elo_p = defaultdict(lambda: cfg["elo_init"])    
        self.elo_j = defaultdict(lambda: cfg["elo_init"])   
        self.player_r = defaultdict(float)                  
        self.player_seen = defaultdict(int)
        self.form = defaultdict(lambda: deque(maxlen=cfg["form_n"]))
        self.n_games = defaultdict(int)
        self.last_date = {}
        self.start_hist = defaultdict(deque)
        self.start_cnt = defaultdict(Counter)
        self.adj_usual = {}
        self.bench_ema = {}
        self.comp = defaultdict(lambda: [0, 0, 0, 0.0]) 

    def comp_stats(self, comp):
        n, hw, dr, gl = self.comp[comp]
        k = self.c["comp_prior_n"]
        return ((hw + k * 0.45) / (n + k), (dr + k * 0.25) / (n + k), (gl + k * 2.7) / (n + k))

    def usual_xi(self, club):
        return {p for p, _ in self.start_cnt[club].most_common(11)}

    def snap(self, club, starters, subs, date):
        """starters=None -> 'typical' snapshot (usual lineup), used for ratings / missing lineups."""
        f = self.form[club]
        s = dict(elo_p=self.elo_p[club], elo_j=self.elo_j[club],
                 ppg=float(np.mean([x[0] for x in f])) if f else np.nan,
                 gdf=float(np.mean([x[1] for x in f])) if f else np.nan,
                 n=min(self.n_games[club], 20),
                 rest=float(min((date - self.last_date[club]).days, 30)) if club in self.last_date else np.nan,
                 adj_usual=self.adj_usual.get(club, 0.0))
        if starters is None:
            s.update(adj=s["adj_usual"], missing=0.0, unseen=0.0,
                     bench=self.bench_ema.get(club, np.nan))
        else:
            s["adj"] = float(sum(self.player_r[p] for p in starters))
            s["missing"] = (11 - len(set(starters) & self.usual_xi(club))
                            if self.start_hist[club] else np.nan)
            s["unseen"] = float(sum(1 for p in starters if self.player_seen[p] == 0))
            s["bench"] = (float(np.mean([self.player_r[p] for p in subs])) if subs else np.nan)
        return s

    def update(self, row, hs, as_):
        c = self.c
        h, a = row["home"], row["away"]
        hg, ag = row["home_goals"], row["away_goals"]
        gd = hg - ag
        S = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        m = _mov(gd)
        res_p = S - _exp(self.elo_p[h] + c["hfa"] - self.elo_p[a])
        self.elo_p[h] += c["k_club"] * m * res_p
        self.elo_p[a] -= c["k_club"] * m * res_p
        res_j = S - _exp(self.elo_j[h] + c["hfa"] + hs["adj"] - self.elo_j[a] - as_["adj"])
        self.elo_j[h] += c["k_club"] * m * res_j
        self.elo_j[a] -= c["k_club"] * m * res_j
        if row["has_lineups"]:
            for side, club, sign in (("home", h, 1.0), ("away", a, -1.0)):
                for p in row[side + "_starters"]:
                    r = self.player_r[p] * (1 - c["player_decay"]) + sign * c["k_player"] * m * res_j
                    self.player_r[p] = float(np.clip(r, -c["player_cap"], c["player_cap"]))
                    self.player_seen[p] += 1
                st = row[side + "_starters"]
                snap = hs if side == "home" else as_
                self.start_hist[club].append(st)
                self.start_cnt[club].update(st)
                if len(self.start_hist[club]) > c["usual_n"]:
                    old = self.start_hist[club].popleft()
                    for p in old:
                        self.start_cnt[club][p] -= 1
                        if self.start_cnt[club][p] <= 0:
                            del self.start_cnt[club][p]
                al = c["ema_alpha"]
                prev = self.adj_usual.get(club, snap["adj"])
                self.adj_usual[club] = al * snap["adj"] + (1 - al) * prev
                if not np.isnan(snap["bench"]):
                    pb = self.bench_ema.get(club, snap["bench"])
                    self.bench_ema[club] = al * snap["bench"] + (1 - al) * pb
        for club, gf, ga in ((h, hg, ag), (a, ag, hg)):
            pts = 3 if gf > ga else 1 if gf == ga else 0
            self.form[club].append((pts, gf - ga))
            self.n_games[club] += 1
            self.last_date[club] = row["date"]
        cs = self.comp[row["competition_id"]]
        cs[0] += 1
        cs[1] += int(gd > 0)
        cs[2] += int(gd == 0)
        cs[3] += hg + ag


def build_features(d, cfg=CFG):
    """One chronological pass. Returns feature table, per-club snapshot history, comp defaults."""
    st = _State(cfg)
    g = d[d["has_goals"]].copy()
    g = g.rename(columns={"home_club_name": "home", "away_club_name": "away",
                          "home_club_goals": "home_goals", "away_club_goals": "away_goals"})
    snap_dates, snap_vals = defaultdict(list), defaultdict(list)
    rows = []
    for date, grp in g.groupby("date", sort=True):
        pending = []
        for r in grp.to_dict("records"):
            if r["has_lineups"]:
                hs = st.snap(r["home"], r["home_starters"], r["home_substitutes"], date)
                as_ = st.snap(r["away"], r["away_starters"], r["away_substitutes"], date)
            else:
                hs = st.snap(r["home"], None, None, date)
                as_ = st.snap(r["away"], None, None, date)
            f = pair_features(hs, as_, st.comp_stats(r["competition_id"]))
            gd = r["home_goals"] - r["away_goals"]
            f.update(game_id=r["game_id"], date=date, season=r["season"],
                     competition_id=r["competition_id"], home=r["home"], away=r["away"],
                     home_goals=r["home_goals"], away_goals=r["away_goals"], gd=gd,
                     y=0 if gd > 0 else 1 if gd == 0 else 2, has_lineups=r["has_lineups"])
            rows.append(f)
            pending.append((r, hs, as_))
        for r, hs, as_ in pending:
            st.update(r, hs, as_)
        for r, _, _ in pending:
            for club in (r["home"], r["away"]):
                if snap_dates[club] and snap_dates[club][-1] == date:
                    continue
                s = st.snap(club, None, None, date)
                s["rest"] = 7.0
                snap_dates[club].append(date)
                snap_vals[club].append(s)
    feats = pd.DataFrame(rows)
    return feats, snap_dates, snap_vals


def assign_splits(feats, good_seasons, cfg=CFG):
    feats = feats.copy()
    ok = feats.has_lineups & feats.season.isin(good_seasons)
    feats["split"] = "unused"
    feats.loc[ok & (feats.date <  "2023-07-01"), "split"] = "train"
    feats.loc[ok & (feats.date >= "2023-07-01") & (feats.date < "2024-07-01"), "split"] = "val"
    feats.loc[ok & (feats.date >= "2024-07-01"), "split"] = "test"
    seasons = {s: sorted(feats.loc[feats.split == s, "season"].unique()) for s in ("train", "val", "test")}
    return feats, seasons

class GBM:
    def __init__(self):
        self.best_n, self.m = None, None

    def _new(self, n):
        if HAVE_LGB:
            return lgb.LGBMClassifier(objective="multiclass", n_estimators=n, learning_rate=0.03,
                                      num_leaves=15, min_child_samples=50, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                      verbose=-1, random_state=0)
        return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=50, l2_regularization=5.0,
                                              early_stopping=False, random_state=0)

    def fit(self, X, y, Xv, yv):
        if HAVE_LGB:
            m = self._new(2000)
            m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            self.best_n = max(int(m.best_iteration_ or 20), 20)
        else:
            m = self._new(300)
            m.fit(X, y)
            losses = [log_loss(yv, p, labels=[0, 1, 2]) for p in m.staged_predict_proba(Xv)]
            self.best_n = int(np.argmin(losses)) + 1
            m = self._new(self.best_n)
            m.fit(X, y)
        self.m = m
        return self

    def refit(self, X, y):
        m = self._new(int(self.best_n * 1.15) + 1)
        m.fit(X, y)
        self.m = m
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)

    def gain_importance(self, cols):
        if HAVE_LGB:
            imp = self.m.booster_.feature_importance("gain")
            return pd.Series(imp / imp.sum(), index=cols)
        return None


def brier(P, y):
    return float(((P - np.eye(3)[y]) ** 2).sum(1).mean())


def ece(P, y, bins=10):
    out = []
    for k in range(3):
        p, o = P[:, k], (y == k).astype(float)
        idx = np.minimum((p * bins).astype(int), bins - 1)
        e = sum(abs(p[idx == b].mean() - o[idx == b].mean()) * (idx == b).mean()
                for b in range(bins) if (idx == b).any())
        out.append(e)
    return float(np.mean(out))


def evaluate(P, y):
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(1, keepdims=True)
    draw = (y == 1)
    return dict(n=int(len(y)), log_loss=float(log_loss(y, P, labels=[0, 1, 2])), brier=brier(P, y),
                accuracy=float((P.argmax(1) == y).mean()), ece=ece(P, y),
                draw_mean_pred=float(P[:, 1].mean()), draw_actual=float(draw.mean()),
                draw_auc=float(roc_auc_score(draw, P[:, 1])) if 0 < draw.sum() < len(y) else np.nan)


def permutation_importance(gbm, X, y, repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    base = log_loss(y, gbm.predict_proba(X), labels=[0, 1, 2])
    out = {}
    for c in X.columns:
        d = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            d.append(log_loss(y, gbm.predict_proba(Xp), labels=[0, 1, 2]) - base)
        out[c] = float(np.mean(d))
    return pd.Series(out).sort_values(ascending=False)


def calibration_table(P, y, bins=10):
    rows = []
    for k, name in enumerate(["home", "draw", "away"]):
        idx = np.minimum((P[:, k] * bins).astype(int), bins - 1)
        for b in range(bins):
            m = idx == b
            if m.any():
                rows.append(dict(cls=name, bin=b, n=int(m.sum()), mean_pred=float(P[m, k].mean()),
                                 freq=float((y[m] == k).mean())))
    return pd.DataFrame(rows)


def plot_calibration(cal, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, name in zip(ax, ["home", "draw", "away"]):
        c = cal[cal.cls == name]
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.plot(c.mean_pred, c.freq, "o-")
        a.set_title(name)
        a.set_xlabel("predicted")
        a.set_ylabel("observed")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

class RatingSystem:
    """Turns the trained model + club state history into dated rankings.

    Rating = expected points against a fixed reference opponent (the average of the clubs being
    ranked), averaged over playing at home and away, with the club fielding its usual lineup.
    """

    def __init__(self, model, feature_cols, snap_dates, snap_vals, feats, comp_default):
        self.model, self.cols = model, feature_cols
        self.snap_dates, self.snap_vals = dict(snap_dates), dict(snap_vals)
        self.comp_default = comp_default
        self.season_end = feats.groupby("season").date.max().to_dict()
        self.season_clubs = {}
        for (s, c), g in feats.groupby(["season", "competition_id"]):
            self.season_clubs[(s, c)] = set(g.home) | set(g.away)
        self.last_date = feats.date.max()

    def _snap_at(self, club, as_of):
        i = bisect.bisect_left(self.snap_dates[club], as_of) - 1
        return self.snap_vals[club][i] if i >= 0 else None

    def ratings(self, as_of=None, season=None, competition_id=None, active_days=365, min_clubs=2):
        if season is not None:
            as_of = self.season_end[season] + pd.Timedelta(days=1)
            clubs = set()
            for (s, c), cl in self.season_clubs.items():
                if s == season and (competition_id is None or c == competition_id):
                    clubs |= cl
        else:
            as_of = pd.Timestamp(as_of) if as_of is not None else self.last_date + pd.Timedelta(days=1)
            clubs = {c for c, ds in self.snap_dates.items()
                     if ds and ds[0] < as_of and (as_of - max(x for x in ds if x < as_of)).days <= active_days}
            if competition_id is not None:
                keep = set()
                for (s, c), cl in self.season_clubs.items():
                    if c == competition_id:
                        keep |= cl
                clubs &= keep
        snaps = {c: self._snap_at(c, as_of) for c in clubs if c in self.snap_dates}
        snaps = {c: s for c, s in snaps.items() if s is not None}
        if len(snaps) < min_clubs:
            return pd.DataFrame()
        keys = list(next(iter(snaps.values())).keys())
        ref = {k: float(np.nanmean([s[k] for s in snaps.values()])) for k in keys}
        comp = self.comp_default
        names = list(snaps)
        Xh = pd.DataFrame([pair_features(snaps[c], ref, comp) for c in names])[self.cols]
        Xa = pd.DataFrame([pair_features(ref, snaps[c], comp) for c in names])[self.cols]
        Ph, Pa = self.model.predict_proba(Xh), self.model.predict_proba(Xa)
        pts = 0.5 * ((3 * Ph[:, 0] + Ph[:, 1]) + (3 * Pa[:, 2] + Pa[:, 1]))
        win = 0.5 * (Ph[:, 0] + Pa[:, 2])
        draw = 0.5 * (Ph[:, 1] + Pa[:, 1])
        out = pd.DataFrame(dict(club=names, exp_points=pts, win_prob=win, draw_prob=draw,
                                elo=[snaps[c]["elo_p"] for c in names],
                                joint_rating=[snaps[c]["elo_j"] + snaps[c]["adj_usual"] for c in names],
                                usual_lineup_adj=[snaps[c]["adj_usual"] for c in names]))
        out = out.sort_values("exp_points", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        out.attrs["as_of"] = as_of
        return out


def league_table(d, season, competition_id):
    g = d[(d.season == season) & (d.competition_id == competition_id) & d.has_goals]
    pts = defaultdict(float)
    for r in g.itertuples():
        hg, ag = r.home_club_goals, r.away_club_goals
        pts[r.home_club_name] += 3 if hg > ag else 1 if hg == ag else 0
        pts[r.away_club_name] += 3 if ag > hg else 1 if hg == ag else 0
    return pd.Series(pts, name="points").sort_values(ascending=False)


def sanity_check(rs, d, season, min_clubs=8):
    """Rank correlation between model ratings at season end and the real points table."""
    rows = []
    for comp in d[(d.season == season) & d.has_goals].competition_id.unique():
        tab = league_table(d, season, comp)
        if len(tab) < min_clubs:
            continue
        r = rs.ratings(season=season, competition_id=comp)
        r = r[r.club.isin(tab.index)]
        if len(r) < min_clubs:
            continue
        a = r.set_index("club").exp_points.rank()
        b = tab.reindex(a.index).rank()
        rows.append(dict(competition_id=comp, n_clubs=len(r), spearman=float(a.corr(b))))
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)

def run(df, out_dir="out", cfg_overrides=None, verbose=True):
    cfg = {**CFG, **(cfg_overrides or {})}
    os.makedirs(out_dir, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    d = prepare(df, cfg)
    by_season, by_comp = coverage_report(d, cfg)
    by_season.to_csv(f"{out_dir}/coverage_by_season.csv")
    by_comp.to_csv(f"{out_dir}/coverage_by_competition_season.csv", index=False)
    log("\n=== Coverage by season (scored games with both starting XIs) ===")
    log(by_season.to_string())
    good = by_season[by_season.good].index.tolist()
    if len(good) < cfg["val_seasons"] + cfg["test_seasons"] + 1:
        raise ValueError("Too few well-covered seasons; relax coverage_min / min_games_season.")
    log(f"\nGood seasons used for train/val/test: {good}")

    feats, snap_dates, snap_vals = build_features(d, cfg)
    feats, split_seasons = assign_splits(feats, good, cfg)
    log("Split seasons:", {k: list(map(str, v)) for k, v in split_seasons.items()})
    log(feats.split.value_counts().to_string())

    cols = [c for c in feats.columns if c not in META]
    tr, va, te = (feats[feats.split == s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = tr[cols].astype(float), va[cols].astype(float), te[cols].astype(float)
    ytr, yva, yte = tr.y.values, va.y.values, te.y.values

    prior = np.bincount(ytr, minlength=3) / len(ytr)
    base_prior = lambda n: np.tile(prior, (n, 1))
    elo_lr = LogisticRegression(max_iter=1000).fit(tr[["elo_diff"]], ytr)
    res = {"val": {}, "test": {}}
    res["val"]["constant_prior"] = evaluate(base_prior(len(yva)), yva)
    res["test"]["constant_prior"] = evaluate(base_prior(len(yte)), yte)
    res["val"]["elo"] = evaluate(elo_lr.predict_proba(va[["elo_diff"]]), yva)
    res["test"]["elo"] = evaluate(elo_lr.predict_proba(te[["elo_diff"]]), yte)
    res["test"]["always_home_accuracy"] = float((yte == 0).mean())

    club_cols = [c for c in cols if not any(t in c for t in
                 ("lineup", "missing", "unseen", "bench", "joint_diff"))]
    gbm_club = GBM().fit(Xtr[club_cols], ytr, Xva[club_cols], yva)
    res["val"]["gbm_club_only"] = evaluate(gbm_club.predict_proba(Xva[club_cols]), yva)
    gbm = GBM().fit(Xtr, ytr, Xva, yva)
    res["val"]["gbm_full"] = evaluate(gbm.predict_proba(Xva), yva)
    Xtv, ytv = pd.concat([Xtr, Xva]), np.concatenate([ytr, yva])
    final = copy.copy(gbm)
    final.refit(Xtv, ytv)
    club_final = copy.copy(gbm_club)
    club_final.refit(Xtv[club_cols], ytv)
    res["test"]["gbm_club_only"] = evaluate(club_final.predict_proba(Xte[club_cols]), yte)
    Pte = final.predict_proba(Xte)
    res["test"]["gbm_full"] = evaluate(Pte, yte)

    log("\n=== Metrics (lower log_loss / brier / ece is better) ===")
    for split in ("val", "test"):
        log(f"-- {split}")
        log(pd.DataFrame({k: v for k, v in res[split].items() if isinstance(v, dict)}).T.round(4).to_string())
    log(f"Always-home accuracy on test: {res['test']['always_home_accuracy']:.3f}")

    cal = calibration_table(Pte, yte)
    cal.to_csv(f"{out_dir}/calibration_test.csv", index=False)
    plot_calibration(cal, f"{out_dir}/calibration_test.png")

    perm = permutation_importance(gbm, Xva, yva)
    imp = pd.DataFrame({"perm_logloss_increase_val": perm})
    gain = gbm.gain_importance(cols)
    if gain is not None:
        imp["gain_share"] = gain
    imp = imp.sort_values("perm_logloss_increase_val", ascending=False)
    imp.to_csv(f"{out_dir}/feature_importance.csv")
    log("\n=== Feature importance ===")
    log(imp.round(5).to_string())

    comp_default = (float(feats.comp_home_rate.mean()), float(feats.comp_draw_rate.mean()),
                    float(feats.comp_goals_pg.mean()))
    rs = RatingSystem(final, cols, snap_dates, snap_vals, feats[feats.has_lineups], comp_default)
    last_season = split_seasons["test"][-1]
    table = rs.ratings(season=last_season)
    table.to_csv(f"{out_dir}/ranking_{last_season}.csv", index=False)
    log(f"\n=== Top 20 clubs, end of season {last_season} ===")
    log(table.head(20).round(3).to_string())
    sc = sanity_check(rs, d, last_season)
    sc.to_csv(f"{out_dir}/sanity_vs_league_tables.csv", index=False)
    log("\n=== Rank correlation with real points tables ===")
    log(sc.to_string() if len(sc) else "(no competition with enough clubs)")

    try:
        feats.to_parquet(f"{out_dir}/features.parquet", index=False)
    except Exception:
        feats.to_csv(f"{out_dir}/features.csv", index=False)
    with open(f"{out_dir}/rating_system.pkl", "wb") as fh:
        pickle.dump(rs, fh)
    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    return dict(feats=feats, rs=rs, metrics=res, importance=imp, coverage=by_season,
                good_seasons=good, ranking=table, sanity=sc)

def make_synthetic(n_clubs=16, n_seasons=10, seed=0):
    rng = np.random.default_rng(seed)
    clubs = [f"Club{i:02d}" for i in range(n_clubs)]
    strength = rng.normal(0, 0.35, n_clubs)
    squads = {c: [(f"{c}_P{j}", rng.normal(0, 0.15)) for j in range(20)] for c in clubs}
    rows, gid = [], 0
    for s in range(n_seasons):
        start = pd.Timestamp(2010 + s, 8, 1)
        fixtures = [(h, a) for h in range(n_clubs) for a in range(n_clubs) if h != a]
        rng.shuffle(fixtures)
        for i, (h, a) in enumerate(fixtures):
            date = start + pd.Timedelta(days=int(i // (n_clubs // 2)) * 4)
            sides = {}
            for side, k in (("home", h), ("away", a)):
                sq = squads[clubs[k]]
                idx = rng.choice(20, 11, replace=False, p=np.linspace(2, 1, 20) / np.linspace(2, 1, 20).sum())
                st = [sq[j][0] for j in idx]
                ab = sum(sq[j][1] for j in idx)
                sides[side] = (st, [sq[j][0] for j in range(20) if j not in idx][:7], ab)
            lam_h = math.exp(0.3 + 0.25 + strength[h] - strength[a] + sides["home"][2] - sides["away"][2])
            lam_a = math.exp(0.3 + strength[a] - strength[h] + sides["away"][2] - sides["home"][2])
            lineups_ok = s >= 2
            rows.append(dict(game_id=gid, date=date, competition_id="L1", season=2010 + s,
                             home_club_name=clubs[h], away_club_name=clubs[a],
                             home_club_goals=rng.poisson(lam_h), away_club_goals=rng.poisson(lam_a),
                             home_starters=sides["home"][0] if lineups_ok else [],
                             home_substitutes=sides["home"][1] if lineups_ok else [],
                             away_starters=sides["away"][0] if lineups_ok else [],
                             away_substitutes=sides["away"][1] if lineups_ok else []))
            gid += 1
    df = pd.DataFrame(rows)
    df["home_players"] = df.home_starters + df.home_substitutes
    df["away_players"] = df.away_starters + df.away_substitutes
    return df


if __name__ == "__main__":
    import duckdb

    con = duckdb.connect("transfermarkt-datasets.duckdb", read_only=True)

    query = """
    WITH lineups AS (
        SELECT
            game_id,
            club_id,
            list(player_name) FILTER (WHERE type = 'starting_lineup') AS starters,
            list(player_name) FILTER (WHERE type = 'substitutes')     AS substitutes,
            list(player_name)                                         AS all_players
        FROM game_lineups
        GROUP BY game_id, club_id
    )
    SELECT
        g.game_id, g.date, g.competition_id, g.season,
        g.home_club_name, g.away_club_name,
        g.home_club_goals, g.away_club_goals,
        hl.starters    AS home_starters,
        hl.substitutes AS home_substitutes,
        hl.all_players AS home_players,
        al.starters    AS away_starters,
        al.substitutes AS away_substitutes,
        al.all_players AS away_players
    FROM games g
    LEFT JOIN lineups hl ON hl.game_id = g.game_id AND hl.club_id = g.home_club_id
    LEFT JOIN lineups al ON al.game_id = g.game_id AND al.club_id = g.away_club_id
    ORDER BY g.date
    """

    df = con.sql(query).df()
    con.close()

    res = run(df, out_dir="out")
    print(res["ranking"].head(20))