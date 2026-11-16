import argparse
import ast
import bisect
import copy
import inspect
import json
import math
import os
import pickle
from collections import Counter, defaultdict, deque

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
    k_club=18.0,
    k_player=0.45,
    player_decay=0.008,
    player_cap=80.0,
    form_n=10,
    usual_n=10,
    min_starters=11,
    coverage_min=0.90,
    min_games_season=300,
    ema_alpha=0.2,
    comp_prior_n=50.0,
    val_seasons=1,
    test_seasons=1,
    season_revert=0.15,
    offseason_days=45,
)

META = [
    "game_id", "date", "season", "f_year", "competition_id", "home", "away",
    "home_name", "away_name", "home_goals", "away_goals", "gd", "y", "has_lineups", "split"
]

LINEUP_TOKENS = (
    "lineup", "missing", "unseen", "bench", "joint", "xi_cont", "prev_overlap"
)


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
    by_season = (
        g.groupby("season")
        .agg(
            n_games=("game_id", "size"),
            lineup_share=("has_lineups", "mean"),
            first=("date", "min"),
            median_date=("date", "median"),
            last=("date", "max"),
        )
        .sort_values("median_date")
    )
    by_season["good"] = (
        (by_season["lineup_share"] >= cfg["coverage_min"])
        & (by_season["n_games"] >= cfg["min_games_season"])
    )
    by_comp = (
        g.groupby(["season", "competition_id"])
        .agg(n_games=("game_id", "size"), lineup_share=("has_lineups", "mean"))
        .reset_index()
    )
    return by_season, by_comp


def _exp(diff):
    return 1.0 / (1.0 + 10.0 ** (-np.clip(diff, -1200.0, 1200.0) / 400.0))


def _mov(gd):
    agd = abs(gd)
    if agd <= 1:
        return 1.0
    return 1.0 + 0.5 * math.log1p(agd - 1)


def pair_features(h, a, comp, hfa=CFG["hfa"]):
    """Feature row (home minus away where sensible) from two club snapshots."""
    f = {}
    f["elo_diff"] = h["elo_p"] - a["elo_p"]
    f["abs_elo_diff"] = abs(f["elo_diff"])
    f["elo_exp"] = _exp(f["elo_diff"] + hfa)
    f["elo_mean"] = 0.5 * (h["elo_p"] + a["elo_p"])
    f["elo_mom_diff"] = h["elo_mom"] - a["elo_mom"]
    f["comp_elo_diff"] = h["comp_elo"] - a["comp_elo"]

    f["joint_club_diff"] = h["elo_j"] - a["elo_j"]
    f["joint_diff"] = (h["elo_j"] + h["adj"]) - (a["elo_j"] + a["adj"])
    f["abs_joint_diff"] = abs(f["joint_diff"])
    f["joint_exp"] = _exp(f["joint_diff"] + hfa)

    f["lineup_adj_diff"] = h["adj"] - a["adj"]
    f["lineup_vs_usual_h"] = h["adj"] - h["adj_usual"]
    f["lineup_vs_usual_a"] = a["adj"] - a["adj_usual"]
    f["lineup_vs_usual_diff"] = f["lineup_vs_usual_h"] - f["lineup_vs_usual_a"]

    f["xi_cont_h"] = h["xi_cont"]
    f["xi_cont_a"] = a["xi_cont"]
    f["xi_cont_diff"] = h["xi_cont"] - a["xi_cont"]
    f["prev_overlap_h"] = h["prev_overlap"]
    f["prev_overlap_a"] = a["prev_overlap"]

    f["missing_h"] = h["missing"]
    f["missing_a"] = a["missing"]
    f["missing_diff"] = h["missing"] - a["missing"]
    f["unseen_h"] = h["unseen"]
    f["unseen_a"] = a["unseen"]
    f["bench_diff"] = h["bench"] - a["bench"]

    f["ppg_diff"] = h["ppg"] - a["ppg"]
    f["gdf_diff"] = h["gdf"] - a["gdf"]
    f["gf_diff"] = h["gf_ema"] - a["gf_ema"]
    f["ga_diff"] = h["ga_ema"] - a["ga_ema"]

    f["rest_h"] = h["rest"]
    f["rest_a"] = a["rest"]
    f["rest_diff"] = h["rest"] - a["rest"]
    f["short_rest_h"] = float(h["rest"] <= 4.0) if not np.isnan(h["rest"]) else np.nan
    f["short_rest_a"] = float(a["rest"] <= 4.0) if not np.isnan(a["rest"]) else np.nan
    f["n_min"] = min(h["n"], a["n"])

    f["comp_home_rate"] = comp[0]
    f["comp_draw_rate"] = comp[1]
    f["comp_goals_pg"] = comp[2]
    return f


class _State:
    """All time-evolving quantities. Snapshots are taken BEFORE a date's matches are applied."""

    def __init__(self, cfg):
        self.c = cfg
        self.elo_p = {}
        self.elo_j = {}
        self.elo_ema = {}
        self.player_r = defaultdict(float)
        self.player_seen = defaultdict(int)
        self.player_last_date = {}
        self.form = defaultdict(lambda: deque(maxlen=cfg["form_n"]))
        self.gf_ema = {}
        self.ga_ema = {}
        self.n_games = defaultdict(int)
        self.last_date = {}
        self.prev_xi = {}
        self.start_hist = defaultdict(deque)
        self.start_cnt = defaultdict(Counter)
        self.adj_usual = {}
        self.bench_ema = {}
        self.comp = defaultdict(lambda: [0, 0, 0, 0.0])
        self.comp_elo = defaultdict(lambda: cfg["elo_init"])
        self.club_primary_comp = {}

    def _init_club_if_needed(self, club, comp_id):
        if club in self.elo_p:
            return
        if self.comp[comp_id][0] >= 20:
            init_elo = self.comp_elo[comp_id] - 50.0
        else:
            init_elo = self.c["elo_init"]
        self.elo_p[club] = init_elo
        self.elo_j[club] = init_elo
        self.elo_ema[club] = init_elo

    def comp_stats(self, comp):
        n, hw, dr, gl = self.comp[comp]
        k = self.c["comp_prior_n"]
        return (
            (hw + k * 0.45) / (n + k),
            (dr + k * 0.25) / (n + k),
            (gl + k * 2.7) / (n + k),
        )

    def usual_xi(self, club):
        return {p for p, _ in self.start_cnt[club].most_common(11)}

    def _effective_player_r(self, p, club_per_player_prior):
        seen = self.player_seen[p]
        if seen == 0:
            return 0.85 * club_per_player_prior
        w = min(seen, 8) / 8.0
        return w * self.player_r[p] + (1.0 - w) * (0.85 * club_per_player_prior)

    def snap(self, club, starters, subs, date, comp_id):
        """starters=None -> 'typical' snapshot (usual lineup), used for ratings / missing lineups."""
        self._init_club_if_needed(club, comp_id)
        f = self.form[club]
        p_comp = self.club_primary_comp.get(club, comp_id)
        adj_u = self.adj_usual.get(club, 0.0)
        club_per_player = adj_u / 11.0

        s = dict(
            elo_p=self.elo_p[club],
            elo_j=self.elo_j[club],
            elo_mom=self.elo_p[club] - self.elo_ema[club],
            comp_elo=self.comp_elo[p_comp],
            ppg=float(np.mean([x[0] for x in f])) if f else np.nan,
            gdf=float(np.mean([x[1] for x in f])) if f else np.nan,
            gf_ema=self.gf_ema.get(club, np.nan),
            ga_ema=self.ga_ema.get(club, np.nan),
            n=min(self.n_games[club], 20),
            rest=float(min((date - self.last_date[club]).days, 30)) if club in self.last_date else np.nan,
            adj_usual=adj_u,
        )
        if starters is None:
            s.update(
                adj=adj_u,
                missing=0.0,
                unseen=0.0,
                xi_cont=1.0,
                prev_overlap=1.0,
                bench=self.bench_ema.get(club, np.nan),
            )
        else:
            s["adj"] = float(sum(self._effective_player_r(p, club_per_player) for p in starters))
            cnt = self.start_cnt[club]
            if self.start_hist[club]:
                s["missing"] = float(11 - len(set(starters) & self.usual_xi(club)))
                top11_sum = sum(sorted(cnt.values(), reverse=True)[:11])
                s["xi_cont"] = float(sum(cnt[p] for p in starters) / max(top11_sum, 1))
            else:
                s["missing"] = np.nan
                s["xi_cont"] = np.nan
            s["prev_overlap"] = (
                float(len(set(starters) & self.prev_xi[club]) / 11.0)
                if club in self.prev_xi
                else np.nan
            )
            s["unseen"] = float(sum(1 for p in starters if self.player_seen[p] == 0))
            s["bench"] = (
                float(np.mean([self._effective_player_r(p, club_per_player) for p in subs]))
                if subs
                else np.nan
            )
        return s

    def update(self, row, hs, as_):
        c = self.c
        h, a = row["home"], row["away"]
        hg, ag = row["home_goals"], row["away_goals"]
        gd = hg - ag
        S = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        m = _mov(gd)

        for club in (h, a):
            if club in self.last_date and (row["date"] - self.last_date[club]).days >= c["offseason_days"]:
                anchor_comp = self.club_primary_comp.get(club, row["competition_id"])
                anchor = 0.15 * c["elo_init"] + 0.85 * self.comp_elo[anchor_comp]
                rev = c["season_revert"]
                self.elo_p[club] = (1.0 - rev) * self.elo_p[club] + rev * anchor
                self.elo_j[club] = (1.0 - rev) * self.elo_j[club] + rev * anchor
                self.elo_ema[club] = (1.0 - rev) * self.elo_ema[club] + rev * anchor

        if row["has_lineups"]:
            for p in row["home_starters"] + row["away_starters"]:
                if p in self.player_last_date and (row["date"] - self.player_last_date[p]).days >= c["offseason_days"]:
                    self.player_r[p] *= (1.0 - c["season_revert"])

        avg_match_elo = 0.5 * (self.elo_p[h] + self.elo_p[a])
        self.comp_elo[row["competition_id"]] = (
            0.98 * self.comp_elo[row["competition_id"]] + 0.02 * avg_match_elo
        )
        if str(row["competition_id"]).endswith("1"):
            self.club_primary_comp[h] = row["competition_id"]
            self.club_primary_comp[a] = row["competition_id"]

        res_p = S - _exp(self.elo_p[h] + c["hfa"] - self.elo_p[a])
        self.elo_p[h] += c["k_club"] * m * res_p
        self.elo_p[a] -= c["k_club"] * m * res_p

        res_j = S - _exp(self.elo_j[h] + c["hfa"] + hs["adj"] - self.elo_j[a] - as_["adj"])
        self.elo_j[h] += c["k_club"] * m * res_j
        self.elo_j[a] -= c["k_club"] * m * res_j

        if row["has_lineups"]:
            for side, club, sign in (("home", h, 1.0), ("away", a, -1.0)):
                st = row[side + "_starters"]
                for p in st:
                    r = self.player_r[p] * (1.0 - c["player_decay"]) + sign * c["k_player"] * m * res_j
                    self.player_r[p] = float(np.clip(r, -c["player_cap"], c["player_cap"]))
                    self.player_seen[p] += 1
                    self.player_last_date[p] = row["date"]
                snap = hs if side == "home" else as_
                self.prev_xi[club] = set(st)
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
                self.adj_usual[club] = al * snap["adj"] + (1.0 - al) * prev
                if not np.isnan(snap["bench"]):
                    pb = self.bench_ema.get(club, snap["bench"])
                    self.bench_ema[club] = al * snap["bench"] + (1.0 - al) * pb

        al = c["ema_alpha"]
        for club, gf, ga in ((h, hg, ag), (a, ag, hg)):
            pts = 3 if gf > ga else 1 if gf == ga else 0
            self.form[club].append((pts, gf - ga))
            self.gf_ema[club] = al * gf + (1.0 - al) * self.gf_ema.get(club, float(gf))
            self.ga_ema[club] = al * ga + (1.0 - al) * self.ga_ema.get(club, float(ga))
            self.elo_ema[club] = al * self.elo_p[club] + (1.0 - al) * self.elo_ema[club]
            self.n_games[club] += 1
            self.last_date[club] = row["date"]

        cs = self.comp[row["competition_id"]]
        cs[0] += 1
        cs[1] += int(gd > 0)
        cs[2] += int(gd == 0)
        cs[3] += hg + ag


def build_features(d, cfg=CFG):
    """One chronological pass. Returns feature table, per-club snapshot history, club_names map."""
    st = _State(cfg)
    g = d[d["has_goals"]].copy()
    if "home_club_id" not in g.columns:
        g["home_club_id"] = g["home_club_name"]
        g["away_club_id"] = g["away_club_name"]

    g = g.rename(columns={
        "home_club_id": "home",
        "away_club_id": "away",
        "home_club_goals": "home_goals",
        "away_club_goals": "away_goals",
    })
    club_names = {}
    snap_dates, snap_vals = defaultdict(list), defaultdict(list)
    rows = []

    for date, grp in g.groupby("date", sort=True):
        pending = []
        for r in grp.to_dict("records"):
            club_names[r["home"]] = r["home_club_name"]
            club_names[r["away"]] = r["away_club_name"]
            if r["has_lineups"]:
                hs = st.snap(r["home"], r["home_starters"], r["home_substitutes"], date, r["competition_id"])
                as_ = st.snap(r["away"], r["away_starters"], r["away_substitutes"], date, r["competition_id"])
            else:
                hs = st.snap(r["home"], None, None, date, r["competition_id"])
                as_ = st.snap(r["away"], None, None, date, r["competition_id"])
            f = pair_features(hs, as_, st.comp_stats(r["competition_id"]), hfa=cfg["hfa"])
            gd = r["home_goals"] - r["away_goals"]
            f.update(
                game_id=r["game_id"],
                date=date,
                season=r["season"],
                competition_id=r["competition_id"],
                home=r["home"],
                away=r["away"],
                home_name=r["home_club_name"],
                away_name=r["away_club_name"],
                home_goals=r["home_goals"],
                away_goals=r["away_goals"],
                gd=gd,
                y=0 if gd > 0 else 1 if gd == 0 else 2,
                has_lineups=r["has_lineups"],
            )
            rows.append(f)
            pending.append((r, hs, as_))

        for r, hs, as_ in pending:
            st.update(r, hs, as_)

        for r, _, _ in pending:
            for club in (r["home"], r["away"]):
                if snap_dates[club] and snap_dates[club][-1] == date:
                    continue
                s = st.snap(club, None, None, date, r["competition_id"])
                s["rest"] = 7.0
                snap_dates[club].append(date)
                snap_vals[club].append(s)

    feats = pd.DataFrame(rows)
    return feats, snap_dates, snap_vals, club_names


def assign_splits(feats, good_seasons, cfg=CFG):
    feats = feats.copy()
    f_year = np.where(feats["date"].dt.month >= 7, feats["date"].dt.year, feats["date"].dt.year - 1)
    feats["f_year"] = f_year

    ok = feats["has_lineups"] & feats["season"].isin(good_seasons)
    year_counts = feats.loc[ok, "f_year"].value_counts()
    valid_years = sorted(year_counts[year_counts >= cfg["min_games_season"]].index)

    n_val = cfg["val_seasons"]
    n_test = cfg["test_seasons"]
    if len(valid_years) < n_val + n_test + 1:
        raise ValueError("Not enough full season years for train/val/test split.")

    test_years = set(valid_years[-n_test:])
    val_years = set(valid_years[-(n_val + n_test):-n_test])
    train_years = set(valid_years[:-(n_val + n_test)])

    feats["split"] = "unused"
    feats.loc[ok & feats["f_year"].isin(train_years), "split"] = "train"
    feats.loc[ok & feats["f_year"].isin(val_years), "split"] = "val"
    feats.loc[ok & feats["f_year"].isin(test_years), "split"] = "test"

    splits_info = {
        "train": sorted(int(y) for y in train_years),
        "val": sorted(int(y) for y in val_years),
        "test": sorted(int(y) for y in test_years),
    }
    return feats, splits_info


class GBM:
    def __init__(self):
        self.best_n, self.m = None, None

    def _new(self, n):
        if HAVE_LGB:
            return lgb.LGBMClassifier(
                objective="multiclass",
                n_estimators=n,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                verbose=-1,
                random_state=0,
            )
        return HistGradientBoostingClassifier(
            max_iter=n,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=0,
        )

    def fit(self, X, y, Xv, yv, sample_weight=None):
        if HAVE_LGB:
            m = self._new(2000)
            fit_params = inspect.signature(m.fit).parameters
            if "eval_X" in fit_params and "eval_y" in fit_params:
                m.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_X=Xv, eval_y=yv,
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
            else:
                m.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_set=[(Xv, yv)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
            self.best_n = max(int(m.best_iteration_ or 20), 20)
        else:
            m = self._new(300)
            m.fit(X, y, sample_weight=sample_weight)
            losses = [log_loss(yv, p, labels=[0, 1, 2]) for p in m.staged_predict_proba(Xv)]
            self.best_n = int(np.argmin(losses)) + 1
            m = self._new(self.best_n)
            m.fit(X, y, sample_weight=sample_weight)
        self.m = m
        return self

    def refit(self, X, y, sample_weight=None):
        m = self._new(int(self.best_n * 1.15) + 1)
        m.fit(X, y, sample_weight=sample_weight)
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
        e = sum(
            abs(p[idx == b].mean() - o[idx == b].mean()) * (idx == b).mean()
            for b in range(bins)
            if (idx == b).any()
        )
        out.append(e)
    return float(np.mean(out))


def evaluate(P, y):
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(1, keepdims=True)
    draw = (y == 1)
    return dict(
        n=int(len(y)),
        log_loss=float(log_loss(y, P, labels=[0, 1, 2])),
        brier=brier(P, y),
        accuracy=float((P.argmax(1) == y).mean()),
        ece=ece(P, y),
        draw_mean_pred=float(P[:, 1].mean()),
        draw_actual=float(draw.mean()),
        draw_auc=float(roc_auc_score(draw, P[:, 1])) if 0 < draw.sum() < len(y) else np.nan,
    )


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
                rows.append(
                    dict(
                        cls=name,
                        bin=b,
                        n=int(m.sum()),
                        mean_pred=float(P[m, k].mean()),
                        freq=float((y[m] == k).mean()),
                    )
                )
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


def _recency_weights(years, half_life=3.0):
    arr = np.asarray(years, dtype=float)
    return np.power(0.5, (arr.max() - arr) / half_life)


def compare_training_windows(tr, va, te, cols, club_cols, train_years):
    """Train & evaluate Elo, Club-GBM, and Full-GBM across progressively smaller training windows."""
    Xva, yva = va[cols].astype(float), va.y.values
    Xte, yte = te[cols].astype(float), te.y.values

    T = len(train_years)
    variants = [(f"all_{T}y", train_years, False), (f"all_{T}y_decay", train_years, True)]
    for k in (6, 3, 1):
        if T > k:
            variants.append((f"last_{k}y", train_years[-k:], False))

    records = []
    test_probs = {}
    artifacts = {}

    for name, yrs, use_decay in variants:
        sub_tr = tr[tr["f_year"].isin(yrs)]
        Xtr_sub = sub_tr[cols].astype(float)
        ytr_sub = sub_tr.y.values
        w_tr = _recency_weights(sub_tr["f_year"].values) if use_decay else None

        sub_tv = pd.concat([sub_tr, va])
        Xtv_sub = sub_tv[cols].astype(float)
        ytv_sub = sub_tv.y.values
        w_tv = _recency_weights(sub_tv["f_year"].values) if use_decay else None

        elo_lr = LogisticRegression(max_iter=1000).fit(
            sub_tr[["elo_diff"]], ytr_sub, sample_weight=w_tr
        )
        elo_lr_tv = LogisticRegression(max_iter=1000).fit(
            sub_tv[["elo_diff"]], ytv_sub, sample_weight=w_tv
        )

        gbm_club = GBM().fit(
            Xtr_sub[club_cols], ytr_sub, Xva[club_cols], yva, sample_weight=w_tr
        )
        gbm_full = GBM().fit(
            Xtr_sub, ytr_sub, Xva, yva, sample_weight=w_tr
        )

        club_final = copy.copy(gbm_club).refit(Xtv_sub[club_cols], ytv_sub, sample_weight=w_tv)
        full_final = copy.copy(gbm_full).refit(Xtv_sub, ytv_sub, sample_weight=w_tv)

        m_val_elo = evaluate(elo_lr.predict_proba(va[["elo_diff"]]), yva)
        m_val_club = evaluate(gbm_club.predict_proba(Xva[club_cols]), yva)
        m_val_full = evaluate(gbm_full.predict_proba(Xva), yva)

        p_te_elo = elo_lr_tv.predict_proba(te[["elo_diff"]])
        p_te_club = club_final.predict_proba(Xte[club_cols])
        p_te_full = full_final.predict_proba(Xte)

        m_te_elo = evaluate(p_te_elo, yte)
        m_te_club = evaluate(p_te_club, yte)
        m_te_full = evaluate(p_te_full, yte)

        test_probs[name] = p_te_full
        artifacts[name] = dict(
            gbm_val=gbm_full,
            gbm_club_val=gbm_club,
            final=full_final,
            club_final=club_final,
            elo_lr=elo_lr,
        )

        records.append(dict(
            variant=name,
            train_years=f"{min(yrs)}-{max(yrs)}",
            n_train=int(len(sub_tr)),
            n_refit=int(len(sub_tv)),
            best_n_full=int(gbm_full.best_n),
            val_elo_logloss=m_val_elo["log_loss"],
            val_club_logloss=m_val_club["log_loss"],
            val_full_logloss=m_val_full["log_loss"],
            test_elo_logloss=m_te_elo["log_loss"],
            test_club_logloss=m_te_club["log_loss"],
            test_full_logloss=m_te_full["log_loss"],
            test_elo_brier=m_te_elo["brier"],
            test_club_brier=m_te_club["brier"],
            test_full_brier=m_te_full["brier"],
            test_elo_acc=m_te_elo["accuracy"],
            test_club_acc=m_te_club["accuracy"],
            test_full_acc=m_te_full["accuracy"],
            test_full_ece=m_te_full["ece"],
            test_full_draw_auc=m_te_full["draw_auc"],
        ))

    comp_df = pd.DataFrame(records)
    return comp_df, test_probs, artifacts


def plot_window_comparison(comp_df, always_home_acc, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [f"{r.variant}\n({r.n_refit:,}g)" for r in comp_df.itertuples()]
    x = np.arange(len(comp_df))
    w = 0.26

    ax = axes[0, 0]
    b1 = ax.bar(x - w, comp_df["test_elo_logloss"], w, label="Elo LR", color="#7f8c8d")
    b2 = ax.bar(x, comp_df["test_club_logloss"], w, label="GBM Club Only", color="#3498db")
    b3 = ax.bar(x + w, comp_df["test_full_logloss"], w, label="GBM Full (Club+Lineup)", color="#2ecc71")
    vals = pd.concat([comp_df["test_elo_logloss"], comp_df["test_club_logloss"], comp_df["test_full_logloss"]])
    pad = max((vals.max() - vals.min()) * 0.25, 0.002)
    ax.set_ylim(vals.min() - pad, vals.max() + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Log-Loss (lower is better)")
    ax.set_title("Test Log-Loss by Training Window & Model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax = axes[0, 1]
    b1 = ax.bar(x - w, comp_df["test_elo_acc"] * 100, w, label="Elo LR", color="#7f8c8d")
    b2 = ax.bar(x, comp_df["test_club_acc"] * 100, w, label="GBM Club Only", color="#3498db")
    b3 = ax.bar(x + w, comp_df["test_full_acc"] * 100, w, label="GBM Full (Club+Lineup)", color="#2ecc71")
    ax.axhline(always_home_acc * 100, color="#e74c3c", ls="--", lw=1.2, label=f"Always-Home ({always_home_acc*100:.1f}%)")
    avals = pd.concat([comp_df["test_elo_acc"], comp_df["test_club_acc"], comp_df["test_full_acc"]]) * 100
    ax.set_ylim(min(avals.min(), always_home_acc * 100) - 1.5, avals.max() + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy % (higher is better)")
    ax.set_title("Test Accuracy by Training Window & Model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}%", ha="center", va="bottom", fontsize=7, rotation=90)

    ax = axes[1, 0]
    b1 = ax.bar(x - w, comp_df["test_elo_brier"], w, label="Elo LR", color="#7f8c8d")
    b2 = ax.bar(x, comp_df["test_club_brier"], w, label="GBM Club Only", color="#3498db")
    b3 = ax.bar(x + w, comp_df["test_full_brier"], w, label="GBM Full (Club+Lineup)", color="#2ecc71")
    bvals = pd.concat([comp_df["test_elo_brier"], comp_df["test_club_brier"], comp_df["test_full_brier"]])
    bpad = max((bvals.max() - bvals.min()) * 0.25, 0.001)
    ax.set_ylim(bvals.min() - bpad, bvals.max() + bpad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("Test Brier Score by Training Window & Model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax = axes[1, 1]
    unweighted = comp_df[~comp_df["variant"].str.contains("decay")].sort_values("n_refit")
    weighted = comp_df[comp_df["variant"].str.contains("decay")]

    ax.plot(unweighted["n_refit"], unweighted["test_elo_logloss"], "o--", color="#7f8c8d", label="Elo LR (unweighted)")
    ax.plot(unweighted["n_refit"], unweighted["test_club_logloss"], "s-", color="#3498db", lw=2, label="GBM Club Only")
    ax.plot(unweighted["n_refit"], unweighted["test_full_logloss"], "^-", color="#2ecc71", lw=2, label="GBM Full")
    if len(weighted):
        ax.scatter(weighted["n_refit"], weighted["test_full_logloss"], s=140, marker="*", color="#e67e22", zorder=5,
                   label=f"GBM Full ({weighted.iloc[0]['variant']})")
        ax.scatter(weighted["n_refit"], weighted["test_club_logloss"], s=90, marker="D", color="#9b59b6", zorder=5,
                   label=f"GBM Club ({weighted.iloc[0]['variant']})")
    for r in unweighted.itertuples():
        ax.annotate(r.variant, (r.n_refit, r.test_full_logloss), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xlabel("Number of Refit Training Games (Train + Val)")
    ax.set_ylabel("Test Log-Loss (lower is better)")
    ax.set_title("Learning Curve: Training Sample Size vs. Test Log-Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Training Window Comparison on Held-Out Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_window_calibration(test_probs, yte, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, cls_name in zip(axes, ["home", "draw", "away"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        for name, probs in test_probs.items():
            cal = calibration_table(probs, yte)
            c = cal[cal.cls == cls_name]
            ax.plot(c.mean_pred, c.freq, "o-", lw=1.5, markersize=4, label=name)
        ax.set_title(f"Calibration: {cls_name}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


class RatingSystem:
    def __init__(self, model, feature_cols, snap_dates, snap_vals, feats, comp_default,
                 club_names=None, hfa=CFG["hfa"], comp_map=None):
        self.model, self.cols = model, feature_cols
        self.snap_dates, self.snap_vals = dict(snap_dates), dict(snap_vals)
        self.comp_default = comp_default
        self.club_names = club_names or {}
        self.hfa = hfa
        self.comp_map = comp_map or {}
        self.season_end = feats.groupby("season").date.max().to_dict()
        self.season_clubs = {}
        for (s, c), g in feats.groupby(["season", "competition_id"]):
            self.season_clubs[(s, c)] = set(g.home) | set(g.away)
        self.last_date = feats.date.max()

    def _snap_at(self, club, as_of):
        dates = self.snap_dates.get(club)
        if not dates:
            return None
        i = bisect.bisect_left(dates, as_of) - 1
        return dict(self.snap_vals[club][i]) if i >= 0 else None

    def ratings(self, as_of=None, season=None, competition_id=None, active_days=365, min_clubs=2, min_games=10, n_bench=15):
        if season is not None:
            as_of = self.season_end[season] + pd.Timedelta(days=1)
            clubs = set()
            for (s, c), cl in self.season_clubs.items():
                if s == season and (competition_id is None or c == competition_id):
                    clubs |= cl
            eff_min_games = 1
        else:
            as_of = pd.Timestamp(as_of) if as_of is not None else self.last_date + pd.Timedelta(days=1)
            clubs = {
                c for c, ds in self.snap_dates.items()
                if ds and ds[0] < as_of and (as_of - max(x for x in ds if x < as_of)).days <= active_days
            }
            if competition_id is not None:
                keep = set()
                for (s, c), cl in self.season_clubs.items():
                    if c == competition_id:
                        keep |= cl
                clubs &= keep
            eff_min_games = min_games

        snaps = {c: self._snap_at(c, as_of) for c in clubs if c in self.snap_dates}
        snaps = {c: s for c, s in snaps.items() if s is not None and s["n"] >= eff_min_games}
        if len(snaps) < min_clubs:
            return pd.DataFrame()

        ordered = sorted(snaps.values(), key=lambda s: s["elo_p"])
        b_idx = np.unique(np.linspace(0, len(ordered) - 1, min(n_bench, len(ordered))).astype(int))
        benchmarks = [ordered[i] for i in b_idx]

        comp = self.comp_default
        names = list(snaps)
        n_b = len(benchmarks)
        rows_h = [pair_features(snaps[c], b, comp, hfa=self.hfa) for c in names for b in benchmarks]
        rows_a = [pair_features(b, snaps[c], comp, hfa=self.hfa) for c in names for b in benchmarks]
        Xh = pd.DataFrame(rows_h)[self.cols]
        Xa = pd.DataFrame(rows_a)[self.cols]

        Ph = self.model.predict_proba(Xh).reshape(len(names), n_b, 3).mean(axis=1)
        Pa = self.model.predict_proba(Xa).reshape(len(names), n_b, 3).mean(axis=1)

        pts = 0.5 * ((3.0 * Ph[:, 0] + Ph[:, 1]) + (3.0 * Pa[:, 2] + Pa[:, 1]))
        win = 0.5 * (Ph[:, 0] + Pa[:, 2])
        draw = 0.5 * (Ph[:, 1] + Pa[:, 1])

        out = pd.DataFrame(dict(
            club_id=names,
            club=[self.club_names.get(c, str(c)) for c in names],
            exp_points=pts,
            win_prob=win,
            draw_prob=draw,
            elo=[snaps[c]["elo_p"] for c in names],
            joint_rating=[snaps[c]["elo_j"] + snaps[c]["adj_usual"] for c in names],
            usual_lineup_adj=[snaps[c]["adj_usual"] for c in names],
        ))
        out = out.sort_values("exp_points", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        out.attrs["as_of"] = as_of
        return out


def league_table(d, season, competition_id):
    g = d[(d.season == season) & (d.competition_id == competition_id) & d.has_goals]
    id_col_h = "home_club_id" if "home_club_id" in g.columns else "home_club_name"
    id_col_a = "away_club_id" if "away_club_id" in g.columns else "away_club_name"
    pts = defaultdict(float)
    for r in g.itertuples():
        hg, ag = r.home_club_goals, r.away_club_goals
        pts[getattr(r, id_col_h)] += 3 if hg > ag else 1 if hg == ag else 0
        pts[getattr(r, id_col_a)] += 3 if ag > hg else 1 if hg == ag else 0
    return pd.Series(pts, name="points").sort_values(ascending=False)


def _is_league_competition(g_comp, min_clubs=8):
    id_h = "home_club_id" if "home_club_id" in g_comp.columns else "home_club_name"
    id_a = "away_club_id" if "away_club_id" in g_comp.columns else "away_club_name"
    counts = Counter(g_comp[id_h].tolist() + g_comp[id_a].tolist())
    if len(counts) < min_clubs:
        return False
    vals = np.array(list(counts.values()), dtype=float)
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    return med >= 10.0 and (q75 - q25) <= 4.0


def sanity_check(rs, d, season, min_clubs=8):
    rows = []
    sub = d[(d.season == season) & d.has_goals]
    for comp, g_comp in sub.groupby("competition_id"):
        if not _is_league_competition(g_comp, min_clubs=min_clubs):
            continue
        tab = league_table(d, season, comp)
        if len(tab) < min_clubs:
            continue
        r = rs.ratings(season=season, competition_id=comp)
        if r.empty:
            continue
        r = r[r.club_id.isin(tab.index)]
        if len(r) < min_clubs:
            continue
        a = r.set_index("club_id").exp_points.rank()
        b = tab.reindex(a.index).rank()
        rows.append(dict(competition_id=comp, n_clubs=len(r), spearman=float(a.corr(b))))
    if not rows:
        return pd.DataFrame(columns=["competition_id", "n_clubs", "spearman"])
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def predict_year(d, feats, final, rs, cols, comp_default, cfg, year=2026):
    """Generate pre-match predictions for all played and upcoming games in a calendar year."""
    lbl = {0: "H", 1: "D", 2: "A"}
    rows = []

    f_yr = feats[feats["date"].dt.year == year].copy()
    if len(f_yr):
        P = final.predict_proba(f_yr[cols].astype(float))
        pred_cls = P.argmax(axis=1)
        for i, r in enumerate(f_yr.itertuples()):
            rows.append(dict(
                date=r.date.strftime("%Y-%m-%d"),
                competition=r.competition_id,
                home=r.home_name,
                away=r.away_name,
                p_home=round(float(P[i, 0]), 3),
                p_draw=round(float(P[i, 1]), 3),
                p_away=round(float(P[i, 2]), 3),
                pred=lbl[int(pred_cls[i])],
                score=f"{int(r.home_goals)}-{int(r.away_goals)}",
                actual=lbl[int(r.y)],
                correct=bool(pred_cls[i] == r.y),
                status="played",
            ))

    up = d[(~d["has_goals"]) & (d["date"].dt.year == year)].copy()
    if len(up):
        id_h = "home_club_id" if "home_club_id" in up.columns else "home_club_name"
        id_a = "away_club_id" if "away_club_id" in up.columns else "away_club_name"

        up_feats, up_meta = [], []
        for r in up.to_dict("records"):
            hs = rs._snap_at(r[id_h], r["date"])
            as_ = rs._snap_at(r[id_a], r["date"])
            if hs is None or as_ is None:
                continue
            comp = rs.comp_map.get(r["competition_id"], comp_default)
            up_feats.append(pair_features(hs, as_, comp, hfa=cfg["hfa"]))
            up_meta.append(r)

        if up_feats:
            P_up = final.predict_proba(pd.DataFrame(up_feats)[cols].astype(float))
            pred_up = P_up.argmax(axis=1)
            for i, r in enumerate(up_meta):
                rows.append(dict(
                    date=r["date"].strftime("%Y-%m-%d"),
                    competition=r["competition_id"],
                    home=r["home_club_name"],
                    away=r["away_club_name"],
                    p_home=round(float(P_up[i, 0]), 3),
                    p_draw=round(float(P_up[i, 1]), 3),
                    p_away=round(float(P_up[i, 2]), 3),
                    pred=lbl[int(pred_up[i])],
                    score="TBD",
                    actual="-",
                    correct=None,
                    status="upcoming",
                ))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "competition", "home"]).reset_index(drop=True)


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

    feats, snap_dates, snap_vals, club_names = build_features(d, cfg)
    feats, split_seasons = assign_splits(feats, good, cfg)
    log("Split seasons:", {k: list(map(str, v)) for k, v in split_seasons.items()})
    log(feats.split.value_counts().to_string())
    unused = feats[feats.split == "unused"]
    if len(unused):
        log(
            f"Unused breakdown -> missing_lineups: {int((~unused.has_lineups).sum())}, "
            f"bad_season_or_partial_fyear: {int(unused.has_lineups.sum())}"
        )

    cols = [c for c in feats.columns if c not in META]
    club_cols = [c for c in cols if not any(t in c for t in LINEUP_TOKENS)]
    tr, va, te = (feats[feats.split == s] for s in ("train", "val", "test"))
    Xva, yva = va[cols].astype(float), va.y.values
    Xte, yte = te[cols].astype(float), te.y.values

    comp_df, test_probs, artifacts = compare_training_windows(
        tr, va, te, cols, club_cols, split_seasons["train"]
    )
    comp_df.to_csv(f"{out_dir}/window_comparison.csv", index=False)

    always_home_acc = float((yte == 0).mean())
    plot_window_comparison(comp_df, always_home_acc, f"{out_dir}/window_comparison.png")
    plot_window_calibration(test_probs, yte, f"{out_dir}/window_calibration_overlay.png")

    log("\n=== Training Window Comparison (lower log_loss / brier / ece is better) ===")
    display_cols = [
        "variant", "train_years", "n_refit", "best_n_full",
        "val_full_logloss", "test_elo_logloss", "test_club_logloss", "test_full_logloss",
        "test_full_brier", "test_full_acc", "test_full_ece", "test_full_draw_auc",
    ]
    log(comp_df[display_cols].round(4).to_string(index=False))
    best_val_row = comp_df.loc[comp_df["val_full_logloss"].idxmin()]
    best_test_row = comp_df.loc[comp_df["test_full_logloss"].idxmin()]
    log(f"Best on val  : {best_val_row['variant']} (val_log_loss={best_val_row['val_full_logloss']:.4f})")
    log(f"Best on test : {best_test_row['variant']} (test_log_loss={best_test_row['test_full_logloss']:.4f}, acc={best_test_row['test_full_acc']:.4f})")
    log(f"Always-home accuracy on test: {always_home_acc:.3f}")

    chosen_name = str(best_val_row["variant"])
    chosen = artifacts[chosen_name]
    gbm = chosen["gbm_val"]
    final = chosen["final"]
    Pte = test_probs[chosen_name]

    res = {
        "chosen_variant": chosen_name,
        "window_comparison": comp_df.to_dict("records"),
        "test": {
            "elo": evaluate(artifacts[chosen_name]["elo_lr"].predict_proba(te[["elo_diff"]]), yte),
            "gbm_full_chosen": evaluate(Pte, yte),
            "always_home_accuracy": always_home_acc,
        },
    }

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
    log(f"\n=== Feature importance (chosen variant: {chosen_name}) ===")
    log(imp.round(5).to_string())

    comp_default = (
        float(feats.comp_home_rate.mean()),
        float(feats.comp_draw_rate.mean()),
        float(feats.comp_goals_pg.mean()),
    )
    comp_map = (
        feats.sort_values("date")
        .groupby("competition_id")[["comp_home_rate", "comp_draw_rate", "comp_goals_pg"]]
        .last()
        .apply(tuple, axis=1)
        .to_dict()
    )
    rs = RatingSystem(
        final, cols, snap_dates, snap_vals, feats[feats.has_lineups],
        comp_default, club_names, hfa=cfg["hfa"], comp_map=comp_map,
    )
    test_end_date = te["date"].max() + pd.Timedelta(days=1)
    table = rs.ratings(as_of=test_end_date)
    table.to_csv(f"{out_dir}/ranking_latest.csv", index=False)
    log(f"\n=== Top 20 clubs as of {test_end_date.date()} ===")
    log(table.head(20).round(3).to_string())

    season_dtype = d["season"].dtype
    latest_season = season_dtype.type(split_seasons["test"][-1])
    sc = sanity_check(rs, d, latest_season)
    if sc.empty and len(split_seasons["test"]) > 0:
        prev_season = season_dtype.type(split_seasons["test"][-1] - 1)
        sc = sanity_check(rs, d, prev_season)
        latest_season = prev_season
    sc.to_csv(f"{out_dir}/sanity_vs_league_tables.csv", index=False)
    log(f"\n=== Rank correlation with real league tables (season {latest_season}) ===")
    log(sc.to_string() if len(sc) else "(no domestic league with enough clubs)")

    preds_2026 = predict_year(d, feats, final, rs, cols, comp_default, cfg, year=2026)
    if len(preds_2026):
        preds_2026.to_csv(f"{out_dir}/predictions_2026.csv", index=False)
        played_2026 = preds_2026[preds_2026["status"] == "played"]
        acc_2026 = played_2026["correct"].mean() if len(played_2026) else float("nan")
        log(f"\nSaved {len(preds_2026)} predictions for 2026 to {out_dir}/predictions_2026.csv (played acc={acc_2026:.3f})")

    # Refit on all good seasons (train+val+test) and save standalone predictor.pkl
    all_ok = feats["split"].isin(["train", "val", "test"])
    prod_gbm = copy.copy(gbm)
    prod_gbm.refit(feats.loc[all_ok, cols].astype(float), feats.loc[all_ok, "y"].values)

    dom = feats[feats["competition_id"].astype(str).str.endswith("1")].sort_values("date")
    club_primary_comp = {}
    for r in dom[["home", "away", "competition_id"]].itertuples():
        club_primary_comp[r.home] = r.competition_id
        club_primary_comp[r.away] = r.competition_id

    predictor_bundle = dict(
        model=prod_gbm.m,
        cols=cols,
        hfa=cfg["hfa"],
        snap_dates=dict(snap_dates),
        snap_vals=dict(snap_vals),
        club_names=club_names,
        comp_default=comp_default,
        comp_map=comp_map,
        club_primary_comp=club_primary_comp,
        last_date=feats["date"].max(),
    )
    with open(f"{out_dir}/predictor.pkl", "wb") as fh:
        pickle.dump(predictor_bundle, fh)
    log(f"Saved standalone predictor to {out_dir}/predictor.pkl")

    try:
        feats.to_parquet(f"{out_dir}/features.parquet", index=False)
    except Exception:
        feats.to_csv(f"{out_dir}/features.csv", index=False)
    with open(f"{out_dir}/rating_system.pkl", "wb") as fh:
        pickle.dump(rs, fh)
    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    return dict(
        feats=feats, rs=rs, metrics=res, window_comparison=comp_df, importance=imp,
        coverage=by_season, good_seasons=good, ranking=table, sanity=sc,
        predictions_2026=preds_2026,
    )


def make_synthetic(n_clubs=18, n_seasons=10, seed=0):
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
            rows.append(dict(
                game_id=gid, date=date, competition_id="L1", season=2010 + s,
                home_club_name=clubs[h], away_club_name=clubs[a],
                home_club_goals=rng.poisson(lam_h), away_club_goals=rng.poisson(lam_a),
                home_starters=sides["home"][0] if lineups_ok else [],
                home_substitutes=sides["home"][1] if lineups_ok else [],
                away_starters=sides["away"][0] if lineups_ok else [],
                away_substitutes=sides["away"][1] if lineups_ok else [],
            ))
            gid += 1
    df = pd.DataFrame(rows)
    df["home_players"] = df.home_starters + df.home_substitutes
    df["away_players"] = df.away_starters + df.away_substitutes
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Club+Player Elo & GBM Match Predictor")
    parser.add_argument("--db", default="transfermarkt-datasets.duckdb", help="Path to DuckDB file")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic test data instead of DuckDB")
    args = parser.parse_args()

    if args.synthetic:
        df = make_synthetic()
    else:
        import duckdb
        con = duckdb.connect(args.db, read_only=True)
        query = """
        WITH lineups AS (
            SELECT
                game_id,
                club_id,
                list(DISTINCT CAST(player_id AS VARCHAR)) FILTER (WHERE type = 'starting_lineup') AS starters,
                list(DISTINCT CAST(player_id AS VARCHAR)) FILTER (WHERE type = 'substitutes')     AS substitutes
            FROM game_lineups
            WHERE player_id IS NOT NULL
            GROUP BY game_id, club_id
        )
        SELECT
            g.game_id, g.date, g.competition_id, g.season,
            g.home_club_id, g.away_club_id,
            COALESCE(hc.name, g.home_club_name) AS home_club_name,
            COALESCE(ac.name, g.away_club_name) AS away_club_name,
            g.home_club_goals, g.away_club_goals,
            hl.starters    AS home_starters,
            hl.substitutes AS home_substitutes,
            al.starters    AS away_starters,
            al.substitutes AS away_substitutes
        FROM games g
        INNER JOIN clubs hc ON hc.club_id = g.home_club_id
        INNER JOIN clubs ac ON ac.club_id = g.away_club_id
        LEFT JOIN lineups hl ON hl.game_id = g.game_id AND hl.club_id = g.home_club_id
        LEFT JOIN lineups al ON al.game_id = g.game_id AND al.club_id = g.away_club_id
        WHERE g.date IS NOT NULL
        ORDER BY g.date
        """
        df = con.sql(query).df()
        con.close()

    res = run(df, out_dir=args.out)
    print(res["ranking"].head(20))
