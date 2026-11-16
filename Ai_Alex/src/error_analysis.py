"""Аналіз помилок моделей (підчат ANALYST).

Висновки для покращення робимо лише на VALIDATION (2022-07-01 … 2023-06-30) і train;
test (>= 2023-07-01) — тільки для підсумкових ілюстрацій (топ промахів, розрив по сезонах).
Запуск: PYTHONUTF8=1 .venv/Scripts/python -m src.error_analysis
Результат: reports/error_analysis.md, reports/figures/err_*.png, print-и з таблицями.
Увага: розділи «Головне» і «Чого модель не може знати» в reports/error_analysis.md дописано вручну
за результатами прогону v2; повторний запуск перезапише файл лише таблицями.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

P = config.PROCESSED
FIG = config.FIGURES
CL = config.CLASSES
MODEL, MARKET = "final_no_odds", "baseline_odds"
OUT: list[str] = []          # рядки markdown-звіту


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown-таблиця без залежності від tabulate."""
    d = df.reset_index() if index else df
    fmt = lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) and abs(x) < 1e5 else str(x)  # noqa: E731
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def table(df: pd.DataFrame, floatfmt: int = 4) -> None:
    say(md_table(df.round(floatfmt)))
    say()


def probs(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[[f"p__{name}__{c}" for c in CL]].to_numpy(float)


def row_ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def ll(p: np.ndarray, y: np.ndarray) -> float:
    return float(row_ll(p, y).mean())


def read(path):
    """Файли можуть перезаписуватися іншими підчатами — пробуємо кілька разів."""
    import time
    for i in range(5):
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            time.sleep(30)
    return pd.read_parquet(path)


# ---------------------------------------------------------------- дані
def load_clubs() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = read(P / "preds_clubs.parquet")
    feats = read(P / "clubs_features.parquet")
    meta_cols = ["match_id", "home", "away", "home_goals", "away_goals"]
    df = preds.merge(feats[meta_cols + [c for c in feats.columns if c.startswith("f_")]], on="match_id", how="left")
    # новачок ліги: команда не грала в цьому дивізіоні в попередньому сезоні
    long = pd.concat([feats[["competition", "season", "home"]].rename(columns={"home": "team"}),
                      feats[["competition", "season", "away"]].rename(columns={"away": "team"})]).drop_duplicates()
    played = set(zip(long.competition, long.season, long.team))
    first_season = feats.groupby("competition")["season"].min().to_dict()
    def newcomer(team, comp, season):
        return season > first_season.get(comp, season) and (comp, season - 1, team) not in played
    df["home_new"] = [newcomer(t, c, s) for t, c, s in zip(df.home, df.competition, df.season)]
    df["away_new"] = [newcomer(t, c, s) for t, c, s in zip(df.away, df.competition, df.season)]
    df["any_new"] = df.home_new | df.away_new
    m = df["date"].dt.month
    df["phase"] = np.select([m.isin([7, 8, 9]), m.isin([10, 11, 12, 1, 2])], ["1 початок (лип–вер)", "2 середина (жов–лют)"], "3 кінець (бер–чер)")
    df["has_shots"] = df["f_home_shots_all5"].notna() & df["f_away_shots_all5"].notna()
    rest = df[["f_home_rest_days", "f_away_rest_days"]].min(axis=1)
    df["rest_bin"] = pd.cut(rest, [-1, 3, 5, 8, 15, 10_000], labels=["≤3", "4–5", "6–8", "9–15", ">15"])
    ed = df["f_own_elo_diff"]
    df["elo_bin"] = pd.cut(ed, [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf])
    yv = df["y"].to_numpy()
    df["ll_model"] = row_ll(probs(df, MODEL), yv)
    df["ll_market"] = row_ll(probs(df, MARKET), yv)
    df["gap"] = df["ll_model"] - df["ll_market"]
    return df, feats


def seg(df: pd.DataFrame, by: str, min_n: int = 150) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(n=("gap", "size"), ll_model=("ll_model", "mean"),
                                          ll_market=("ll_market", "mean"), gap=("gap", "mean"))
    g["gap_total"] = g["gap"] * g["n"] / len(df)      # внесок у загальний розрив
    return g[g["n"] >= min_n]


# ---------------------------------------------------------------- 1. розрив з ринком
def gap_analysis(df: pd.DataFrame) -> dict:
    v = df[(df.split == "val") & df[[f"p__{MARKET}__H"]].notna().all(axis=1)].copy()
    say("## 1. Де модель без коефіцієнтів програє ринку (validation 2022/23)")
    say()
    say(f"Validation: {len(v):,} матчів із коефіцієнтами. Log loss моделі `{MODEL}` (навчена лише на train) = "
        f"{v.ll_model.mean():.4f}, ринку = {v.ll_market.mean():.4f}, розрив = **{v.gap.mean():+.4f}**. "
        "`gap` — середня різниця log loss (модель − ринок) у сегменті, `gap_total` — внесок сегмента в загальний розрив.")
    say()
    res = {}
    for by, title in [("phase", "Фаза сезону"), ("elo_bin", "Різниця власного Elo (господарі − гості)"),
                      ("has_clubelo", "Є ClubElo"), ("has_shots", "Є ковзна статистика ударів"),
                      ("any_new", "У матчі є новачок ліги"), ("rest_bin", "Мінімальний відпочинок, днів")]:
        say(f"**{title}**")
        say()
        t = seg(v, by)
        table(t)
        res[by] = t
    say("**Ліги (відсортовано за внеском у розрив)**")
    say()
    t = seg(v, "competition").sort_values("gap_total", ascending=False)
    table(t)
    res["competition"] = t
    # фаза сезону як кількість зіграних командою матчів
    v["played_bin"] = pd.cut(v[["f_home_season_played", "f_away_season_played"]].min(axis=1), [-1, 0, 3, 8, 15, 25, 100],
                             labels=["0", "1–3", "4–8", "9–15", "16–25", ">25"])
    say("**Скільки матчів сезону вже зіграла команда (мінімум із двох)**")
    say()
    t = seg(v, "played_bin")
    table(t)
    res["played_bin"] = t
    # ілюстрація: test по сезонах (лише для звіту)
    tt = df[(df.split == "test") & df[f"p__{MARKET}__H"].notna()]
    say("**Test (лише ілюстрація): розрив по сезонах**")
    say()
    table(seg(tt, "season", 50))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    lg = res["competition"].sort_values("gap")
    colors = ["#C62828" if g > 0 else "#2E7D32" for g in lg["gap"]]
    axes[0].barh(lg.index, lg["gap"], color=colors)
    axes[0].axvline(v.gap.mean(), color="k", ls="--", lw=1, label=f"усі ліги {v.gap.mean():+.4f}")
    axes[0].set_title("Розрив log loss модель − ринок по лігах (val 2022/23)")
    axes[0].set_xlabel("Δ log loss (більше = модель гірша за ринок)"); axes[0].legend()
    axes[0].tick_params(axis="y", labelsize=8)
    eb = res["elo_bin"]
    axes[1].bar([str(i) for i in eb.index], eb["gap"], color="#4C72B0")
    axes[1].set_title("Розрив модель − ринок за різницею власного Elo")
    axes[1].set_xlabel("f_own_elo_diff"); axes[1].set_ylabel("Δ log loss")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout(); fig.savefig(FIG / "err_gap_by_league_elo.png", dpi=120); plt.close(fig)
    return res


# ---------------------------------------------------------------- 2. залишки проти ринку
def residual_analysis(df: pd.DataFrame) -> pd.DataFrame:
    v = df[(df.split == "val") & df[f"p__{MARKET}__H"].notna()].sort_values("date").copy()
    say("## 2. Що ринок знає, а модель ні: залишок r = p_ринку(H) − p_моделі(H)")
    say()
    v["r_H"] = v[f"p__{MARKET}__H"] - v[f"p__{MODEL}__H"]
    v["r_D"] = v[f"p__{MARKET}__D"] - v[f"p__{MODEL}__D"]
    say(f"Середнє |r_H| = {v.r_H.abs().mean():.3f}, std r_H = {v.r_H.std():.3f}; середнє r_D = {v.r_D.mean():+.4f} "
        f"(модель {'занижує' if v.r_D.mean() > 0 else 'завищує'} нічию відносно ринку).")
    say()
    feats = [c for c in df.columns if c.startswith("f_") and not c.startswith(("f_odds_", "f_hodds_"))]
    extra = ["home_new", "away_new", "has_clubelo", "month"]
    v["month"] = v["date"].dt.month
    X = v[feats + extra].copy()
    for c in ["home_new", "away_new", "has_clubelo"]:
        X[c] = X[c].astype(float)
    X["f_cat_division"] = X["f_cat_division"].astype("category")
    half = len(X) // 2
    params = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100,
                  feature_fraction=0.8, verbose=-1, seed=config.RANDOM_STATE)
    bst = lgb.train(params, lgb.Dataset(X.iloc[:half], v.r_H.iloc[:half]), 300)
    pred = bst.predict(X.iloc[half:])
    yt = v.r_H.iloc[half:].to_numpy()
    r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    say(f"LightGBM на залишок (навчання на 1-й половині val, перевірка на 2-й): R² = **{r2:.3f}** — "
        "частка розбіжності з ринком, яку можна пояснити наявними ознаками (тобто модель їх недовикористовує).")
    say()
    imp = pd.Series(bst.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    imp = (imp / imp.sum() * 100).head(15).rename("gain_%").to_frame()
    imp["spearman_r_H"] = [X[c].astype(float).corr(v.r_H, method="spearman") if X[c].dtype.name != "category" else np.nan for c in imp.index]
    table(imp, 3)
    # залишок по лігах і новачках
    g = v.groupby("competition")["r_H"].agg(["mean", "size"]).sort_values("mean")
    say("Найбільший систематичний зсув p(H) відносно ринку по лігах (mean r_H > 0 — ринок дає господарям більше):")
    say()
    table(pd.concat([g.head(4), g.tail(4)]), 3)
    t = v.groupby(["home_new", "away_new"])["r_H"].agg(["mean", "size"])
    say("Зсув для новачків ліги (home_new, away_new):")
    say()
    table(t, 3)
    return imp


# ---------------------------------------------------------------- 3. калібрування
def calibration(df: pd.DataFrame) -> None:
    v = df[df.split == "val"]
    say("## 3. Калібрування моделі без коефіцієнтів (val)")
    say()
    y = v["y"].to_numpy()
    pm, pk = probs(v, MODEL), probs(v, MARKET)
    rows = []
    for k, c in enumerate(CL):
        rows.append({"клас": c, "факт": (y == k).mean(), "середня p моделі": pm[:, k].mean(),
                     "середня p ринку": np.nanmean(pk[:, k]), "max p моделі (99-й перц.)": np.quantile(pm[:, k], 0.99),
                     "частка матчів, де клас — argmax моделі": (pm.argmax(1) == k).mean()})
    table(pd.DataFrame(rows).set_index("клас"), 3)
    lg = v.groupby("competition").apply(lambda d: pd.Series({
        "n": len(d), "D факт": (d.y == 1).mean(), "D модель": d[f"p__{MODEL}__D"].mean(),
        "H факт": (d.y == 0).mean(), "H модель": d[f"p__{MODEL}__H"].mean()}), include_groups=False)
    lg["D зсув"] = lg["D модель"] - lg["D факт"]
    lg["H зсув"] = lg["H модель"] - lg["H факт"]
    say("Ліги з найбільшим зсувом (модель − факт) за нічиїми і перемогами господарів (на одному сезоні val, шум ±2–3 п.п. при n≈300):")
    say()
    table(lg.reindex(lg["D зсув"].abs().sort_values(ascending=False).index).head(8), 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    edges = np.linspace(0, 0.9, 19)
    pal = {"H": "#2E7D32", "D": "#9E9E9E", "A": "#C62828"}
    for k, c in enumerate(CL):
        d = pd.DataFrame({"p": pm[:, k], "y": (y == k).astype(float), "b": pd.cut(pm[:, k], edges)})
        cal = d.groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        cal = cal[cal.n >= 60]
        axes[0].plot(cal.p, cal.y, marker="o", color=pal[c], label=c)
    axes[0].plot([0, 0.9], [0, 0.9], "k--", lw=1)
    axes[0].set_title("Калібрування моделі без коефіцієнтів (val)")
    axes[0].set_xlabel("Імовірність моделі"); axes[0].set_ylabel("Фактична частота"); axes[0].legend()
    sns.kdeplot(pm[:, 1], ax=axes[1], label="модель", color="#4C72B0", fill=True)
    sns.kdeplot(pk[~np.isnan(pk[:, 1]), 1], ax=axes[1], label="ринок", color="#DD8452", fill=True)
    axes[1].set_title("Розподіл p(нічия): модель vs ринок (val)"); axes[1].set_xlabel("p(D)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "err_calibration.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------- 4. абляція
GROUPS = {
    "ClubElo (f_elo_*)": lambda c: c.startswith("f_elo_"),
    "власний Elo (f_own_elo_*)": lambda c: c.startswith("f_own_elo_"),
    "форма (form*)": lambda c: "form" in c,
    "ковзні голи (gf/ga)": lambda c: "_gf_" in c or "_ga_" in c,
    "ковзні удари/кутові": lambda c: any(s in c for s in ("_shots_", "_sot_", "_corners_")),
    "H2H": lambda c: c.startswith("f_h2h"),
    "сезонний контекст (season_*)": lambda c: "season_" in c,
    "нічиї (draw_share, league_draw_rate)": lambda c: "draw_share" in c or "league_draw_rate" in c,
    "відпочинок (rest)": lambda c: "rest" in c,
    "ліга (f_cat_division)": lambda c: c == "f_cat_division",
}


def ablation(feats_df: pd.DataFrame) -> pd.DataFrame:
    say("## 4. Групова абляція (клуби, без коефіцієнтів)")
    say()
    f = feats_df
    tr = f[(f.date < config.VAL_START) & (f.season >= 2014)]
    va = f[(f.date >= config.VAL_START) & (f.date < config.TEST_START)]
    tr = tr.sample(n=min(80_000, len(tr)), random_state=config.RANDOM_STATE)
    allf = [c for c in f.columns if c.startswith("f_") and not c.startswith(("f_odds_", "f_hodds_"))]
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_depth=6,
                  verbose=-1, seed=config.RANDOM_STATE, num_threads=4)

    def fit_eval(cols):
        b = lgb.train(params, lgb.Dataset(tr[cols], tr.y), 350)
        return ll(b.predict(va[cols]), va.y.to_numpy())

    base = fit_eval(allf)
    rows = [{"група": "усі ознаки (база)", "n_ознак": len(allf), "val log loss": base, "Δ без групи": 0.0}]
    for name, rule in GROUPS.items():
        drop = [c for c in allf if rule(c)]
        if not drop:
            continue
        s = fit_eval([c for c in allf if c not in drop])
        rows.append({"група": name, "n_ознак": len(drop), "val log loss": s, "Δ без групи": s - base})
        print(f"  абляція {name}: {s:.4f}")
    only_elo = [c for c in allf if GROUPS["власний Elo (f_own_elo_*)"](c) or GROUPS["ClubElo (f_elo_*)"](c)] + ["f_cat_division"]
    rows.append({"група": "ЛИШЕ Elo (обидва) + ліга", "n_ознак": len(only_elo), "val log loss": fit_eval(only_elo),
                 "Δ без групи": np.nan})
    t = pd.DataFrame(rows).set_index("група")
    t.loc["ЛИШЕ Elo (обидва) + ліга", "Δ без групи"] = t.loc["ЛИШЕ Elo (обидва) + ліга", "val log loss"] - base
    say(f"LightGBM із фіксованими параметрами (350 дерев, lr 0.05), train — випадкові 80 000 матчів сезонів 2014–2021, "
        f"оцінка на всьому val ({len(va):,}). Δ > 0 — без групи гірше (група корисна); Δ ≈ 0 або < 0 — шум. "
        "Шум самої процедури ~±0.001.")
    say()
    table(t)
    fig, ax = plt.subplots(figsize=(9, 5))
    tt = t.drop(["усі ознаки (база)", "ЛИШЕ Elo (обидва) + ліга"]).sort_values("Δ без групи")
    ax.barh(tt.index, tt["Δ без групи"], color=["#2E7D32" if d > 0.0005 else "#9E9E9E" for d in tt["Δ без групи"]])
    ax.axvline(0, color="k", lw=1)
    ax.set_title("Абляція: на скільки погіршується val log loss без групи ознак")
    ax.set_xlabel("Δ log loss (більше = група корисніша)")
    fig.tight_layout(); fig.savefig(FIG / "err_ablation.png", dpi=120); plt.close(fig)
    return t


# ---------------------------------------------------------------- 5. збірні
def intl() -> None:
    say("## 5. Збірні: де модель виграє/програє в бейзлайну «лише Elo»")
    say()
    p = read(P / "preds_intl.parquet")
    f = read(P / "intl_features.parquet")
    df = p.merge(f[["match_id", "home", "away", "home_goals", "away_goals", "f_neutral", "f_elo_diff",
                    "f_cat_home_confed", "f_cat_away_confed"]], on="match_id", how="left")
    y = df.y.to_numpy()
    df["ll_model"] = row_ll(probs(df, MODEL), y)
    df["ll_elo"] = row_ll(probs(df, "baseline_elo"), y)
    df["gap"] = df["ll_model"] - df["ll_elo"]
    df["elo_bin"] = pd.cut(df.f_elo_diff.abs(), [-1, 50, 100, 200, 350, 10_000],
                           labels=["0–50", "50–100", "100–200", "200–350", ">350"])
    df["confed"] = df.f_cat_home_confed.astype(str)
    v = df[df.split == "val"]
    say(f"Validation збірних: лише {len(v)} матчів, тому сегменти шумні; поруч — test ({(df.split == 'test').sum()}) як ілюстрація. "
        "`gap` < 0 — модель краща за Elo-бейзлайн.")
    say()
    for by, title in [("f_cat_tournament_type", "Тип турніру"), ("f_neutral", "Нейтральне поле"),
                      ("elo_bin", "|різниця Elo|"), ("confed", "Конфедерація господарів")]:
        rows = []
        for split in ["val", "test"]:
            d = df[df.split == split]
            g = d.groupby(by, observed=True).agg(n=("gap", "size"), ll_model=("ll_model", "mean"), gap=("gap", "mean"),
                                                 D_fact=("y", lambda s: (s == 1).mean()))
            g.columns = [f"{c}_{split}" for c in g.columns]
            rows.append(g)
        say(f"**{title}**")
        say()
        table(pd.concat(rows, axis=1), 3)
    pm = probs(v, MODEL)
    say(f"Нічиї у val збірних: факт {(v.y == 1).mean():.3f}, середня p моделі {pm[:, 1].mean():.3f}, "
        f"нічия — argmax моделі в {(pm.argmax(1) == 1).mean():.1%} матчів.")
    say()
    tt = df[df.split == "test"].groupby("elo_bin", observed=True).agg(gap=("gap", "mean"), acc=("y", "size"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, by in zip(axes, ["f_cat_tournament_type", "elo_bin"]):
        g = df.groupby([by, "split"], observed=True)["gap"].mean().unstack()
        g.plot.bar(ax=ax, color={"val": "#4C72B0", "test": "#DD8452"})
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("Δ log loss модель − Elo-бейзлайн"); ax.set_xlabel(by); ax.tick_params(axis="x", rotation=0)
    axes[0].set_title("Збірні: виграш над Elo за типом турніру"); axes[1].set_title("…і за |різницею Elo|")
    fig.tight_layout(); fig.savefig(FIG / "err_intl.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------- 6. промахи
def top_misses(df: pd.DataFrame) -> None:
    say("## 6. Найбільші промахи (test, модель без коефіцієнтів) — ілюстрація")
    say()
    t = df[df.split == "test"].copy()
    pm = probs(t, MODEL)
    t["pred"] = np.array(CL)[pm.argmax(1)]
    t["conf"] = pm.max(1)
    t["fact"] = np.array(CL)[t.y]
    t["p_fact_model"] = pm[np.arange(len(t)), t.y]
    t["p_fact_market"] = probs(t, MARKET)[np.arange(len(t)), t.y]
    w = t[t.pred != t.fact].sort_values("conf", ascending=False).head(10)
    w["рахунок"] = w.home_goals.astype(int).astype(str) + ":" + w.away_goals.astype(int).astype(str)
    out = w[["date", "competition", "home", "away", "рахунок", "pred", "conf", "p_fact_model", "p_fact_market"]].copy()
    out["date"] = out.date.dt.date
    say(md_table(out.round(3), index=False))
    say()
    say(f"Для порівняння: у тих самих 10 матчах ринок давав фактичному результату в середньому "
        f"{w.p_fact_market.mean():.3f}, модель — {w.p_fact_model.mean():.3f}. Це «сенсації», яких не бачив ніхто.")
    say()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    say("# Аналіз помилок моделей")
    say()
    say("Висновки для покращення — лише на validation (сезон 2022/23, моделі навчені на train). Test (з 2023-07-01) — "
        "тільки для ілюстрацій. Модель: `final_no_odds` (ансамбль LightGBM + логрегресія, v2); ринок — імовірності "
        "букмекерів без маржі (`baseline_odds`). Згенеровано `python -m src.error_analysis`.")
    say()
    df, feats = load_clubs()
    gap_analysis(df)
    residual_analysis(df)
    calibration(df)
    ablation(feats)
    intl()
    top_misses(df)
    (config.REPORTS / "error_analysis.md").write_text("\n".join(OUT), encoding="utf-8")
    print("\nзбережено reports/error_analysis.md і reports/figures/err_*.png")


if __name__ == "__main__":
    main()
