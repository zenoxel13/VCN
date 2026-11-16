"""Ознаки для матчів національних збірних + API для демо.

Ковзні ознаки й Elo рахуються на ПОВНІЙ історії з 1872 року, у вихідний файл
data/processed/intl_features.parquet ідуть лише матчі з config.MIN_DATE.
Усі ковзні — строго groupby(team) → shift(1) → rolling: поточний матч не враховується.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.clean_intl import GOALS_PATH, TOURNAMENT_TYPES, tournament_type
from src.elo_intl import (OFFICIAL_PARAMS, EloHistory, PoissonRatings, expected_home, home_bonus, k_factor,
                          poisson_outcome_probs, poisson_xg)

IN_PATH = config.PROCESSED / "intl_matches.parquet"
OUT_PATH = config.PROCESSED / "intl_features.parquet"

ROLL_WINDOWS = (5, 10)
# ковзні середні за попередні матчі: очки, голи, середній Elo суперників,
# «понад очікування» (фактичний результат − очікуваний за Elo), частка нічиїх
ROLL_STATS = ("pts", "gf", "ga", "opp_elo", "over", "draw")
DENSITY_DAYS = (30, 90)          # щільність календаря: матчів за останні N днів
GS_WINDOW = 10                   # вікно (матчів) для ознак бомбардирів
# частки голів команди за GS_WINDOW попередніх матчів; чисельник — колонка з goalscorers
GS_SHARES = {"pen_share": "gs_pen", "second_half_share": "gs_second_half",
             "late_share": "gs_late", "top_scorer_share": "gs_top_scorer"}
GS_COLS = ["gs_goals", "gs_pen", "gs_second_half", "gs_late", "gs_first_min", "gs_top_scorer"]
H2H_WINDOW = 5                   # скільки останніх очних зустрічей враховувати
REST_DAYS_CAP = 365              # довші паузи (рік+) обрізаємо: різниці там уже не важливі
POINTS = {"win": 3.0, "draw": 1.0, "loss": 0.0}
LEAK_CORR_LIMIT = 0.6            # |corr| ознаки з різницею голів вище цього = підозра на витік
CONFEDERATIONS = ["UEFA", "CAF", "AFC", "CONMEBOL", "CONCACAF", "OFC", "other"]
# префікс назви турніру → конфедерація (відбори й фінали континентальних турнірів)
CONFED_BY_PREFIX = {
    "UEFA": "UEFA", "African Cup of Nations": "CAF", "AFC": "AFC", "Copa América": "CONMEBOL",
    "Gold Cup": "CONCACAF", "CONCACAF": "CONCACAF", "CFU Caribbean Cup": "CONCACAF",
    "Oceania Nations Cup": "OFC",
}
POIS_COLS = ["pois_att_home", "pois_def_home", "pois_att_away", "pois_def_away"]
META_COLS = ["match_id", "date", "home", "away", "competition",
             "target", "y", "home_goals", "away_goals"]


def _outcome(diff: pd.Series, win, draw, loss) -> np.ndarray:
    """Відображає різницю голів у значення для перемоги/нічиєї/поразки (NaN без рахунку)."""
    return np.select([diff > 0, diff == 0, diff < 0], [win, draw, loss], np.nan)


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Довгий формат: по рядку на (матч, команда) з голами, очками, Elo й даними бомбардирів."""
    p_home = expected_home(matches["elo_home_pre"], matches["elo_away_pre"], matches["neutral"])
    sides = []
    for side, opp, expected in (("home", "away", p_home), ("away", "home", 1.0 - p_home)):
        part = matches[["match_id", "date", side, f"{side}_goals", f"{opp}_goals",
                        f"elo_{side}_pre", f"elo_{opp}_pre"]
                       + [f"{side}_{c}" for c in GS_COLS]].copy()
        part.columns = ["match_id", "date", "team", "gf", "ga", "elo_pre", "opp_elo"] + GS_COLS
        part["side"] = side
        part["expected"] = expected
        sides.append(part)
    long = pd.concat(sides, ignore_index=True)
    diff = long["gf"] - long["ga"]
    long["pts"] = _outcome(diff, POINTS["win"], POINTS["draw"], POINTS["loss"])
    long["over"] = _outcome(diff, 1.0, 0.5, 0.0) - long["expected"]
    long["draw"] = _outcome(diff, 0.0, 1.0, 0.0)
    return long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


def _recent_count(dates: pd.Series, days: int) -> np.ndarray:
    """Скільки попередніх матчів команди припадає на останні days днів (dates відсортовані)."""
    arr = dates.to_numpy()
    start = np.searchsorted(arr, arr - np.timedelta64(days, "D"), side="left")
    return np.arange(len(arr)) - start


def _shifted_rolling(long: pd.DataFrame, col: str, window: int, how: str) -> pd.Series:
    """groupby(team) → shift(1) → rolling(window).<how>: лише попередні матчі команди.

    min_periods=1: команда з малою історією отримує значення за наявні матчі, а не NaN.
    """
    prev = long.groupby("team", sort=False)[col].shift(1)
    roll = prev.groupby(long["team"], sort=False).rolling(window, min_periods=1)
    return getattr(roll, how)().reset_index(level=0, drop=True)


def _rolling_team_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Форма, голи, суперники, імпульс Elo, календар, бомбардири, відпочинок, досвід ДО матчу."""
    grouped = long.groupby("team", sort=False)
    stats = long[["match_id", "side"]].copy()
    for col in ROLL_STATS:
        for w in ROLL_WINDOWS:
            stats[f"{col}{w}"] = _shifted_rolling(long, col, w, "mean")
    for w in ROLL_WINDOWS:
        # імпульс: передматчевий Elo зараз мінус передматчевий Elo w матчів тому
        stats[f"elo_mom{w}"] = long["elo_pre"] - grouped["elo_pre"].shift(w)
    for days in DENSITY_DAYS:
        stats[f"n_last{days}d"] = grouped["date"].transform(
            lambda d, days=days: _recent_count(d, days)).astype(float)
    # бомбардири: суми за GS_WINDOW попередніх матчів (матчі без покриття = NaN, пропускаються)
    goals = _shifted_rolling(long, "gs_goals", GS_WINDOW, "sum")
    goals = goals.where(goals > 0)
    for name, col in GS_SHARES.items():
        stats[name] = _shifted_rolling(long, col, GS_WINDOW, "sum") / goals
    stats["first_goal_min"] = _shifted_rolling(long, "gs_first_min", GS_WINDOW, "mean")
    stats["rest_days"] = ((long["date"] - grouped["date"].shift(1)).dt.days
                          .clip(upper=REST_DAYS_CAP).astype(float))
    stats["n_matches"] = grouped.cumcount().astype(float)
    return stats


def _h2h_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Останні H2H_WINDOW очних зустрічей: кількість і баланс (перемоги − поразки господаря)."""
    home_first = matches["home"] <= matches["away"]
    sign = np.where(home_first, 1.0, -1.0)
    h2h = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["date"],
        "pair": np.where(home_first, matches["home"] + "|" + matches["away"],
                         matches["away"] + "|" + matches["home"]),
        # результат з погляду команди, що перша за алфавітом у парі
        "res": sign * np.sign(matches["home_goals"] - matches["away_goals"]),
        "sign": sign,
    }).sort_values(["pair", "date", "match_id"], kind="stable")
    prev = h2h.groupby("pair", sort=False)["res"].shift(1)
    roll = prev.groupby(h2h["pair"], sort=False).rolling(H2H_WINDOW, min_periods=1)
    h2h["f_h2h_n"] = roll.count().reset_index(level=0, drop=True).astype(float)
    h2h["f_h2h_balance"] = (roll.sum().reset_index(level=0, drop=True).fillna(0.0)
                            * h2h["sign"])
    return h2h[["match_id", "f_h2h_n", "f_h2h_balance"]]


def team_confederations(matches: pd.DataFrame) -> dict[str, str]:
    """Конфедерація збірної — найчастіша серед континентальних турнірів, де вона грала.

    Це статичний атрибут команди (а не результат матчу), тому береться з усієї історії.
    Той самий словник слугує й для країни проведення (назви країн = назви збірних).
    """
    confed = pd.Series(np.nan, index=matches.index, dtype="object")
    for prefix, name in CONFED_BY_PREFIX.items():
        confed[matches["competition"].str.startswith(prefix) & confed.isna()] = name
    known = matches.assign(confed=confed)[confed.notna()]
    teams = pd.concat([known[["home", "confed"]].rename(columns={"home": "team"}),
                       known[["away", "confed"]].rename(columns={"away": "team"})])
    return teams.groupby("team")["confed"].agg(lambda s: s.mode().iloc[0]).to_dict()


def _confed_category(teams: pd.Series, confeds: dict[str, str]) -> pd.Categorical:
    """Категоріальна колонка конфедерації (невідомі → other)."""
    return pd.Categorical(teams.map(confeds).fillna("other"), categories=CONFEDERATIONS)


def _add_match_level(df: pd.DataFrame, confeds: dict[str, str]) -> None:
    """Ознаки самого матчу: Elo, Пуассон, поле, турнір, конфедерації й географія (на місці)."""
    df["f_elo_home"] = df["elo_home_pre"].astype(float)
    df["f_elo_away"] = df["elo_away_pre"].astype(float)
    df["f_elo_diff"] = df["f_elo_home"] - df["f_elo_away"] + home_bonus(df["neutral"])
    df["f_elo_absdiff"] = df["f_elo_diff"].abs()
    df["f_elo_off_diff"] = (df["elo_off_home_pre"] - df["elo_off_away_pre"]
                            + home_bonus(df["neutral"]))
    df["f_elo_p_home"] = expected_home(df["f_elo_home"], df["f_elo_away"], df["neutral"])
    xg_home, xg_away = poisson_xg(*(df[c] for c in POIS_COLS), df["neutral"])
    df["f_xg_home"], df["f_xg_away"] = xg_home, xg_away
    df["f_xg_total"] = xg_home + xg_away
    probs = poisson_outcome_probs(xg_home, xg_away)
    df["f_pois_p_home"], df["f_pois_p_draw"], df["f_pois_p_away"] = probs.T
    df["f_neutral"] = df["neutral"].astype(float)
    df["f_home_true_home"] = (df["home"] == df["country"]).astype(float)
    df["f_away_true_home"] = (df["away"] == df["country"]).astype(float)
    df["f_k_factor"] = [k_factor(c, t) for c, t in zip(df["competition"], df["tournament_type"])]
    df["f_cat_tournament_type"] = pd.Categorical(df["tournament_type"],
                                                 categories=TOURNAMENT_TYPES)
    df["f_cat_home_confed"] = _confed_category(df["home"], confeds)
    df["f_cat_away_confed"] = _confed_category(df["away"], confeds)
    df["f_cat_venue_confed"] = _confed_category(df["country"], confeds)
    venue_known = df["f_cat_venue_confed"] != "other"
    for side in ("home", "away"):
        df[f"f_{side}_on_continent"] = (venue_known & (df[f"f_cat_{side}_confed"]
                                                       == df["f_cat_venue_confed"])).astype(float)
    known = (df["f_cat_home_confed"] != "other") & (df["f_cat_away_confed"] != "other")
    df["f_inter_confed"] = (known & (df["f_cat_home_confed"] != df["f_cat_away_confed"])
                            ).astype(float)


def build_features(matches: pd.DataFrame, confeds: dict[str, str] | None = None) -> pd.DataFrame:
    """Будує всі f_-ознаки для таблиці матчів.

    matches — результат _attach_elo (колонки clean_intl + Elo/Пуассон/бомбардири);
    confeds — результат team_confederations() на повній історії (None → рахується з matches).
    Рядки з NaN-рахунком (гіпотетичні матчі) теж отримують ознаки.
    """
    df = matches.copy()
    if confeds is None:
        confeds = team_confederations(df)
    _add_match_level(df, confeds)
    stats = _rolling_team_stats(_team_long(df))
    for side in ("home", "away"):
        part = stats[stats["side"] == side].drop(columns="side")
        part = part.rename(columns={c: f"f_{side}_{c}" for c in part.columns if c != "match_id"})
        df = df.merge(part, on="match_id", how="left")
    for w in ROLL_WINDOWS:
        df[f"f_pts{w}_diff"] = df[f"f_home_pts{w}"] - df[f"f_away_pts{w}"]
        df[f"f_gd{w}_diff"] = ((df[f"f_home_gf{w}"] - df[f"f_home_ga{w}"])
                               - (df[f"f_away_gf{w}"] - df[f"f_away_ga{w}"]))
        df[f"f_over{w}_diff"] = df[f"f_home_over{w}"] - df[f"f_away_over{w}"]
        df[f"f_elo_mom{w}_diff"] = df[f"f_home_elo_mom{w}"] - df[f"f_away_elo_mom{w}"]
    df = df.merge(_h2h_stats(df), on="match_id", how="left")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Усі колонки-ознаки (префікс f_)."""
    return [c for c in df.columns if c.startswith("f_")]


def _load_goals() -> pd.DataFrame | None:
    """Агрегати голів з goalscorers (None, якщо clean_intl їх не зберіг)."""
    return pd.read_parquet(GOALS_PATH) if GOALS_PATH.exists() else None


def _attach_goals(matches: pd.DataFrame, goals: pd.DataFrame | None) -> pd.DataFrame:
    """Додає {home,away}_gs_* — дані бомбардирів САМОГО матчу (лише для ковзних ознак).

    Матч без записів у goalscorers → NaN; покритий матч, де команда не забила → 0 голів.
    """
    out = matches.copy()
    for side in ("home", "away"):
        for c in GS_COLS:
            out[f"{side}_{c}"] = np.nan
    if goals is None:
        return out
    keys = ["date", "home", "away"]
    covered = (out[keys].merge(goals[keys].drop_duplicates().assign(cov=1.0), on=keys, how="left")
               ["cov"].notna().to_numpy())
    for side in ("home", "away"):
        part = goals[goals["team"] == goals[side]].drop(columns="team")
        merged = out[keys].merge(part, on=keys, how="left")
        for c in GS_COLS:
            vals = merged[c].to_numpy(dtype=float)
            if c != "gs_first_min":
                vals = np.where(np.isnan(vals), 0.0, vals)
            out[f"{side}_{c}"] = np.where(covered, vals, np.nan)
    return out


def _attach_elo(matches: pd.DataFrame
                ) -> tuple[pd.DataFrame, EloHistory, EloHistory, PoissonRatings]:
    """Проганяє Elo (усі й лише офіційні матчі) та Пуассон-рейтинги, додає передматчеві значення й голи."""
    elo = EloHistory(matches)
    elo_off = EloHistory(matches, OFFICIAL_PARAMS)
    pois = PoissonRatings(matches)
    out = _attach_goals(matches, _load_goals())
    out["elo_home_pre"], out["elo_away_pre"] = elo.pre_home, elo.pre_away
    out["elo_off_home_pre"], out["elo_off_away_pre"] = elo_off.pre_home, elo_off.pre_away
    for i, col in enumerate(POIS_COLS):
        out[col] = pois.pre[:, i]
    return out, elo, elo_off, pois


def report(feats: pd.DataFrame, elo: EloHistory) -> None:
    """Друкує санітарні перевірки таблиці ознак."""
    fcols = feature_columns(feats)
    print(f"Розмір: {feats.shape}; ознак f_: {len(fcols)}")
    print(f"match_id унікальний: {feats['match_id'].is_unique}")
    print(f"Діапазон дат: {feats['date'].min().date()} … {feats['date'].max().date()}")
    print("Частка NaN в ознаках (> 1 %):\n" + feats[fcols].isna().mean().round(3)
          [lambda s: s > 0.01].to_string())
    print("Розподіл target:\n" + feats["target"].value_counts(normalize=True).round(3).to_string())
    num = feats[fcols].select_dtypes("number")
    corr = num.corrwith(feats["home_goals"] - feats["away_goals"]).abs().sort_values(ascending=False)
    print("Тест на витік — |corr| з різницею голів (топ-8):\n" + corr.head(8).round(3).to_string())
    if corr.max() >= LEAK_CORR_LIMIT:
        raise ValueError(f"Підозра на витік: |corr| {corr.index[0]} з рахунком = {corr.max():.3f}")
    print("Топ-15 Elo на сьогодні (команди, що грали з 2022 р.):\n"
          + elo.top(15, since="2022-01-01").to_string())


def main() -> None:
    """Точка входу: matches → Elo/Пуассон → ознаки → фільтр за датою → parquet."""
    matches, elo, _, _ = _attach_elo(pd.read_parquet(IN_PATH))
    feats = build_features(matches, team_confederations(matches))
    feats = feats[feats["date"] >= config.MIN_DATE].sort_values(["date", "match_id"])
    feats = feats[META_COLS + feature_columns(feats)].reset_index(drop=True)
    report(feats, elo)
    feats.to_parquet(OUT_PATH, index=False)
    print(f"Збережено: {OUT_PATH}")


# ---------------------------------------------------------------- API для демо

@lru_cache(maxsize=1)
def _history() -> tuple[pd.DataFrame, EloHistory, EloHistory, PoissonRatings, dict[str, str]]:
    """Кешована повна історія з Elo, Пуассон-рейтингами й конфедераціями (один раз на процес)."""
    matches, elo, elo_off, pois = _attach_elo(pd.read_parquet(IN_PATH))
    return matches, elo, elo_off, pois, team_confederations(matches)


def build_match_row(home: str, away: str, date, competition: str,
                    neutral: bool = False, country: str | None = None,
                    **_extra) -> pd.DataFrame:
    """1 рядок з усіма f_-колонками для гіпотетичного матчу home–away на дату date.

    Стан команд береться лише з матчів СТРОГО до date: до історії двох команд
    дописується фіктивний рядок і проганяється та сама build_features, що й при навчанні.
    Країна проведення — country, якщо задано; інакше home (не нейтральне поле) або невідома.
    """
    matches, elo, elo_off, pois, confeds = _history()
    date = pd.Timestamp(date)
    own = matches[(matches["date"] < date)
                  & (matches["home"].isin([home, away]) | matches["away"].isin([home, away]))]
    dummy = pd.DataFrame([{
        "match_id": int(matches["match_id"].max()) + 1, "date": date, "home": home,
        "away": away, "home_goals": np.nan, "away_goals": np.nan,
        "competition": competition, "tournament_type": tournament_type(competition),
        "neutral": bool(neutral),
        "country": country if country is not None else ("" if neutral else home),
        "elo_home_pre": elo.rating_on(home, date), "elo_away_pre": elo.rating_on(away, date),
        "elo_off_home_pre": elo_off.rating_on(home, date),
        "elo_off_away_pre": elo_off.rating_on(away, date),
        **dict(zip(POIS_COLS, (*pois.state_on(home, date), *pois.state_on(away, date)))),
    }])
    feats = build_features(pd.concat([own, dummy], ignore_index=True), confeds)
    return feats[feats["match_id"] == dummy["match_id"].iloc[0]][
        feature_columns(feats)].reset_index(drop=True)


def list_competitions() -> list[str]:
    """Назви турнірів, що проводилися з config.MIN_DATE (за спаданням кількості матчів)."""
    matches = _history()[0]
    recent = matches[matches["date"] >= config.MIN_DATE]
    return recent["competition"].value_counts().index.tolist()


def list_teams(competition: str | None = None) -> list[str]:
    """Збірні, що грали з config.MIN_DATE (опційно — лише в цьому турнірі)."""
    matches = _history()[0]
    recent = matches[matches["date"] >= config.MIN_DATE]
    if competition is not None:
        recent = recent[recent["competition"] == competition]
    return sorted(set(recent["home"]) | set(recent["away"]))


if __name__ == "__main__":
    main()
