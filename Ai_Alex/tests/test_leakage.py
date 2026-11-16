"""Головний тест на витік даних.

«Тест збурення»: беремо очищену історію команд реального матчу, рахуємо ознаки тією самою
функцією, що будує parquet, потім змінюємо результат і всю статистику САМЕ цього матчу та
перераховуємо. Ознаки цього матчу мають лишитися незмінними (вони відомі до початку гри), а ознаки
наступного матчу команди — змінитися (інакше тест нічого не перевіряє).
"""
import numpy as np
import pandas as pd
import pytest

from src import features_clubs, features_intl
from tests.conftest import assert_rows_equal, feature_cols, read_processed

LEAK_CORR_LIMIT = 0.6
# Статистика клубного матчу, відома лише після гри (збурюємо всю).
CLUB_IN_MATCH = ["HTHome", "HTAway", "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
                 "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners", "HomeYellow", "AwayYellow",
                 "HomeRed", "AwayRed"]
# Коефіцієнти ПОТОЧНОГО матчу: можуть впливати лише на f_odds_* цього матчу, але не на f_hodds_/f_pi_/f_pois_ тощо.
CLUB_ODDS = ["OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway", "Over25", "Under25",
             "MaxOver25", "MaxUnder25", "HandiSize", "HandiHome", "HandiAway"]
CLUB_TEAM, INTL_TEAM = "Arsenal", "Spain"


def _pick_match(matches: pd.DataFrame, team: str) -> tuple[pd.DataFrame, int, int]:
    """Історія обох команд передостаннього матчу `team`; повертає (історія, id матчу, id наступного матчу team)."""
    own = matches[(matches["home"] == team) | (matches["away"] == team)].sort_values(["date", "match_id"])
    if len(own) < 3:
        pytest.skip(f"Замало матчів команди {team} для тесту збурення")
    target, nxt = own.iloc[-2], own.iloc[-1]
    teams = [target["home"], target["away"]]
    hist = matches[matches["home"].isin(teams) | matches["away"].isin(teams)].copy()
    return hist, int(target["match_id"]), int(nxt["match_id"])


def _perturb_result(df: pd.DataFrame, match_id: int, extra_cols: list[str]) -> pd.DataFrame:
    """Копія таблиці, де в матчі match_id перевернуто рахунок (9:0 ↔ 0:9) і змінено статистику."""
    out = df.copy()
    idx = out.index[out["match_id"] == match_id]
    hg, ag = out.loc[idx, "home_goals"].iloc[0], out.loc[idx, "away_goals"].iloc[0]
    new_h, new_a = (0.0, 9.0) if hg >= ag else (9.0, 0.0)
    out.loc[idx, "home_goals"], out.loc[idx, "away_goals"] = new_h, new_a
    out.loc[idx, "target"] = "H" if new_h > new_a else "A"
    out.loc[idx, "y"] = 0 if new_h > new_a else 2
    for c in extra_cols:
        if c in out.columns:
            out.loc[idx, c] = pd.to_numeric(out.loc[idx, c], errors="coerce").fillna(0) + 17
    return out


def _row(feats: pd.DataFrame, match_id: int) -> pd.Series:
    """Рядок ознак матчу за match_id."""
    return feats.loc[feats["match_id"] == match_id].iloc[0]


def _assert_perturbation(base: pd.DataFrame, pert: pd.DataFrame, mid: int, nxt: int, track: str,
                         allowed: tuple[str, ...] = ()) -> None:
    """Ознаки матчу mid (крім префіксів allowed) незмінні, ознаки наступного матчу nxt — змінилися."""
    cols = [c for c in feature_cols(base) if not c.startswith(allowed)] if allowed else feature_cols(base)
    assert_rows_equal(_row(base, mid), _row(pert, mid), cols,
                      f"[{track}] ВИТІК: ознаки матчу залежать від його власного результату/статистики")
    a, b = _row(base, nxt), _row(pert, nxt)
    num = [c for c in cols if not c.startswith("f_cat_")]
    changed = [c for c in num if not (pd.isna(a[c]) and pd.isna(b[c])) and not np.isclose(a[c], b[c], equal_nan=True)]
    assert changed, f"[{track}] збурення не вплинуло на наступний матч — тест не має сили"


def _intl_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Ознаки збірних так само, як у features_intl.main(): Elo по історії → build_features."""
    with_elo = features_intl._attach_elo(matches.sort_values(["date", "match_id"], kind="stable")
                                            .reset_index(drop=True))[0]
    return features_intl.build_features(with_elo)


def _clubs_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Ознаки клубів так само, як у features_clubs.main(): (власний Elo по історії, якщо є у v2) → compute_features."""
    add_own_elo = getattr(features_clubs, "add_own_elo", None)
    return features_clubs.compute_features(add_own_elo(matches) if add_own_elo else matches)


def test_perturbation_clubs():
    """Клуби: зміна рахунку, статистики й коефіцієнтів матчу не змінює його ознак, крім f_odds_ (compute_features)."""
    matches = read_processed("clubs_matches")
    hist, mid, nxt = _pick_match(matches, CLUB_TEAM)
    base = _clubs_features(hist)
    pert = _clubs_features(_perturb_result(hist, mid, CLUB_IN_MATCH + CLUB_ODDS))
    # f_odds_* — коефіцієнти самого матчу (дозволено, відомі до гри); усе інше, зокрема f_hodds_, — ні
    _assert_perturbation(base, pert, mid, nxt, "clubs", allowed=("f_odds_",))
    hodds = [c for c in feature_cols(base) if c.startswith("f_hodds_")]
    if hodds:
        a, b = _row(base, nxt)[hodds], _row(pert, nxt)[hodds]
        assert not np.allclose(a.astype(float), b.astype(float), equal_nan=True),             "[clubs] коефіцієнти матчу не вплинули на f_hodds_ наступного матчу — тест історії ринку не має сили"


def test_perturbation_intl():
    """Збірні: зміна рахунку матчу не змінює його ознак (ковзні, H2H і Elo — лише з попередніх матчів)."""
    matches = read_processed("intl_matches")
    hist, mid, nxt = _pick_match(matches, INTL_TEAM)
    base = _intl_features(hist)
    pert = _intl_features(_perturb_result(hist, mid, []))
    _assert_perturbation(base, pert, mid, nxt, "intl")


@pytest.mark.parametrize("track", ["clubs", "intl"])
def test_build_match_row_matches_parquet(track):
    """API демо для реального минулого матчу дає ті самі ознаки, що й parquet (крім f_odds_/f_elo_ у клубів)."""
    feats = read_processed(f"{track}_features")
    module = features_clubs if track == "clubs" else features_intl
    sample = feats[feats["date"] >= pd.Timestamp("2024-01-01")].sample(3, random_state=42)
    for _, r in sample.iterrows():
        extra = {} if track == "clubs" else {"neutral": bool(r["f_neutral"])}
        row = module.build_match_row(r["home"], r["away"], r["date"], r["competition"], **extra).iloc[0]
        cols = [c for c in feature_cols(feats) if c in row.index]
        if track == "intl":
            # Країни проведення немає в parquet, а демо без параметра country бере home/невідому —
            # ознаки місця проведення порівнювати не можна.
            cols = [c for c in cols if "venue" not in c and "on_continent" not in c and "true_home" not in c]
        if track == "clubs":
            # Elo в parquet — з Matches.csv, у демо — зі знімку ClubElo; коефіцієнти в демо не передаються.
            cols = [c for c in cols if not c.startswith(("f_odds_", "f_elo_"))]
        assert set(feature_cols(feats)) == set(row.index), f"[{track}] набір f_-колонок demo ≠ parquet"
        assert_rows_equal(row, r, cols, f"[{track}] build_match_row ≠ parquet для {r['home']}–{r['away']} {r['date'].date()}")


def test_no_feature_correlates_with_goal_diff(track_features):
    """Жодна числова f_ не має |corr| > 0.6 з різницею голів (грубий сигнал витоку)."""
    track, df = track_features
    num = [c for c in feature_cols(df) if not c.startswith("f_cat_")]
    corr = df[num].corrwith(df["home_goals"] - df["away_goals"]).abs()
    bad = corr[corr > LEAK_CORR_LIMIT]
    assert bad.empty, f"[{track}] підозра на витік: {bad.round(3).to_dict()}"
