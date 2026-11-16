"""Перевірка схеми таблиць ознак за CONTRACT.md (стійка до нових f_-колонок у v2)."""
import numpy as np
import pandas as pd

from src import config
from tests.conftest import feature_cols

META = ["match_id", "date", "home", "away", "competition", "target", "y", "home_goals", "away_goals"]
META_EXTRA = {"clubs": ["season"], "intl": []}
REQUIRED_F = {
    "clubs": ["f_elo_home", "f_elo_away", "f_elo_diff", "f_cat_division",
              "f_odds_p_home", "f_odds_p_draw", "f_odds_p_away"],
    "intl": ["f_elo_home", "f_elo_away", "f_elo_diff", "f_neutral", "f_cat_tournament_type"],
}
ODDS_P = ["f_odds_p_home", "f_odds_p_draw", "f_odds_p_away"]


def test_required_columns(track_features):
    """Усі обов'язкові мета-колонки й ознаки з контракту присутні."""
    track, df = track_features
    missing = [c for c in META + META_EXTRA[track] + REQUIRED_F[track] if c not in df.columns]
    assert not missing, f"[{track}] бракує колонок: {missing}"


def test_only_meta_and_features(track_features):
    """Кожна колонка — або мета, або ознака з префіксом f_ (модель не може взяти рахунок як ознаку)."""
    track, df = track_features
    allowed = set(META + META_EXTRA[track])
    extra = [c for c in df.columns if c not in allowed and not c.startswith("f_")]
    assert not extra, f"[{track}] колонки поза схемою (не мета і без префікса f_): {extra}"


def test_feature_dtypes(track_features):
    """f_cat_* — category; решта f_* — числові."""
    track, df = track_features
    for c in feature_cols(df):
        if c.startswith("f_cat_"):
            assert isinstance(df[c].dtype, pd.CategoricalDtype), f"[{track}] {c} має бути category"
        else:
            assert pd.api.types.is_numeric_dtype(df[c]), f"[{track}] {c} має бути числовою"


def test_match_id_unique_and_dates(track_features):
    """match_id унікальний, дати відсортовані й не раніше MIN_DATE."""
    track, df = track_features
    assert df["match_id"].is_unique, f"[{track}] match_id не унікальний"
    dates = pd.to_datetime(df["date"])
    assert dates.is_monotonic_increasing, f"[{track}] дати не відсортовані"
    assert dates.min() >= pd.Timestamp(config.MIN_DATE), f"[{track}] є матчі до {config.MIN_DATE}"


def test_target_consistent(track_features):
    """y відповідає target (0=H, 1=D, 2=A), а target — рахунку."""
    track, df = track_features
    mapping = {c: i for i, c in enumerate(config.CLASSES)}
    assert (df["target"].astype(str).map(mapping) == df["y"]).all(), f"[{track}] y не узгоджений з target"
    gd = df["home_goals"] - df["away_goals"]
    by_goals = np.select([gd > 0, gd == 0], ["H", "D"], "A")
    assert (df["target"].astype(str).to_numpy() == by_goals).all(), f"[{track}] target не узгоджений з голами"


def test_odds_probabilities(track_features):
    """f_odds_p_* у [0, 1] і в сумі ≈ 1 (або всі NaN, якщо коефіцієнтів немає)."""
    track, df = track_features
    if track != "clubs":
        assert not [c for c in df.columns if c.startswith("f_odds_")], "у збірних не має бути f_odds_"
        return
    p = df[ODDS_P]
    filled = p.notna().all(axis=1)
    assert (p.notna().any(axis=1) == filled).all(), "частково заповнені f_odds_p_* (мають бути всі або жодна)"
    vals = p[filled].to_numpy()
    assert ((vals >= 0) & (vals <= 1)).all(), "f_odds_p_* поза [0, 1]"
    assert np.allclose(vals.sum(axis=1), 1.0, atol=1e-6), "сума f_odds_p_* не дорівнює 1"
