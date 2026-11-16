"""Transfermarkt (f_tm_*) без підглядання в майбутнє і набори ознак моделей (v3)."""
import joblib
import pandas as pd
import pytest


from src.modeling import feature_sets
from tests.conftest import assert_rows_equal, read_processed, require_model

ft = pytest.importorskip("src.features_tm", reason="Немає src/features_tm.py (ознаки Transfermarkt)")
TM_RAW = ("games", "appearances", "player_valuations", "players")
MARKET_PREFIXES = ("f_odds_", "f_hodds_")


_RAW_CACHE: dict = {}


def _cached_read(name: str, usecols=None) -> pd.DataFrame:
    """Читає сирий файл TM один раз на сесію (обидві побудови стану використовують ту саму копію)."""
    key = (name, tuple(usecols) if usecols is not None else None)
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = _ORIGINAL_READ(name, usecols)
    return _RAW_CACHE[key].copy()


_ORIGINAL_READ = ft._read


def _raw_reader(cut: pd.Timestamp):
    """Підміна features_tm._read: лише записи, відомі ДО дня cut.

    games — ігри з датою ≤ cut (рядок самого дня потрібен як «передматчевий стан», тренер відомий
    до гри), але рахунок ігор дня cut стерто; appearances — лише з ігор < cut; оцінки — з датою < cut.
    """
    original = _cached_read
    games_all = original("games", ["game_id", "date"])
    games_all["date"] = pd.to_datetime(games_all["date"], errors="coerce")
    past_ids = set(games_all.loc[games_all["date"] < cut, "game_id"])

    def read(name, usecols=None):
        df = original(name, usecols)
        if name == "games":
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"] <= cut].copy()
            for c in ("home_club_goals", "away_club_goals"):
                if c in df:
                    df.loc[df["date"] == cut, c] = float("nan")
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        elif name == "appearances":
            df = df[df["game_id"].isin(past_ids)]
        elif name == "player_valuations":
            df = df[pd.to_datetime(df["date"], errors="coerce") < cut]
        return df
    return read


def _tm_for(matches: pd.DataFrame, state: pd.DataFrame, tmp_path, monkeypatch, tag: str) -> pd.DataFrame:
    """f_tm_ для матчів за заданою таблицею стану (через той самий add_tm_features)."""
    path = tmp_path / f"state_{tag}.parquet"
    state.to_parquet(path, index=False)
    monkeypatch.setattr(ft, "STATE_PATH", path)
    monkeypatch.setattr(ft, "_STATE_CACHE", None)
    return ft.add_tm_features(matches)


def test_tm_truncation(tmp_path, monkeypatch):
    """f_tm_ матчів дня D не змінюються, якщо з сирих даних TM прибрати все, що стало відомо з дня D."""
    if not all((ft.TM_DIR / f"{n}.csv.gz").exists() for n in TM_RAW):
        pytest.skip(f"Немає сирих даних Transfermarkt у {ft.TM_DIR}")
    m = read_processed("clubs_matches")
    e0 = m[(m["division"] == "E0") & m["date"].between("2024-01-01", "2024-05-31")]
    if e0.empty:
        pytest.skip("Немає матчів АПЛ у січні–травні 2024 для перевірки")
    cut = e0["date"].value_counts().idxmax()
    day = e0.loc[e0["date"] == cut, ["match_id", "date", "home", "away"]].reset_index(drop=True)

    # Словник club_id → назва фіксуємо з повних даних: він залежить від кількості збігів за всю
    # історію (відомий компроміс, у ознаки не протікає), а тест перевіряє саме стан команд.
    mapping = ft._map_clubs
    fixed: dict = {}

    def pinned(games, ours):
        """Перший виклик рахує словник на повних даних, наступні повертають той самий."""
        if not fixed:
            fixed.update(mapping(games, ours))
        return dict(fixed)
    monkeypatch.setattr(ft, "_map_clubs", pinned)
    monkeypatch.setattr(ft, "_read", _cached_read)

    full = _tm_for(day, ft.build_state(), tmp_path, monkeypatch, "full")
    tm_cols = [c for c in full.columns if c.startswith("f_tm_")]
    assert full[tm_cols].notna().any().any(), "f_tm_ для матчів АПЛ 2024 порожні — тест не має сили"

    monkeypatch.setattr(ft, "_read", _raw_reader(pd.Timestamp(cut)))
    cut_state = ft.build_state()
    trunc = _tm_for(day, cut_state, tmp_path, monkeypatch, "cut")
    for i in range(len(day)):
        assert_rows_equal(full.iloc[i], trunc.iloc[i], tm_cols,
                          f"ВИТІК TM: {day.loc[i, 'home']}–{day.loc[i, 'away']} {cut.date()} залежить від даних з дня матчу")


def test_feature_sets_exclude_market():
    """Набір no_odds — без f_odds_ і f_hodds_; hist_odds — без f_odds_ (за таблицею ознак клубів)."""
    sets = feature_sets(read_processed("clubs_features"))
    assert not [c for c in sets["no_odds"] if c.startswith(MARKET_PREFIXES)]
    if "hist_odds" in sets:
        assert not [c for c in sets["hist_odds"] if c.startswith("f_odds_")]


@pytest.mark.parametrize("filename,banned", [
    ("clubs_no_odds.joblib", MARKET_PREFIXES),
    ("clubs_hist_odds.joblib", ("f_odds_",)),
    ("intl.joblib", MARKET_PREFIXES),
])
def test_saved_model_features(filename, banned):
    """Збережені моделі не беруть заборонених для свого набору колонок (і всі їхні ознаки мають префікс f_)."""
    model = joblib.load(require_model(filename))
    bad = [c for c in model.features if c.startswith(banned) or not c.startswith("f_")]
    assert not bad, f"{filename}: недозволені ознаки {bad[:10]}"
