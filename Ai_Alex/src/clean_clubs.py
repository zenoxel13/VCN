"""Очищення клубних матчів: data/raw/Matches.csv → data/processed/clubs_matches.parquet.

Кроки: типи й дати, фільтр за датою, викидання рядків без команд/результату, дедуплікація,
узгодження FTResult із голами, нереальні коефіцієнти → NaN, сезон і match_id.
"""
import numpy as np
import pandas as pd

from src import config

RAW_FILE = config.RAW / "Matches.csv"
OUT_FILE = config.PROCESSED / "clubs_matches.parquet"

# Перейменування ключових колонок у «мета»-назви контракту; решта лишаються як у джерелі.
RENAME = {
    "Division": "division",
    "MatchDate": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHome": "home_goals",
    "FTAway": "away_goals",
    "FTResult": "target",
}
DEDUP_KEYS = ["division", "date", "home", "away"]
# Десятковий коефіцієнт <= 1 означає нульовий або від'ємний виграш — таке неможливо.
MIN_VALID_ODD = 1.0
ODDS_COLS = ["OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
             "Over25", "Under25", "MaxOver25", "MaxUnder25", "HandiHome", "HandiAway"]
SEASON_START_MONTH = 7  # сезон липень–червень

# Джерело з часом змінило написання кількох клубів — без зведення до однієї назви історія команди
# (форма, Elo, очні зустрічі) обнулялася б посеред тестового періоду. Ключ — варіант, значення — канонічна назва.
TEAM_ALIASES = {
    "Nott'm Forest": "Nottm Forest",
    "M'gladbach": "MGladbach",
    "HamKam": "Ham-Kam",
    "Colon Santa Fe": "Colon Santa FE",
}


def normalize_team(names: pd.Series) -> pd.Series:
    """Прибирає зайві пробіли й зводить відомі варіанти написання клубу до канонічної назви."""
    return names.astype("string").str.strip().replace(TEAM_ALIASES)


def load_raw() -> pd.DataFrame:
    """Читає сирий CSV і приводить типи (дата, рядки, числа)."""
    df = pd.read_csv(RAW_FILE, low_memory=False).rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("division", "home", "away", "target"):
        df[col] = df[col].astype("string").str.strip()
    for col in ("home", "away"):
        df[col] = normalize_team(df[col])
    numeric = [c for c in df.columns if c not in ("division", "date", "home", "away",
                                                  "target", "MatchTime", "HTResult")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def result_from_goals(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    """Повертає H/D/A за рахунком."""
    res = np.select([home_goals > away_goals, home_goals < away_goals], ["H", "A"], "D")
    return pd.Series(res, index=home_goals.index, dtype="string")


def season_of(dates: pd.Series) -> pd.Series:
    """Рік початку сезону (липень–червень): 2023-08 → 2023, 2024-03 → 2023."""
    return (dates.dt.year - (dates.dt.month < SEASON_START_MONTH)).astype(int)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Застосовує всі правила очищення й повертає відсортовану таблицю з match_id і season."""
    n0 = len(df)
    df = df[df["date"] >= pd.Timestamp(config.MIN_DATE)]
    df = df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    df = df[df["home"] != df["away"]]
    print(f"Викинуто за датою/пропусками/home==away: {n0 - len(df)}")

    n_dup = df.duplicated(DEDUP_KEYS).sum()
    df = df.drop_duplicates(DEDUP_KEYS, keep="first")
    print(f"Дублікатів ({'+'.join(DEDUP_KEYS)}): {n_dup}")

    by_goals = result_from_goals(df["home_goals"], df["away_goals"])
    n_bad = (df["target"] != by_goals).fillna(True).sum()
    print(f"FTResult не узгоджений із голами (виправлено за голами): {n_bad}")
    df = df.assign(target=by_goals)

    odds = df[ODDS_COLS]
    n_bad_odds = int((odds <= MIN_VALID_ODD).sum().sum())
    df[ODDS_COLS] = odds.where(odds > MIN_VALID_ODD)
    print(f"Нереальних коефіцієнтів (<= {MIN_VALID_ODD}) → NaN: {n_bad_odds}")

    df = df.sort_values(["date", "division", "home"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "match_id", np.arange(len(df), dtype="int64"))
    df.insert(2, "season", season_of(df["date"]))
    df["y"] = df["target"].map({c: i for i, c in enumerate(config.CLASSES)}).astype(int)
    return df


def report(df: pd.DataFrame) -> None:
    """Друкує санітарні перевірки очищеної таблиці."""
    print(f"Розмір: {df.shape}")
    print(f"Дублікатів після очищення: {df.duplicated(DEDUP_KEYS).sum()}")
    print(f"Діапазон дат: {df['date'].min().date()} … {df['date'].max().date()}")
    print(f"Ліг: {df['division'].nunique()}, команд: {pd.concat([df['home'], df['away']]).nunique()}")
    print("Розподіл результатів:", df["target"].value_counts(normalize=True).round(3).to_dict())
    na = df.isna().mean().round(3)
    print("Частка пропусків (колонки з NaN):")
    print(na[na > 0].to_string())


def main() -> None:
    """Точка входу: сирий CSV → очищений parquet."""
    df = clean(load_raw())
    report(df)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Збережено: {OUT_FILE}")


if __name__ == "__main__":
    main()
