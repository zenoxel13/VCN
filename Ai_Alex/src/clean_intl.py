"""Очищення даних матчів національних збірних (martj42/international_results).

Результат: data/processed/intl_matches.parquet — ПОВНА історія з 1872 року
(потрібна для послідовного Elo), один рядок на матч, відсортовано за датою.
"""
import warnings

import pandas as pd
import requests

from src import config

OUT_PATH = config.PROCESSED / "intl_matches.parquet"
GOALS_PATH = config.PROCESSED / "intl_goals.parquet"
GOALSCORERS_RAW = config.RAW / "goalscorers.csv"
GOALSCORERS_URL = ("https://raw.githubusercontent.com/martj42/international_results/"
                   "refs/heads/master/goalscorers.csv")
LATE_MINUTE = 75                 # «пізній» гол — після 75-ї хвилини
SECOND_HALF_MINUTE = 45

FRIENDLY = "Friendly"
TOURNAMENT_TYPES = ["friendly", "qualifier", "final", "other"]
# Фінальні частини великих турнірів (точні назви з results.csv)
MAJOR_FINALS = {
    "FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "Confederations Cup", "UEFA Nations League",
    "CONCACAF Nations League", "CONCACAF Championship", "Oceania Nations Cup",
}


def tournament_type(name: str) -> str:
    """Тип турніру за назвою: friendly / qualifier / final / other."""
    if name == FRIENDLY:
        return "friendly"
    if "qualification" in name.lower():
        return "qualifier"
    if name in MAJOR_FINALS:
        return "final"
    return "other"


def result_label(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    """Ціль H/D/A за рахунком (серія пенальті не враховується → нічия D)."""
    diff = home_goals - away_goals
    return pd.Series(pd.NA, index=diff.index, dtype="object").mask(diff > 0, "H") \
        .mask(diff == 0, "D").mask(diff < 0, "A")


def normalize_names(df: pd.DataFrame, cols: list[str], former: pd.DataFrame) -> int:
    """Замінює колишні назви збірних на поточні (у межах дат дії старої назви).

    Змінює df на місці; повертає кількість замінених значень.
    """
    replaced = 0
    for row in former.itertuples(index=False):
        in_period = df["date"].between(row.start_date, row.end_date)
        for col in cols:
            mask = in_period & (df[col] == row.former)
            replaced += int(mask.sum())
            df.loc[mask, col] = row.current
    return replaced


def load_former_names() -> pd.DataFrame:
    """Читає former_names.csv з датами як datetime."""
    former = pd.read_csv(config.RAW / "former_names.csv")
    former["start_date"] = pd.to_datetime(former["start_date"])
    former["end_date"] = pd.to_datetime(former["end_date"])
    return former


def load_shootouts(former: pd.DataFrame) -> pd.DataFrame:
    """Читає серії пенальті з нормалізованими назвами (лише мета/EDA)."""
    so = pd.read_csv(config.RAW / "shootouts.csv")
    so["date"] = pd.to_datetime(so["date"])
    normalize_names(so, ["home_team", "away_team", "winner"], former)
    so = so.drop_duplicates(["date", "home_team", "away_team"])
    return so.rename(columns={"home_team": "home", "away_team": "away",
                              "winner": "shootout_winner"})[
        ["date", "home", "away", "shootout_winner"]]


def download_goalscorers() -> bool:
    """Завантажує goalscorers.csv (пропуск, якщо файл є). Повертає, чи файл доступний."""
    if GOALSCORERS_RAW.exists():
        return True
    try:
        resp = requests.get(GOALSCORERS_URL, timeout=60)
        resp.raise_for_status()
        GOALSCORERS_RAW.write_bytes(resp.content)
        return True
    except requests.RequestException as err:
        warnings.warn(f"goalscorers.csv недоступний ({err}) — ознаки бомбардирів будуть NaN")
        return False


def goals_by_team(former: pd.DataFrame) -> pd.DataFrame:
    """Агрегати голів команди в матчі з goalscorers.csv (лише мета для ковзних ознак).

    Колонки: date, home, away, team, gs_goals, gs_pen, gs_second_half, gs_late,
    gs_first_min, gs_top_scorer (голи найрезультативнішого гравця в матчі).
    Автоголи не зараховуються жодному бомбардиру, але входять у голи команди.
    """
    gs = pd.read_csv(GOALSCORERS_RAW)
    gs["date"] = pd.to_datetime(gs["date"])
    normalize_names(gs, ["home_team", "away_team", "team"], former)
    gs = gs.rename(columns={"home_team": "home", "away_team": "away"})
    gs["minute"] = pd.to_numeric(gs["minute"], errors="coerce")
    for col in ("own_goal", "penalty"):
        gs[col] = gs[col].astype(str).str.upper().eq("TRUE")
    gs["second_half"] = gs["minute"] > SECOND_HALF_MINUTE
    gs["late"] = gs["minute"] > LATE_MINUTE
    keys = ["date", "home", "away", "team"]
    agg = gs.groupby(keys).agg(gs_goals=("minute", "size"), gs_pen=("penalty", "sum"),
                               gs_second_half=("second_half", "sum"), gs_late=("late", "sum"),
                               gs_first_min=("minute", "min"))
    top = (gs[~gs["own_goal"]].groupby(keys + ["scorer"]).size()
           .groupby(level=keys).max().rename("gs_top_scorer"))
    return agg.join(top).fillna({"gs_top_scorer": 0}).reset_index()


def clean() -> pd.DataFrame:
    """Повний цикл очищення; повертає таблицю матчів."""
    raw = pd.read_csv(config.RAW / "results.csv")
    print(f"Сирих рядків: {len(raw)}")
    df = raw.rename(columns={"home_team": "home", "away_team": "away",
                             "home_score": "home_goals", "away_score": "away_goals",
                             "tournament": "competition"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_goals", "away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["date", "home_goals", "away_goals"])
    print(f"Викинуто без дати/рахунку (майбутні/NA): {n_before - len(df)}")

    former = load_former_names()
    print(f"Замінено колишніх назв: {normalize_names(df, ['home', 'away'], former)}")

    n_before = len(df)
    df = df.drop_duplicates(["date", "home", "away"], keep="first")
    print(f"Видалено дублікатів (date, home, away): {n_before - len(df)}")

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["tournament_type"] = df["competition"].map(tournament_type)
    df["target"] = result_label(df["home_goals"], df["away_goals"])
    df["y"] = df["target"].map({c: i for i, c in enumerate(config.CLASSES)}).astype(int)

    df = df.merge(load_shootouts(former), on=["date", "home", "away"], how="left")
    df["was_shootout"] = df["shootout_winner"].notna()

    # стабільне сортування: дата, потім порядок у вихідному файлі
    df = df.reset_index(drop=True).sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))
    cols = ["match_id", "date", "home", "away", "home_goals", "away_goals", "target", "y",
            "competition", "tournament_type", "city", "country", "neutral",
            "was_shootout", "shootout_winner"]
    return df[cols]


def report(df: pd.DataFrame) -> None:
    """Друкує санітарні перевірки очищеної таблиці."""
    print(f"Розмір: {df.shape}")
    print(f"Дублікати (date, home, away): {df.duplicated(['date', 'home', 'away']).sum()}")
    print(f"Діапазон дат: {df['date'].min().date()} … {df['date'].max().date()}")
    print("Пропуски:\n" + df.isna().sum()[lambda s: s > 0].to_string())
    print("Розподіл target:\n" + df["target"].value_counts(normalize=True).round(3).to_string())
    print("Типи турнірів:\n" + df["tournament_type"].value_counts().to_string())
    print(f"Серій пенальті приєднано: {df['was_shootout'].sum()}")


def main() -> None:
    """Точка входу: очищення → перевірки → parquet."""
    df = clean()
    report(df)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Збережено: {OUT_PATH}")
    if download_goalscorers():
        goals = goals_by_team(load_former_names())
        covered = df.merge(goals[["date", "home", "away"]].drop_duplicates(),
                           on=["date", "home", "away"])
        print(f"goalscorers: {len(goals)} записів (матч, команда); "
              f"покрито матчів: {len(covered)} з {len(df)}")
        goals.to_parquet(GOALS_PATH, index=False)
        print(f"Збережено: {GOALS_PATH}")


if __name__ == "__main__":
    main()
