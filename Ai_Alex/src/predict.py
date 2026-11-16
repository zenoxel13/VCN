"""Прогноз результату одного матчу (H/D/A) за збереженими моделями.

Приклад CLI (із кореня проєкту):
    python -m src.predict "Arsenal" "Chelsea" 2026-11-20 E0
    python -m src.predict "Ukraine" "England" 2026-11-20 "FIFA World Cup qualification"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import config, features_clubs, features_intl

TRACK_MODULES = {"clubs": features_clubs, "intl": features_intl}
# Моделі: (трек, варіант) → файл. Варіанти клубів: no_odds — сувора (без жодних коефіцієнтів),
# hist_odds — з історією ринку в минулих матчах, with_odds — з коефіцієнтами цього матчу.
MODEL_FILES = {
    ("clubs", "no_odds"): "clubs_no_odds.joblib",
    ("clubs", "hist_odds"): "clubs_hist_odds.joblib",
    ("clubs", "with_odds"): "clubs_with_odds.joblib",
    ("intl", "no_odds"): "intl.joblib",
}
MODEL_LABELS = {
    ("clubs", "no_odds"): "Сувора модель (без жодних коефіцієнтів БК)",
    ("clubs", "hist_odds"): "Модель з історією ринку (коефіцієнти лише минулих матчів команд)",
    ("clubs", "with_odds"): "Модель з коефіцієнтами БК цього матчу",
    ("intl", "no_odds"): "Модель для збірних",
}
MATCH_FILES = {"clubs": "clubs_matches.parquet", "intl": "intl_matches.parquet"}
FEATURE_FILES = {"clubs": "clubs_features.parquet", "intl": "intl_features.parquet"}


@dataclass
class Prediction:
    """Результат прогнозу: ймовірності, рядок ознак і назва використаної моделі."""
    proba: dict[str, float]
    features: pd.DataFrame
    model_name: str
    track: str
    variant: str = "no_odds"

    @property
    def model_label(self) -> str:
        """Людська назва використаної моделі."""
        return MODEL_LABELS.get((self.track, self.variant), self.model_name)


# ---------------------------------------------------------------- довідники (кешовані)

@lru_cache(maxsize=None)
def competitions(track: str) -> tuple[str, ...]:
    """Ліги (клуби) або турніри (збірні), відомі треку."""
    return tuple(TRACK_MODULES[track].list_competitions())


@lru_cache(maxsize=None)
def teams(track: str, competition: str | None = None) -> tuple[str, ...]:
    """Команди треку; для клубів із competition — ті, що грали в лізі останні сезони.

    Якщо ліга давно не оновлювалась (за останні сезони порожньо) — усі її команди з історії.
    """
    found = TRACK_MODULES[track].list_teams(competition)
    if not found and competition is not None:
        h = match_history(track)
        h = h[h["competition"] == competition]
        found = sorted(set(h["home"]) | set(h["away"]))
    return tuple(found)


def detect_track(competition: str) -> str:
    """'clubs', якщо competition — код клубної ліги, інакше 'intl'."""
    return "clubs" if competition in competitions("clubs") else "intl"


# ---------------------------------------------------------------- моделі

def _mtime(path) -> float:
    """Час зміни файлу (0, якщо його немає) — ключ кешу, щоб після перенавчання бралися нові файли."""
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=8)
def _load_model_file(path, mtime: float):
    """Завантажує joblib-файл; кеш залежить від mtime, тож оновлений файл перечитується."""
    return joblib.load(path)


def load_model(filename: str):
    """FinalModel з models/ (кешується, доки файл не змінився)."""
    path = config.MODELS / filename
    if not path.exists():
        raise ValueError(f"Файл моделі {path.name} не знайдено — спершу запустіть python -m src.run_all")
    return _load_model_file(path, _mtime(path))


def models_version() -> tuple[float, ...]:
    """Відбиток стану моделей і таблиць ознак — для ключів кешу в застосунку."""
    files = [config.MODELS / f for f in MODEL_FILES.values()]
    files += [config.PROCESSED / f for f in FEATURE_FILES.values()]
    return tuple(_mtime(f) for f in files)


def model_exists(track: str, variant: str) -> bool:
    """Чи є файл моделі для (трек, варіант)."""
    name = MODEL_FILES.get((track, variant))
    return bool(name) and (config.MODELS / name).exists()


def choose_variant(track: str, with_odds: bool, strict: bool = False) -> str:
    """Варіант моделі: введено коефіцієнти → with_odds; інакше hist_odds (якщо не strict і файл є); інакше no_odds."""
    if track == "clubs" and with_odds and model_exists(track, "with_odds"):
        return "with_odds"
    if track == "clubs" and not strict and model_exists(track, "hist_odds"):
        return "hist_odds"
    return "no_odds"


def choose_model(track: str, with_odds: bool, strict: bool = False):
    """FinalModel за правилами choose_variant."""
    return load_model(MODEL_FILES[(track, choose_variant(track, with_odds, strict))])


def warmup() -> None:
    """Прогріває кеші історії (Elo, pi-рейтинги тощо) одним фіктивним прогнозом на кожен трек."""
    for track in TRACK_MODULES:
        team_list = teams(track)
        if len(team_list) >= 2:
            TRACK_MODULES[track].build_match_row(team_list[0], team_list[1], pd.Timestamp.today(),
                                                 competitions(track)[0])


@lru_cache(maxsize=None)
def venue_countries() -> tuple[str, ...]:
    """Країни, де грали збірні (для вибору місця проведення на нейтральному полі)."""
    m = pd.read_parquet(config.PROCESSED / MATCH_FILES["intl"], columns=["country"])
    return tuple(sorted(m["country"].dropna().astype(str).unique()))


# ---------------------------------------------------------------- перевірка вводу

def _validate(home: str, away: str, competition: str, track: str) -> None:
    """Кидає ValueError з поясненням українською, якщо ввід некоректний."""
    if track not in TRACK_MODULES:
        raise ValueError(f"Невідомий трек «{track}» — очікується clubs або intl")
    if not home or not away:
        raise ValueError("Вкажіть обидві команди")
    if home == away:
        raise ValueError("Команди мають бути різними")
    if track == "clubs" and competition not in competitions("clubs"):
        raise ValueError(f"Невідома ліга «{competition}». Доступні: {', '.join(competitions('clubs'))}")
    known = set(teams(track))
    for team in (home, away):
        if team not in known:
            raise ValueError(f"Невідома команда «{team}» — її немає в історичних даних")


# Сума обернених коефіцієнтів справжньої лінії 1X2: 1 + маржа букмекера (історично до ~15%).
MIN_IMPLIED_SUM = 0.95
MAX_IMPLIED_SUM = 1.35


def _parse_odds(odds) -> tuple[float, float, float] | None:
    """Перевіряє коефіцієнти БК: три числа > 1, або None."""
    if odds is None:
        return None
    odds = tuple(float(o) for o in odds)
    if len(odds) != 3 or any(o <= 1.0 for o in odds):
        raise ValueError("Коефіцієнти БК мають бути трьома числами більшими за 1 (П1, Х, П2)")
    implied = sum(1.0 / o for o in odds)
    if not MIN_IMPLIED_SUM <= implied <= MAX_IMPLIED_SUM:
        raise ValueError(
            f"Коефіцієнти не схожі на справжню лінію 1X2: сума обернених = {implied:.2f} "
            f"(у букмекерів вона трохи більша за 1, допустимо {MIN_IMPLIED_SUM}–{MAX_IMPLIED_SUM})")
    return odds


# ---------------------------------------------------------------- прогноз

def predict_details(home: str, away: str, date, competition: str, track: str | None = None,
                    odds=None, neutral: bool = False, strict: bool = False,
                    country: str | None = None) -> Prediction:
    """Прогноз разом із рядком ознак і назвою моделі (для показу в застосунку).

    strict — для клубів без коефіцієнтів брати сувору модель замість моделі з історією ринку;
    country — країна проведення матчу збірних (необов'язково).
    """
    track = track or detect_track(competition)
    _validate(home, away, competition, track)
    odds = _parse_odds(odds) if track == "clubs" else None
    extra = {"odds": odds} if track == "clubs" else {"neutral": bool(neutral), "country": country or None}
    row = TRACK_MODULES[track].build_match_row(home, away, pd.Timestamp(date), competition, **extra)
    variant = choose_variant(track, with_odds=odds is not None, strict=strict)
    model = load_model(MODEL_FILES[(track, variant)])
    proba = model.predict_proba(row)[0]
    if track == "intl" and neutral:
        proba = _symmetrize_neutral(proba, model, away, home, date, competition, extra)
    proba = proba / proba.sum()
    return Prediction(proba=dict(zip(config.CLASSES, map(float, proba))),
                      features=row, model_name=model.name, track=track, variant=variant)


def _symmetrize_neutral(proba, model, first: str, second: str, date, competition: str, extra: dict):
    """На нейтральному полі «господаря» немає, тож прогноз не має залежати від порядку команд.

    Модель навчена на таблиці, де одна з команд записана першою, і для дуже сильних пар дає різні
    відповіді для A–B та B–A. Усереднюємо прогноз із дзеркальним (матч B–A з переставленими H та A).
    """
    mirrored_row = TRACK_MODULES["intl"].build_match_row(first, second, pd.Timestamp(date), competition, **extra)
    mirrored = model.predict_proba(mirrored_row)[0]
    return (proba + mirrored[::-1]) / 2.0


def predict(home: str, away: str, date, competition: str, track: str | None = None,
            odds=None, neutral: bool = False, strict: bool = False,
            country: str | None = None) -> dict[str, float]:
    """Ймовірності {"H", "D", "A"} (сума = 1) для матчу home–away на дату date."""
    return predict_details(home, away, date, competition, track, odds, neutral, strict, country).proba


def market_proba(odds) -> dict[str, float]:
    """Імовірності букмекера без маржі: 1/коефіцієнт, нормовані на суму."""
    inv = [1.0 / o for o in _parse_odds(odds)]
    return dict(zip(config.CLASSES, (v / sum(inv) for v in inv)))


# ---------------------------------------------------------------- історія матчів

@lru_cache(maxsize=None)
def match_history(track: str) -> pd.DataFrame:
    """Зіграні матчі треку в єдиному форматі (date, competition, home, away, голи, target)."""
    m = pd.read_parquet(config.PROCESSED / MATCH_FILES[track])
    if track == "clubs":
        m = m.rename(columns={"division": "competition"})
    cols = ["date", "competition", "home", "away", "home_goals", "away_goals", "target"]
    m = m[cols].dropna(subset=["home_goals", "away_goals"])
    m["competition"] = m["competition"].astype(str)
    return m.sort_values("date", kind="stable").reset_index(drop=True)


def _score(home_goals, away_goals) -> pd.Series:
    """Рахунок рядком «2:1»."""
    return home_goals.astype(int).astype(str) + ":" + away_goals.astype(int).astype(str)


def recent_matches(track: str, team: str, date, n: int = 5) -> pd.DataFrame:
    """Останні n матчів команди строго до date (з погляду цієї команди)."""
    h = match_history(track)
    h = h[(h["date"] < pd.Timestamp(date)) & ((h["home"] == team) | (h["away"] == team))].tail(n).iloc[::-1]
    at_home = h["home"] == team
    gf = h["home_goals"].where(at_home, h["away_goals"])
    ga = h["away_goals"].where(at_home, h["home_goals"])
    result = np.select([gf > ga, gf == ga], ["В", "Н"], default="П")
    return pd.DataFrame({
        "Дата": h["date"].dt.strftime("%d.%m.%Y"),
        "Суперник": h["away"].where(at_home, h["home"]),
        "Поле": np.where(at_home, "вдома", "у гостях"),
        "Рахунок": _score(gf, ga),
        "Підсумок": result,
        "Турнір": h["competition"],
    }).reset_index(drop=True)


def head_to_head(track: str, team_a: str, team_b: str, date, n: int = 10) -> pd.DataFrame:
    """Останні n очних зустрічей двох команд строго до date."""
    h = match_history(track)
    pair = ((h["home"] == team_a) & (h["away"] == team_b)) | ((h["home"] == team_b) & (h["away"] == team_a))
    h = h[pair & (h["date"] < pd.Timestamp(date))].tail(n).iloc[::-1]
    return pd.DataFrame({
        "Дата": h["date"].dt.strftime("%d.%m.%Y"),
        "Господарі": h["home"], "Гості": h["away"],
        "Рахунок": _score(h["home_goals"], h["away_goals"]),
        "Турнір": h["competition"],
    }).reset_index(drop=True)


# ---------------------------------------------------------------- перевірка на історії (тестовий період)

def test_features(track: str) -> pd.DataFrame:
    """Рядки таблиці ознак тестового періоду (date >= TEST_START); ознаки пораховані лише з минулого."""
    path = config.PROCESSED / FEATURE_FILES[track]
    return _read_test_features(path, _mtime(path))


@lru_cache(maxsize=4)
def _read_test_features(path, mtime: float) -> pd.DataFrame:
    """Читає тестовий період з parquet (кеш залежить від mtime файлу)."""
    df = pd.read_parquet(path, filters=[("date", ">=", pd.Timestamp(config.TEST_START))])
    df["competition"] = df["competition"].astype(str)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def test_competitions(track: str) -> list[str]:
    """Ліги/турніри тестового періоду за спаданням кількості матчів."""
    return test_features(track)["competition"].value_counts().index.tolist()


def backtest(track: str, competition: str, n: int = 20, variant: str = "hist_odds") -> pd.DataFrame:
    """Прогнози моделі variant (без коефіцієнтів цього матчу) для останніх n матчів турніру в тестовому періоді.

    Колонки: date, home, away, home_goals, away_goals, target, p_H/p_D/p_A, pred, hit;
    для клубів ще m_H/m_D/m_A — імовірності букмекера без маржі (NaN, якщо коефіцієнтів немає).
    """
    df = test_features(track)
    df = df[df["competition"] == competition].tail(n)
    out = df[["date", "home", "away", "home_goals", "away_goals", "target"]].copy()
    if df.empty:
        return out
    if not model_exists(track, variant):
        variant = "no_odds"
    proba = load_model(MODEL_FILES[(track, variant)]).predict_proba(df)
    for i, cls in enumerate(config.CLASSES):
        out[f"p_{cls}"] = proba[:, i]
    out["pred"] = np.array(config.CLASSES)[proba.argmax(axis=1)]
    out["hit"] = out["pred"] == out["target"]
    odds_cols = [f"f_odds_p_{side}" for side in ("home", "draw", "away")]
    if all(c in df.columns for c in odds_cols):
        for cls, col in zip(config.CLASSES, odds_cols):
            out[f"m_{cls}"] = df[col].to_numpy()
    return out.iloc[::-1].reset_index(drop=True)


# ---------------------------------------------------------------- CLI

def _parse_args() -> argparse.Namespace:
    """Аргументи командного рядка."""
    p = argparse.ArgumentParser(description="Прогноз результату футбольного матчу (H/D/A)")
    p.add_argument("home", help="господарі")
    p.add_argument("away", help="гості")
    p.add_argument("date", help="дата матчу, РРРР-ММ-ДД")
    p.add_argument("competition", help="код ліги (E0, SP1, …) або назва турніру збірних")
    p.add_argument("--track", choices=list(TRACK_MODULES), default=None)
    p.add_argument("--odds", nargs=3, type=float, metavar=("П1", "Х", "П2"), default=None)
    p.add_argument("--neutral", action="store_true", help="нейтральне поле (збірні)")
    p.add_argument("--country", default=None, help="країна проведення (збірні)")
    p.add_argument("--strict", action="store_true", help="клуби: сувора модель без історії ринку")
    return p.parse_args()


def main() -> None:
    """Друкує ймовірності прогнозу для матчу з командного рядка."""
    a = _parse_args()
    try:
        res = predict_details(a.home, a.away, a.date, a.competition, a.track, a.odds, a.neutral, a.strict, a.country)
    except ValueError as err:
        raise SystemExit(f"Помилка: {err}")
    labels = {"H": "Перемога господарів", "D": "Нічия", "A": "Перемога гостей"}
    print(f"{a.home} — {a.away}, {a.date}, {a.competition} (трек {res.track})")
    print(f"Модель: {res.model_label} [{res.model_name}]")
    for k, p in res.proba.items():
        print(f"  {labels[k]:<20} {p:6.1%}   чесний коеф. {1 / p:5.2f}")


if __name__ == "__main__":
    main()
