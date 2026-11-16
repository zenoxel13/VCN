"""Спільні фікстури тестів: завантаження побудованих parquet і моделей (або pytest.skip)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

TRACKS = ("clubs", "intl")


def read_processed(name: str) -> pd.DataFrame:
    """Читає data/processed/<name>.parquet або пропускає тест із поясненням."""
    path = config.PROCESSED / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"Немає {path.name} — спершу запустіть python -m src.run_all (або src.features)")
    return pd.read_parquet(path)


def require_model(filename: str) -> Path:
    """Шлях до файлу моделі в models/ або skip, якщо модель ще не навчено."""
    path = config.MODELS / filename
    if not path.exists():
        pytest.skip(f"Немає models/{filename} — спершу запустіть python -m src.train")
    return path


@pytest.fixture(scope="session")
def features_tables() -> dict[str, pd.DataFrame]:
    """Таблиці ознак обох треків (читаються один раз на сесію)."""
    return {}


@pytest.fixture(params=TRACKS)
def track_features(request, features_tables) -> tuple[str, pd.DataFrame]:
    """(трек, таблиця ознак) — тест проганяється для клубів і для збірних."""
    track = request.param
    if track not in features_tables:
        features_tables[track] = read_processed(f"{track}_features")
    return track, features_tables[track]


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Усі колонки-ознаки (префікс f_)."""
    return [c for c in df.columns if c.startswith("f_")]


def assert_rows_equal(a: pd.Series, b: pd.Series, cols: list[str], msg: str) -> None:
    """Перевіряє, що два рядки ознак рівні (NaN == NaN, числа — з допуском 1e-9)."""
    bad = []
    for c in cols:
        x, y = a[c], b[c]
        if pd.isna(x) and pd.isna(y):
            continue
        if pd.isna(x) or pd.isna(y):
            bad.append(c)
        elif isinstance(x, str) or isinstance(y, str):
            if str(x) != str(y):
                bad.append(c)
        elif abs(float(x) - float(y)) > 1e-9:
            bad.append(c)
    assert not bad, f"{msg}: розбіжність у {bad[:10]}"
