"""Спільні будівельні блоки моделювання для обох треків (клуби / збірні).

Тут: розбиття за часом, вибір ознак за префіксами, метрики, базові моделі,
логістична регресія, LightGBM, калібратори та клас FinalModel для демо.
Усі класи — верхнього рівня модуля, тож об'єкти серіалізуються через joblib/pickle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src import config

N_CLASSES = len(config.CLASSES)
EPS = 1e-6

# Ознаки, тимчасово виключені з моделей (напр. через підозру на витік). Кожне виключення — у DECISIONS.md.
EXCLUDED_FEATURES: list[str] = []

ODDS_PROBA_COLS = ["f_odds_p_home", "f_odds_p_draw", "f_odds_p_away"]


# ----------------------------------------------------------------------------- дані
def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Розбиває таблицю за датою: train < VAL_START <= val < TEST_START <= test."""
    date = pd.to_datetime(df["date"])
    val_start, test_start = pd.Timestamp(config.VAL_START), pd.Timestamp(config.TEST_START)
    return {
        "train": df[date < val_start].reset_index(drop=True),
        "val": df[(date >= val_start) & (date < test_start)].reset_index(drop=True),
        "test": df[date >= test_start].reset_index(drop=True),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Усі колонки-ознаки (префікс f_) за винятком EXCLUDED_FEATURES."""
    return [c for c in df.columns if c.startswith("f_") and c not in EXCLUDED_FEATURES]


# Набір ознак = усі f_ мінус ознаки з виключеними префіксами (порядок — від найсуворішого).
SET_EXCLUDED_PREFIXES = {
    "no_odds": ("f_odds_", "f_hodds_"),   # суворий: жодної ринкової інформації
    "hist_odds": ("f_odds_",),            # лише ринкова оцінка команд у МИНУЛИХ матчах
    "with_odds": (),                      # усе, включно з коефіцієнтами цього матчу
}


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Набори ознак треку за SET_EXCLUDED_PREFIXES; набір, що не відрізняється від попереднього, пропускається."""
    cols = feature_columns(df)
    sets: dict[str, list[str]] = {}
    for name, prefixes in SET_EXCLUDED_PREFIXES.items():
        feats = [c for c in cols if not c.startswith(prefixes)] if prefixes else list(cols)
        if not sets or len(feats) > len(list(sets.values())[-1]):
            sets[name] = feats
    return sets


def recency_weights(dates: pd.Series, half_life: float | None) -> np.ndarray | None:
    """Ваги свіжості 0.5 ** (вік у роках / півперіод), вік — від найсвіжішої дати вибірки; None — без ваг."""
    if half_life is None:
        return None
    age_years = (pd.to_datetime(dates).max() - pd.to_datetime(dates)).dt.days.to_numpy() / 365.25
    w = 0.5 ** (age_years / half_life)
    return w / w.mean()  # середня вага 1 — масштаб регуляризації не змінюється


def categorical_columns(features: list[str]) -> list[str]:
    """Категоріальні ознаки (префікс f_cat_)."""
    return [c for c in features if c.startswith("f_cat_")]


def numeric_columns(features: list[str]) -> list[str]:
    """Числові ознаки (усе, що не f_cat_)."""
    return [c for c in features if not c.startswith("f_cat_")]


# ----------------------------------------------------------------------------- метрики
def _one_hot(y: np.ndarray) -> np.ndarray:
    """Перетворює мітки 0..2 на one-hot матрицю (n, 3)."""
    return np.eye(N_CLASSES)[np.asarray(y, dtype=int)]


def brier_score(y: np.ndarray, proba: np.ndarray) -> float:
    """Мультикласовий Brier: середня сума квадратів відхилень по трьох класах."""
    return float(np.mean(np.sum((proba - _one_hot(y)) ** 2, axis=1)))


def rps_score(y: np.ndarray, proba: np.ndarray) -> float:
    """Ranked Probability Score для впорядкованих класів H < D < A (менше — краще)."""
    cum_p = np.cumsum(proba, axis=1)[:, :-1]
    cum_o = np.cumsum(_one_hot(y), axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (N_CLASSES - 1)))


def compute_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Повний набір метрик: accuracy, log-loss, Brier, RPS, macro-F1."""
    y = np.asarray(y, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=list(range(N_CLASSES)))),
        "brier": brier_score(y, proba),
        "rps": rps_score(y, proba),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(N_CLASSES)))),
    }


def class_frequencies(y: np.ndarray) -> np.ndarray:
    """Частоти класів [H, D, A] у вибірці."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=N_CLASSES).astype(float)
    return counts / counts.sum()


# ----------------------------------------------------------------------------- базові моделі
class PriorModel:
    """Базова «завжди господарі»: ймовірності = частоти класів train (argmax — H)."""

    def __init__(self) -> None:
        self.features: list[str] = []
        self.prior = np.full(N_CLASSES, 1.0 / N_CLASSES)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "PriorModel":
        """Запам'ятовує частоти класів."""
        self.prior = class_frequencies(y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Однаковий розподіл для кожного рядка."""
        return np.tile(self.prior, (len(df), 1))


class OddsModel:
    """Базова «коефіцієнти БК»: f_odds_p_* напряму як ймовірності; без коефіцієнтів — частоти train."""

    def __init__(self) -> None:
        self.features = list(ODDS_PROBA_COLS)
        self.prior = np.full(N_CLASSES, 1.0 / N_CLASSES)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "OddsModel":
        """Запам'ятовує частоти класів як запасний варіант."""
        self.prior = class_frequencies(y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Нормовані ймовірності БК; рядки з NaN отримують частоти train."""
        proba = df[self.features].to_numpy(dtype=float, copy=True)
        missing = np.isnan(proba).any(axis=1)
        proba[missing] = self.prior
        return proba / proba.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- навчувані моделі
class AnchorModel:
    """«Сильний стартовий прогноз»: ринкові ймовірності f_odds_p_* (якщо use_odds і вони є),
    інакше — прогноз Elo-логрегресії. Використовується як init_score LightGBM і як ознака логрегресії."""

    def __init__(self, elo_model, use_odds: bool) -> None:
        self.elo_model = elo_model
        self.use_odds = use_odds
        self.features = list(elo_model.features) + (list(ODDS_PROBA_COLS) if use_odds else [])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Ймовірності [H, D, A] стартового прогнозу."""
        proba = np.asarray(self.elo_model.predict_proba(df), dtype=float).copy()
        if self.use_odds and all(c in df.columns for c in ODDS_PROBA_COLS):
            odds = df[ODDS_PROBA_COLS].to_numpy(dtype=float, copy=True)
            ok = ~np.isnan(odds).any(axis=1)
            proba[ok] = odds[ok]
        proba = np.clip(proba, EPS, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)

    def log_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Логарифми ймовірностей (стартові «сирі» бали для softmax)."""
        return np.log(self.predict_proba(df))


ANCHOR_COLS = ["anchor_logp_H", "anchor_logp_D", "anchor_logp_A"]


class LogRegModel:
    """Мультиноміальна логістична регресія: медіана train → StandardScaler; f_cat_ → one-hot.
    Якщо задано anchor — його лог-ймовірності додаються як три числові ознаки."""

    def __init__(self, features: list[str], C: float = 1.0, anchor: AnchorModel | None = None,
                 half_life: float | None = None, max_iter: int = 300) -> None:
        self.features = list(features)
        self.C = C
        self.anchor = anchor
        self.half_life = half_life    # півперіод ваг свіжості (роки); None — без ваг
        self.max_iter = max_iter
        self.pipeline: Pipeline | None = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводить типи: числові → float, категоріальні → рядки (пропуск = '__NA__')."""
        cols = {c: pd.to_numeric(df[c], errors="coerce").astype(float) for c in numeric_columns(self.features)}
        cols.update({c: df[c].astype(object).where(df[c].notna(), "__NA__").astype(str)
                     for c in categorical_columns(self.features)})
        X = pd.DataFrame(cols, index=df.index)
        if self.anchor is not None:
            X[ANCHOR_COLS] = self.anchor.log_proba(df)
        return X

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LogRegModel":
        """Навчає препроцесинг і логрегресію (з вагами свіжості, якщо задано half_life)."""
        sample_weight = recency_weights(df["date"], self.half_life) if self.half_life else None
        num, cat = numeric_columns(self.features), categorical_columns(self.features)
        if self.anchor is not None:
            num = num + ANCHOR_COLS
        transformers = []
        if num:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), num))
        if cat:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
        self.pipeline = Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter, tol=1e-4)),
        ])
        # BLAS обмежено одним потоком: на 22 ядрах багатопотоковий BLAS у lbfgs «зависав» на хвилини (151 ознака)
        with threadpool_limits(1, user_api="blas"):
            self.pipeline.fit(self._prepare(df), np.asarray(y, dtype=int), clf__sample_weight=sample_weight)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Ймовірності [H, D, A]."""
        with threadpool_limits(1, user_api="blas"):
            return self.pipeline.predict_proba(self._prepare(df))


class LGBMModel:
    """Обгортка над навченим lightgbm.Booster: сама приводить f_cat_ до категорій train."""

    def __init__(self, features: list[str], booster, cat_levels: dict[str, list], params: dict,
                 anchor: AnchorModel | None = None, half_life: float | None = None) -> None:
        self.features = list(features)
        self.booster = booster
        self.cat_levels = cat_levels
        self.params = params
        self.anchor = anchor          # якщо є — бустинг вчить поправку до стартового прогнозу (init_score)
        self.half_life = half_life    # півперіод ваг свіжості (роки), з яким навчено; None — без ваг

    @staticmethod
    def prepare(df: pd.DataFrame, features: list[str], cat_levels: dict[str, list]) -> pd.DataFrame:
        """Таблиця для LightGBM: числові → float, f_cat_ → category з фіксованим набором рівнів."""
        cols = {c: (pd.Categorical(df[c].astype(object), categories=cat_levels[c]) if c in cat_levels
                    else pd.to_numeric(df[c], errors="coerce").astype(float)) for c in features}
        return pd.DataFrame(cols, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Ймовірності [H, D, A] на найкращій ітерації (з урахуванням стартового прогнозу, якщо є)."""
        X = self.prepare(df, self.features, self.cat_levels)
        it = self.booster.best_iteration or None
        if self.anchor is None:
            return self.booster.predict(X, num_iteration=it)
        raw = self.booster.predict(X, num_iteration=it, raw_score=True) + self.anchor.log_proba(df)
        raw -= raw.max(axis=1, keepdims=True)
        proba = np.exp(raw)
        return proba / proba.sum(axis=1, keepdims=True)

    def feature_importance(self) -> pd.DataFrame:
        """Важливість ознак за gain (сума по всіх деревах), відсортована за спаданням."""
        gain = self.booster.feature_importance(importance_type="gain")
        return (pd.DataFrame({"feature": self.booster.feature_name(), "gain": gain})
                .sort_values("gain", ascending=False).reset_index(drop=True))


def category_levels(df: pd.DataFrame, features: list[str]) -> dict[str, list]:
    """Набір категорій кожної f_cat_ ознаки, як у train."""
    levels = {}
    for c in categorical_columns(features):
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            levels[c] = list(df[c].cat.categories)
        else:
            levels[c] = sorted(df[c].dropna().astype(str).unique().tolist())
    return levels


# ----------------------------------------------------------------------------- калібрування
class IdentityCalibrator:
    """Без калібрування."""
    name = "none"

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        """Нічого не вчить."""
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Повертає ймовірності як є."""
        return proba


class IsotonicCalibrator:
    """Ізотонічна регресія окремо по кожному класу (one-vs-rest) з перенормуванням до суми 1."""
    name = "isotonic"

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """Вчить по одній ізотонічній регресії на клас."""
        onehot = _one_hot(y)
        self.models = [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                       .fit(proba[:, k], onehot[:, k]) for k in range(N_CLASSES)]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Калібрує й перенормовує."""
        out = np.column_stack([m.predict(proba[:, k]) for k, m in enumerate(self.models)])
        out = np.clip(out, EPS, None)
        return out / out.sum(axis=1, keepdims=True)


class LogitCalibrator:
    """Мультиноміальна логрегресія на log-ймовірностях (узагальнений Platt scaling)."""
    name = "logit"

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "LogitCalibrator":
        """Вчить логрегресію на log(p)."""
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(self._x(proba), np.asarray(y, int))
        return self

    @staticmethod
    def _x(proba: np.ndarray) -> np.ndarray:
        """Логарифм ймовірностей як вхід калібратора."""
        return np.log(np.clip(proba, EPS, 1.0))

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Калібровані ймовірності."""
        return self.model.predict_proba(self._x(proba))


class TemperatureCalibrator:
    """Температура: p ** (1/T), перенормоване; T > 1 стискає крайні ймовірності. T — сітка за log-loss."""
    name = "temperature"
    GRID = np.round(np.arange(0.8, 1.51, 0.05), 2)

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        """Обирає T з сітки за log-loss."""
        self.T = min(self.GRID, key=lambda t: compute_metrics(y, self._apply(proba, t))["log_loss"])
        return self

    @staticmethod
    def _apply(proba: np.ndarray, t: float) -> np.ndarray:
        out = np.clip(proba, EPS, 1.0) ** (1.0 / t)
        return out / out.sum(axis=1, keepdims=True)

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Застосовує обрану температуру."""
        return self._apply(proba, self.T)


CALIBRATORS = (IdentityCalibrator, IsotonicCalibrator, LogitCalibrator, TemperatureCalibrator)


def choose_calibrator(proba_val: np.ndarray, y_val: np.ndarray) -> tuple[object, dict[str, float]]:
    """Обирає калібратор чесно: вчить на першій (за часом) половині val, оцінює log-loss на другій.

    Переможця перенавчає на всьому val. Повертає (калібратор, {назва: log-loss на 2-й половині}).
    """
    half = len(y_val) // 2
    scores = {}
    for cls in CALIBRATORS:
        cal = cls().fit(proba_val[:half], y_val[:half])
        scores[cls.name] = compute_metrics(y_val[half:], cal.transform(proba_val[half:]))["log_loss"]
    best = min(scores, key=scores.get)
    best_cls = next(c for c in CALIBRATORS if c.name == best)
    return best_cls().fit(proba_val, y_val), scores


# ----------------------------------------------------------------------------- ансамбль
class EnsembleModel:
    """Зважене середнє ймовірностей кількох моделей (LightGBM, логрегресія [, ринок])."""

    def __init__(self, members: dict[str, object], weights: dict[str, float] | None = None) -> None:
        self.members = dict(members)
        self.weights = weights or {k: 1.0 / len(self.members) for k in self.members}
        self.features = sorted({f for m in self.members.values() for f in m.features})

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Зважене середнє ймовірностей учасників, перенормоване до суми 1."""
        proba = sum(self.weights[k] * np.asarray(m.predict_proba(df), dtype=float)
                    for k, m in self.members.items() if self.weights[k] > 0)
        return proba / proba.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- фінальна модель
class FinalModel:
    """Фінальна модель для демо: базова модель + калібратор.

    Атрибути: features (колонки на вході), name, track, meta (метрики val, параметри).
    predict_proba(df) → (n, 3) у порядку [H, D, A], уже відкалібровано.
    """

    def __init__(self, base, calibrator, name: str, track: str, meta: dict | None = None) -> None:
        self.base = base
        self.calibrator = calibrator
        self.features = list(base.features)
        self.name = name
        self.track = track
        self.meta = meta or {}
        self.classes = list(config.CLASSES)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Калібровані ймовірності [H, D, A]; відсутні ознаки трактуються як NaN."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            df = df.assign(**{c: np.nan for c in missing})
        proba = self.calibrator.transform(np.asarray(self.base.predict_proba(df), dtype=float))
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Найімовірніший клас ('H'/'D'/'A') для кожного рядка."""
        return np.array(self.classes)[self.predict_proba(df).argmax(axis=1)]

    def __repr__(self) -> str:
        return f"FinalModel(track={self.track!r}, name={self.name!r}, n_features={len(self.features)})"
