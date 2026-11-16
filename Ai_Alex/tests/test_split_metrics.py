"""Розбиття за часом і метрики якості (RPS, Brier, log-loss) на ручних прикладах."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.modeling import brier_score, compute_metrics, rps_score, split_by_time


# ----------------------------------------------------------------------------- розбиття
def test_split_disjoint_and_ordered(track_features):
    """train/val/test не перетинаються, покривають усі рядки й ідуть строго одне за одним у часі."""
    track, df = track_features
    parts = split_by_time(df)
    ids = {k: set(v["match_id"]) for k, v in parts.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["val"] & ids["test"]) and not (ids["train"] & ids["test"])
    assert sum(len(v) for v in parts.values()) == len(df), f"[{track}] розбиття втратило рядки"
    assert all(len(v) for v in parts.values()), f"[{track}] порожня частина розбиття"
    assert parts["train"]["date"].max() < pd.Timestamp(config.VAL_START) <= parts["val"]["date"].min()
    assert parts["val"]["date"].max() < pd.Timestamp(config.TEST_START) <= parts["test"]["date"].min()


def test_split_synthetic_boundaries():
    """Межові дати потрапляють у правильну частину (VAL_START → val, TEST_START → test)."""
    dates = pd.to_datetime(["2022-06-30", config.VAL_START, "2023-06-30", config.TEST_START])
    df = pd.DataFrame({"match_id": range(4), "date": dates})
    parts = split_by_time(df)
    assert list(parts["train"]["match_id"]) == [0]
    assert list(parts["val"]["match_id"]) == [1, 2]
    assert list(parts["test"]["match_id"]) == [3]


# ----------------------------------------------------------------------------- метрики
Y = np.array([0, 1, 2])
PERFECT = np.eye(3)
UNIFORM = np.full((3, 3), 1 / 3)


def test_perfect_forecast_is_zero():
    """Ідеальний прогноз: RPS = Brier = 0, log-loss ≈ 0, accuracy = 1."""
    assert rps_score(Y, PERFECT) == pytest.approx(0.0)
    assert brier_score(Y, PERFECT) == pytest.approx(0.0)
    m = compute_metrics(Y, PERFECT)
    assert m["log_loss"] == pytest.approx(0.0, abs=1e-4)
    assert m["accuracy"] == 1.0


def test_uniform_forecast_known_values():
    """Рівномірний прогноз: log-loss = ln 3, Brier = 2/3; RPS: H → 5/18, D → 1/9, A → 5/18."""
    m = compute_metrics(Y, UNIFORM)
    assert m["log_loss"] == pytest.approx(np.log(3))
    assert m["brier"] == pytest.approx(2 / 3)
    assert rps_score(np.array([0]), UNIFORM[:1]) == pytest.approx(5 / 18)
    assert rps_score(np.array([1]), UNIFORM[:1]) == pytest.approx(1 / 9)
    assert rps_score(np.array([2]), UNIFORM[:1]) == pytest.approx(5 / 18)
    assert m["rps"] == pytest.approx((5 / 18 + 1 / 9 + 5 / 18) / 3)


def test_rps_respects_order():
    """RPS карає «далеку» помилку сильніше: факт H, прогноз A гірший за прогноз D."""
    y = np.array([0])
    assert rps_score(y, np.array([[0, 0, 1.0]])) > rps_score(y, np.array([[0, 1.0, 0]]))
    assert rps_score(y, np.array([[0, 0, 1.0]])) == pytest.approx(1.0)
