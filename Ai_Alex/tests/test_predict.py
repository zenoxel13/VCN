"""Прогноз для демо: формат відповіді, валідація вводу, здоровий глузд ймовірностей."""
import pytest

from src import predict as pr
from tests.conftest import require_model

DATE = "2026-11-20"
CLUB = ("Arsenal", "Chelsea", "E0")
INTL_STRONG, INTL_WEAK, INTL_COMP = "Spain", "San Marino", "FIFA World Cup qualification"


@pytest.fixture(scope="module", autouse=True)
def _models():
    """Тести прогнозу потребують навчених моделей і очищених матчів."""
    require_model("clubs_no_odds.joblib")
    require_model("intl.joblib")


def _check_proba(p: dict) -> None:
    """Словник {H, D, A}, кожна ймовірність у (0, 1), сума = 1."""
    assert set(p) == {"H", "D", "A"}
    assert all(0.0 < v < 1.0 for v in p.values()), p
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_clubs_format():
    """Клуби: H/D/A із сумою 1 (без коефіцієнтів і з ними)."""
    home, away, comp = CLUB
    _check_proba(pr.predict(home, away, DATE, comp))
    _check_proba(pr.predict(home, away, DATE, comp, odds=(2.0, 3.5, 3.8)))


def test_predict_intl_format():
    """Збірні: H/D/A із сумою 1, і на нейтральному полі теж."""
    _check_proba(pr.predict("Ukraine", "England", DATE, INTL_COMP))
    _check_proba(pr.predict("Ukraine", "England", DATE, INTL_COMP, neutral=True))


@pytest.mark.parametrize("args", [CLUB[:1] * 2 + CLUB[2:], (INTL_STRONG, INTL_STRONG, INTL_COMP)])
def test_same_team_raises(args):
    """Однакові команди → ValueError."""
    home, away, comp = args
    with pytest.raises(ValueError):
        pr.predict(home, away, DATE, comp)


def test_unknown_team_raises():
    """Невідома команда → ValueError, а не падіння."""
    with pytest.raises(ValueError):
        pr.predict("Неіснуюча команда", "England", DATE, INTL_COMP)


def test_stronger_team_favoured():
    """Іспанія проти Сан-Марино: Іспанія — явний фаворит і вдома, і в гостях."""
    home = pr.predict(INTL_STRONG, INTL_WEAK, DATE, INTL_COMP)
    away = pr.predict(INTL_WEAK, INTL_STRONG, DATE, INTL_COMP)
    assert home["H"] > 0.7 and home["H"] > home["A"], home
    assert away["A"] > 0.6 and away["A"] > away["H"], away


@pytest.mark.parametrize("home,away,comp", [CLUB, ("Ukraine", "England", INTL_COMP)])
def test_swap_changes_probabilities(home, away, comp):
    """Зміна місць команд змінює ймовірності (модель враховує, хто господар)."""
    p1 = pr.predict(home, away, DATE, comp)
    p2 = pr.predict(away, home, DATE, comp)
    assert abs(p1["H"] - p2["H"]) > 1e-3 or abs(p1["A"] - p2["A"]) > 1e-3, (p1, p2)
