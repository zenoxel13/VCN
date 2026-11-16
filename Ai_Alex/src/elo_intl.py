"""Власний Elo для національних збірних у дусі World Football Elo (eloratings.net).

Рейтинги рахуються послідовно з 1872 року. Для кожного матчу повертаються
ПЕРЕДМАТЧЕВІ рейтинги (оновлення — лише після матчу), тож витоку немає.
Параметри (бонус поля, масштаб K, множник за різницю голів) підібрано на train
функцією tune() — запуск: `python -m src.elo_intl`.
"""
from bisect import bisect_left
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src import config

START_RATING = 1500.0
ELO_SCALE = 400.0
K_BY_TYPE = {"friendly": 20, "qualifier": 40, "final": 50, "other": 30}
K_OVERRIDES = {"FIFA World Cup": 60, "UEFA Nations League": 40, "CONCACAF Nations League": 40}


@dataclass(frozen=True)
class EloParams:
    """Параметри Elo: бонус поля, множник до K, чи враховувати різницю голів."""
    home_advantage: float = 100.0
    k_scale: float = 1.0
    use_goal_diff: bool = True
    official_only: bool = False    # True → товариські не змінюють рейтинг (Elo лише офіційних)


DEFAULT_PARAMS = EloParams()                      # класичний World Football Elo (v1)
TUNED_PARAMS = EloParams(home_advantage=100.0, k_scale=1.25, use_goal_diff=True)  # tune() на train
OFFICIAL_PARAMS = EloParams(home_advantage=100.0, k_scale=1.25, official_only=True)


def k_factor(competition: str, tournament_type: str) -> float:
    """Вага турніру K: товариські 20, інші 30, відбір 40, фінали 50, ЧС 60."""
    return float(K_OVERRIDES.get(competition, K_BY_TYPE[tournament_type]))


def home_bonus(neutral, params: EloParams = TUNED_PARAMS) -> np.ndarray:
    """Бонус поля господаря: 0 на нейтральному полі, інакше params.home_advantage."""
    return np.where(np.asarray(neutral, dtype=bool), 0.0, params.home_advantage)


def expected_home(elo_home, elo_away, neutral, params: EloParams = TUNED_PARAMS) -> np.ndarray:
    """Очікуваний результат господаря (0…1) з урахуванням бонусу поля."""
    diff = np.asarray(elo_home) + home_bonus(neutral, params) - np.asarray(elo_away)
    return 1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE))


def goal_multiplier(goal_diff: int) -> float:
    """Множник за різницю голів: 1 / 1.5 / (11 + n) / 8."""
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


class EloHistory:
    """Історія рейтингів: передматчеві значення й рейтинг команди на будь-яку дату."""

    def __init__(self, matches: pd.DataFrame, params: EloParams = TUNED_PARAMS):
        """Проганяє Elo по матчах (мають бути відсортовані за датою/match_id)."""
        self.params = params
        n = len(matches)
        self.pre_home = np.empty(n)
        self.pre_away = np.empty(n)
        self._dates: dict[str, list] = {}
        self._ratings: dict[str, list[float]] = {}
        ratings: dict[str, float] = {}
        cols = zip(matches["date"].to_numpy(), matches["home"], matches["away"],
                   matches["home_goals"], matches["away_goals"], matches["neutral"],
                   matches["competition"], matches["tournament_type"])
        for i, (date, home, away, hg, ag, neutral, comp, ttype) in enumerate(cols):
            r_home = ratings.get(home, START_RATING)
            r_away = ratings.get(away, START_RATING)
            self.pre_home[i], self.pre_away[i] = r_home, r_away
            if params.official_only and ttype == "friendly":
                continue
            bonus = 0.0 if neutral else params.home_advantage
            expected = 1.0 / (1.0 + 10.0 ** (-(r_home + bonus - r_away) / ELO_SCALE))
            actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            mult = goal_multiplier(hg - ag) if params.use_goal_diff else 1.0
            delta = params.k_scale * k_factor(comp, ttype) * mult * (actual - expected)
            ratings[home], ratings[away] = r_home + delta, r_away - delta
            for team in (home, away):
                self._dates.setdefault(team, []).append(date)
                self._ratings.setdefault(team, []).append(ratings[team])
        self.current = ratings

    def rating_on(self, team: str, date) -> float:
        """Рейтинг команди після всіх її матчів СТРОГО до date (1500 для нових)."""
        dates = self._dates.get(team)
        if not dates:
            return START_RATING
        idx = bisect_left(dates, np.datetime64(pd.Timestamp(date), "ns"))
        return self._ratings[team][idx - 1] if idx > 0 else START_RATING

    def top(self, n: int = 15, since=None) -> pd.Series:
        """Топ-n команд за поточним рейтингом (опційно — лише ті, що грали з since)."""
        teams = self.current
        if since is not None:
            cutoff = np.datetime64(pd.Timestamp(since), "ns")
            teams = {t: r for t, r in teams.items() if self._dates[t][-1] >= cutoff}
        return pd.Series(teams).sort_values(ascending=False).head(n).round(1)


# ------------------------------------------------------------ підбір параметрів

GRID = {"home_advantage": [50.0, 75.0, 100.0, 125.0],
        "k_scale": [0.75, 1.0, 1.25, 1.5],
        "use_goal_diff": [True, False]}


def elo_logloss(matches: pd.DataFrame, params: EloParams) -> tuple[float, float]:
    """Log-loss найпростішого прогнозу з Elo: мультиноміальна логрегресія на різниці Elo.

    Логрегресія навчається лише на train (MIN_DATE ≤ date < VAL_START);
    повертає (log-loss train, log-loss val).
    """
    elo = EloHistory(matches, params)
    diff = (elo.pre_home + home_bonus(matches["neutral"], params) - elo.pre_away) / ELO_SCALE
    x, y, date = diff.reshape(-1, 1), matches["y"].to_numpy(), matches["date"]
    train = ((date >= config.MIN_DATE) & (date < config.VAL_START)).to_numpy()
    val = ((date >= config.VAL_START) & (date < config.TEST_START)).to_numpy()
    model = LogisticRegression().fit(x[train], y[train])
    return (log_loss(y[train], model.predict_proba(x[train]), labels=[0, 1, 2]),
            log_loss(y[val], model.predict_proba(x[val]), labels=[0, 1, 2]))


def tune(matches: pd.DataFrame) -> pd.DataFrame:
    """Сітка параметрів Elo; відбір — ЛИШЕ за train log-loss (val друкується для довідки)."""
    rows = []
    for ha, ks, gd in product(*GRID.values()):
        params = EloParams(ha, ks, gd)
        train_ll, val_ll = elo_logloss(matches, params)
        rows.append({"home_advantage": ha, "k_scale": ks, "use_goal_diff": gd,
                     "train_logloss": train_ll, "val_logloss": val_ll})
    return pd.DataFrame(rows).sort_values("train_logloss").reset_index(drop=True)


def main() -> None:
    """Підбір параметрів Elo і порівняння з класичними (v1)."""
    matches = pd.read_parquet(config.PROCESSED / "intl_matches.parquet")
    grid = tune(matches)
    print("Топ-10 комбінацій за train log-loss:\n" + grid.head(10).round(4).to_string())
    for name, params in (("класичні (v1)", DEFAULT_PARAMS), ("обрані (v2)", TUNED_PARAMS)):
        train_ll, val_ll = elo_logloss(matches, params)
        print(f"{name}: {params} → train {train_ll:.4f}, val {val_ll:.4f}")


if __name__ == "__main__":
    main()


# ------------------------------------------------- рейтинги атаки й оборони (Пуассон)

POIS_BASE = np.log(1.3)          # лог середньої кількості голів команди на нейтральному полі
POIS_HOME = 0.25                 # лог-перевага поля в голах (~ +28 % голів господаря)
POIS_LR = 0.04                   # крок онлайн-оновлення (×K/40); підібрано на train: 0.01–0.07
MAX_GOALS = 10                   # межа сітки рахунків для ймовірностей H/D/A


def poisson_xg(att_home, def_home, att_away, def_away, neutral) -> tuple[np.ndarray, np.ndarray]:
    """Очікувані голи (λ) господаря й гостя з рейтингів атаки/оборони."""
    home = np.where(np.asarray(neutral, dtype=bool), 0.0, POIS_HOME)
    xg_home = np.exp(POIS_BASE + home + np.asarray(att_home) - np.asarray(def_away))
    xg_away = np.exp(POIS_BASE + np.asarray(att_away) - np.asarray(def_home))
    return xg_home, xg_away


def poisson_outcome_probs(xg_home, xg_away) -> np.ndarray:
    """Ймовірності [H, D, A] з двох незалежних пуассонівських розподілів голів."""
    goals = np.arange(MAX_GOALS + 1)
    log_fact = np.cumsum(np.log(np.maximum(goals, 1)))

    def pmf(lam):
        lam = np.asarray(lam, dtype=float)[:, None]
        return np.exp(goals * np.log(lam) - lam - log_fact)

    grid = pmf(xg_home)[:, :, None] * pmf(xg_away)[:, None, :]
    probs = np.stack([np.tril(np.ones((MAX_GOALS + 1,) * 2), -1),
                      np.eye(MAX_GOALS + 1),
                      np.triu(np.ones((MAX_GOALS + 1,) * 2), 1)])
    out = np.einsum("nij,kij->nk", grid, probs)
    return out / out.sum(axis=1, keepdims=True)


class PoissonRatings:
    """Онлайн-рейтинги атаки й оборони збірних (у дусі Dixon–Coles / «Elo для голів»).

    log λ = база + перевага поля + атака − оборона суперника. Після матчу рейтинги
    зсуваються на lr·(K/40)·(голи − λ). Передматчеві значення пишуться ДО оновлення.
    """

    def __init__(self, matches: pd.DataFrame, lr: float = POIS_LR):
        """Проганяє модель по матчах (відсортованих за датою/match_id)."""
        n = len(matches)
        self.pre = np.empty((n, 4))     # att_home, def_home, att_away, def_away
        self._dates: dict[str, list] = {}
        self._states: dict[str, list[tuple[float, float]]] = {}
        att: dict[str, float] = {}
        dfn: dict[str, float] = {}
        cols = zip(matches["date"].to_numpy(), matches["home"], matches["away"],
                   matches["home_goals"], matches["away_goals"], matches["neutral"],
                   matches["competition"], matches["tournament_type"])
        for i, (date, home, away, hg, ag, neutral, comp, ttype) in enumerate(cols):
            ah, dh, aa, da = att.get(home, 0.0), dfn.get(home, 0.0), att.get(away, 0.0), dfn.get(away, 0.0)
            self.pre[i] = ah, dh, aa, da
            bonus = 0.0 if neutral else POIS_HOME
            lam_h = np.exp(POIS_BASE + bonus + ah - da)
            lam_a = np.exp(POIS_BASE + aa - dh)
            step = lr * k_factor(comp, ttype) / 40.0
            err_h, err_a = step * (hg - lam_h), step * (ag - lam_a)
            att[home], dfn[away] = ah + err_h, da - err_h
            att[away], dfn[home] = aa + err_a, dh - err_a
            for team in (home, away):
                self._dates.setdefault(team, []).append(date)
                self._states.setdefault(team, []).append((att[team], dfn[team]))

    def state_on(self, team: str, date) -> tuple[float, float]:
        """(атака, оборона) команди після всіх її матчів СТРОГО до date."""
        dates = self._dates.get(team)
        if not dates:
            return 0.0, 0.0
        idx = bisect_left(dates, np.datetime64(pd.Timestamp(date), "ns"))
        return self._states[team][idx - 1] if idx > 0 else (0.0, 0.0)


def poisson_logloss(matches: pd.DataFrame, lr: float) -> tuple[float, float]:
    """Log-loss Пуассон-прогнозу H/D/A (без жодного навчання) на train і val."""
    pre = PoissonRatings(matches, lr).pre
    probs = poisson_outcome_probs(*poisson_xg(pre[:, 0], pre[:, 1], pre[:, 2], pre[:, 3],
                                              matches["neutral"]))
    y, date = matches["y"].to_numpy(), matches["date"]
    train = ((date >= config.MIN_DATE) & (date < config.VAL_START)).to_numpy()
    val = ((date >= config.VAL_START) & (date < config.TEST_START)).to_numpy()
    return (log_loss(y[train], probs[train], labels=[0, 1, 2]),
            log_loss(y[val], probs[val], labels=[0, 1, 2]))
