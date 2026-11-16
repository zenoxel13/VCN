"""Ознаки для клубних матчів: clubs_matches.parquet → clubs_features.parquet + API для демо.

Усі ознаки рахуються лише з інформації, відомої ДО матчу:
- Elo (передматчевий ClubElo з Matches.csv, пропуски — останній знімок EloRatings.csv строго до дати);
- форма (очки) і ковзні середні статистики за ПОПЕРЕДНІ матчі: groupby(команда) → shift(1) → rolling;
- H2H за попередніми очними зустрічами (групування за невпорядкованою парою команд);
- дні відпочинку; ліга (категорія); безмаржові ймовірності з коефіцієнтів БК (f_odds_*).

Одна й та сама функція `compute_features` використовується і для навчальної таблиці, і в
`build_match_row` для демо — тому ознаки в демо ідентичні навчальним.
"""
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.clean_clubs import TEAM_ALIASES

MATCHES_FILE = config.PROCESSED / "clubs_matches.parquet"
ELO_FILE = config.RAW / "EloRatings.csv"
OUT_FILE = config.PROCESSED / "clubs_features.parquet"

# Перевага власного поля в Elo-пунктах. Підібрано на train (date < VAL_START) мінімізацією
# квадратичної похибки очікуваного очка Elo проти фактичного (див. fit_home_advantage, DECISIONS.md).
ELO_HOME_ADV = 70.0
ELO_SCALE = 400.0                 # стандартна шкала Elo: різниця 400 → шанси 10:1
ROLL_WINDOWS = (5, 10)            # вікна ковзних середніх (кількість попередніх матчів)
FORM_WINDOWS = (3, 5, 10)         # вікна форми (середні очки за матч)
MIN_PERIODS_SHARE = 0.4           # ковзне вважаємо валідним, якщо є ≥ 40% вікна (5→2, 10→4, 3→2)
H2H_WINDOW = 5                    # кількість попередніх очних зустрічей
REST_CAP_DAYS = 30                # дні відпочинку обрізаємо: >30 днів ≈ міжсезоння, різниці вже немає
RECENT_SEASONS = 2                # list_teams: команди, що грали в лізі за останні N сезонів
RUN_DIAGNOSTICS = False           # True → у main() ще й перевірка Form5 джерела, підбір переваги поля, звірка build_match_row
LEAK_CORR_LIMIT = 0.6             # |corr(ознака, різниця голів)| вище цього — підозра на витік
POINTS = {"W": 3.0, "D": 1.0, "L": 0.0}
V2_PREFIXES = ("f_own_elo", "f_league_draw", "f_season_ppg_diff")   # для звіту про нові ознаки
V3_PREFIXES = ("f_hodds_", "f_pi_", "f_pois_", "f_month", "f_weekday", "f_tm_")
V3_SUFFIXES = ("_prev_season_ppg", "_prev_season_gdpg", "_adj_att", "_adj_def", "_matches_14d", "_matches_30d", "_unbeaten_run", "_winless_run",
               "_league_newcomer")
V2_SUFFIXES = ("_season_played", "_season_ppg", "_season_gdpg", "_draw_share10", "_form_ewm")

# Власний Elo (рахуємо самі для ВСІХ ліг, бо ClubElo покриває лише ~60% матчів).
OWN_ELO_K = 20.0                  # базовий крок оновлення
OWN_ELO_HOME_ADV = 60.0           # перевага поля; підібрано на train (fit_own_home_advantage)
OWN_ELO_START = 1500.0            # рейтинг першої команди ліги, якої ще немає в історії
OWN_ELO_NEW_TEAM_GAP = 50.0       # новачок ліги (зазвичай підвищений клуб) слабший за середнього на 50
OWN_ELO_HFA_GRID = np.arange(0, 121, 10)
DRAW_WINDOW = 10                  # вікно частки нічиїх команди
LEAGUE_DRAW_SEASONS = 3           # частка нічиїх у лізі — за N попередніх сезонів
EWM_HALFLIFE = 5                  # півперіод (у матчах) експоненційно зваженої форми
# v3. Pi-рейтинги (Constantinou & Fenton, 2013): літературні параметри.
PI_LAMBDA = 0.035                 # швидкість навчання рейтингу
PI_GAMMA = 0.7                    # частка оновлення, що переноситься між домашнім і виїзним рейтингом
PI_BASE = 10.0                    # рейтинг → очікувана різниця голів: sign(R)·(10^(|R|/3) − 1)
PI_DIVISOR = 3.0
# v3. Атака/оборона з поправкою на суперника і Пуассон.
ADJ_WINDOW = 10                   # вікно ковзних голів (і скоригованих) для атаки/оборони
ADJ_MIN_DENOM = 0.3               # нижня межа середніх голів суперника в знаменнику (захист від ділення на ~0)
POISSON_MAX_GOALS = 10            # сітка рахунків 0..10 для Пуассон-ймовірностей
# v3. Календар, серії, історія ринку.
DENSITY_DAYS = (14, 30)           # кількість матчів команди за останні N днів
NEWCOMER_SEASONS = 3              # новачок ліги: не грав у ній N попередніх сезонів
HODDS_EWM_HALFLIFE = 5            # «ринковий рейтинг» — ewm ринкових оцінок за попередні матчі
LEAGUE_RECENT_MATCHES = 300       # демо: середній рейтинг ліги для новачка — за ~сезон останніх матчів

# Статистика матчу з погляду команди: назва → (колонка господаря, колонка гостя).
# «for» — своє значення, «against» — суперника. Береться ЛИШЕ для ковзних за попередні матчі.
TEAM_STATS = {
    "gf": ("home_goals", "away_goals"),
    "ga": ("away_goals", "home_goals"),
    "shots": ("HomeShots", "AwayShots"),
    "shots_ag": ("AwayShots", "HomeShots"),
    "sot": ("HomeTarget", "AwayTarget"),
    "corners": ("HomeCorners", "AwayCorners"),
}
ODDS_1X2 = {"": ("OddHome", "OddDraw", "OddAway"), "max_": ("MaxHome", "MaxDraw", "MaxAway")}


# ---------------------------------------------------------------- допоміжні

def _min_periods(window: int) -> int:
    """Мінімальна кількість спостережень для ковзного вікна."""
    return max(1, int(np.ceil(window * MIN_PERIODS_SHARE)))


def _prev_rolling_mean(df: pd.DataFrame, keys: list[str], cols: list[str], window: int) -> pd.DataFrame:
    """Середнє за `window` ПОПЕРЕДНІХ рядків у групі `keys` (shift(1) → rolling). df відсортований за часом."""
    shifted = df.groupby(keys, sort=False)[cols].shift(1)
    group_index = [df[k] for k in keys]
    rolled = shifted.groupby(group_index, sort=False).rolling(window, min_periods=_min_periods(window)).mean()
    return rolled.reset_index(level=list(range(len(keys))), drop=True).reindex(df.index)


def elo_expected_home(elo_diff: pd.Series, home_adv: float = ELO_HOME_ADV) -> pd.Series:
    """Очікуване очко господаря за Elo з урахуванням переваги поля."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_diff + home_adv) / ELO_SCALE))


@lru_cache(maxsize=1)
def load_elo_snapshots() -> pd.DataFrame:
    """Знімки ClubElo (1-го і 15-го числа), відсортовані за датою."""
    elo = pd.read_csv(ELO_FILE, parse_dates=["date"])
    elo["elo"] = pd.to_numeric(elo["elo"], errors="coerce")
    elo["club"] = elo["club"].str.strip().replace(TEAM_ALIASES)  # ті самі канонічні назви, що в матчах
    elo = elo.dropna(subset=["elo"]).drop_duplicates(["date", "club"], keep="last")
    return elo.sort_values("date")[["date", "club", "elo"]]


def _elo_before(dates: pd.Series, teams: pd.Series, snapshots: pd.DataFrame) -> pd.Series:
    """Останній знімок Elo команди СТРОГО до дати матчу (merge_asof, allow_exact_matches=False)."""
    left = pd.DataFrame({"date": dates.values, "club": teams.astype(str).values, "_pos": np.arange(len(dates))})
    merged = pd.merge_asof(left.sort_values("date"), snapshots, on="date", by="club",
                           allow_exact_matches=False, direction="backward")
    return pd.Series(merged.sort_values("_pos")["elo"].values, index=dates.index)


# ---------------------------------------------------------------- власний Elo

def _goal_multiplier(goal_diff: float) -> float:
    """Множник кроку за різницю голів (як у World Football Elo): 1, 1.5, далі (11+n)/8."""
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    return 1.5 if n == 2 else (11.0 + n) / 8.0


def own_elo_pass(m: pd.DataFrame, home_adv: float = OWN_ELO_HOME_ADV) -> pd.DataFrame:
    """Послідовний прохід Elo по всій історії `m` (відсортованій за часом).

    Повертає передматчеві (`own_elo_home`, `own_elo_away`) і післяматчеві (`*_post`) рейтинги.
    Рейтинг оновлюється лише ПІСЛЯ матчу, тож передматчеве значення не знає результату.
    Нова команда стартує з середнього поточного рейтингу своєї ліги мінус OWN_ELO_NEW_TEAM_GAP
    (у порожній лізі — OWN_ELO_START); середнє ліги підтримується інкрементно.
    """
    teams, team_names = pd.factorize(pd.concat([m["home"], m["away"]]).astype(str))
    n = len(m)
    home_idx, away_idx = teams[:n], teams[n:]
    div_idx, _ = pd.factorize(m["division"].astype(str))
    rating = np.full(len(team_names), np.nan)
    team_div = np.full(len(team_names), -1)
    div_sum = np.zeros(div_idx.max() + 1)
    div_cnt = np.zeros(div_idx.max() + 1)
    gd = (m["home_goals"] - m["away_goals"]).to_numpy(dtype=float)
    out = np.full((n, 4), np.nan)

    def place(t: int, d: int) -> None:
        """Ініціалізує нову команду або переносить її в іншу лігу (для середнього ліги)."""
        if np.isnan(rating[t]):
            mean = div_sum[d] / div_cnt[d] - OWN_ELO_NEW_TEAM_GAP if div_cnt[d] else OWN_ELO_START
            rating[t] = mean
        elif team_div[t] != d:
            div_sum[team_div[t]] -= rating[t]
            div_cnt[team_div[t]] -= 1
        else:
            return
        team_div[t] = d
        div_sum[d] += rating[t]
        div_cnt[d] += 1

    for i in range(n):
        h, a, d = home_idx[i], away_idx[i], div_idx[i]
        place(h, d)
        place(a, d)
        out[i, 0], out[i, 1] = rating[h], rating[a]
        if np.isnan(gd[i]):                       # майбутній матч: рейтинг не оновлюємо
            continue
        expected = 1.0 / (1.0 + 10.0 ** (-(rating[h] + home_adv - rating[a]) / ELO_SCALE))
        score = 1.0 if gd[i] > 0 else (0.5 if gd[i] == 0 else 0.0)
        delta = OWN_ELO_K * _goal_multiplier(gd[i]) * (score - expected)
        rating[h] += delta
        rating[a] -= delta
        div_sum[team_div[h]] += delta
        div_sum[team_div[a]] -= delta
        out[i, 2], out[i, 3] = rating[h], rating[a]
    cols = ["own_elo_home", "own_elo_away", "own_elo_home_post", "own_elo_away_post"]
    return pd.DataFrame(out, columns=cols, index=m.index)


def _pi_goals(rating: float) -> float:
    """Pi-рейтинг → очікувана перевага в голах."""
    return np.sign(rating) * (PI_BASE ** (abs(rating) / PI_DIVISOR) - 1.0)


def pi_rating_pass(m: pd.DataFrame) -> pd.DataFrame:
    """Pi-рейтинги: окремий домашній і виїзний рейтинг кожної команди, оновлення ЛИШЕ після матчу.

    Помилка прогнозу різниці голів насичується логарифмом (3·log10(1+e)); γ-частка оновлення
    домашнього рейтингу переноситься на виїзний і навпаки. Повертає передматчеві pi_home (домашній
    рейтинг господаря), pi_away (виїзний рейтинг гостя) і післяматчеві обидва рейтинги обох команд.
    """
    teams, names = pd.factorize(pd.concat([m["home"], m["away"]]).astype(str))
    n = len(m)
    home_idx, away_idx = teams[:n], teams[n:]
    r_home = np.zeros(len(names))
    r_away = np.zeros(len(names))
    gd = (m["home_goals"] - m["away_goals"]).to_numpy(dtype=float)
    out = np.full((n, 6), np.nan)
    for i in range(n):
        h, a = home_idx[i], away_idx[i]
        out[i, 0], out[i, 1] = r_home[h], r_away[a]
        if not np.isnan(gd[i]):
            expected = _pi_goals(r_home[h]) - _pi_goals(r_away[a])
            error = abs(gd[i] - expected)
            psi = PI_DIVISOR * np.log10(1.0 + error) * (1.0 if gd[i] > expected else -1.0)
            r_home[h] += psi * PI_LAMBDA
            r_away[h] += psi * PI_LAMBDA * PI_GAMMA
            r_away[a] -= psi * PI_LAMBDA
            r_home[a] -= psi * PI_LAMBDA * PI_GAMMA
        out[i, 2:] = r_home[h], r_away[h], r_home[a], r_away[a]
    cols = ["pi_home", "pi_away", "pi_h_home_post", "pi_h_away_post", "pi_a_home_post", "pi_a_away_post"]
    return pd.DataFrame(out, columns=cols, index=m.index)


def adjusted_goal_observations(m: pd.DataFrame) -> pd.DataFrame:
    """Голи матчу, скориговані на силу суперника (вхід для ковзних атаки/оборони; самі НЕ ознаки).

    att = забиті / середні пропущені суперника за ADJ_WINDOW матчів ДО цього матчу,
    def = пропущені / середні забиті суперника. Рахується на повній історії, бо потрібна історія суперника.
    """
    long = to_long(m)
    prev = _prev_rolling_mean(long, ["team"], ["gf", "ga"], ADJ_WINDOW).clip(lower=ADJ_MIN_DENOM)
    prev[["match_id", "is_home"]] = long[["match_id", "is_home"]]
    out = pd.DataFrame(index=m.index)
    for is_home, side in ((1, "home"), (0, "away")):
        own = long[long["is_home"] == is_home].set_index("match_id")
        opp = prev[prev["is_home"] == 1 - is_home].set_index("match_id")
        out[f"adj_att_{side}"] = (own["gf"] / opp["ga"]).reindex(m["match_id"]).to_numpy()
        out[f"adj_def_{side}"] = (own["ga"] / opp["gf"]).reindex(m["match_id"]).to_numpy()
    return out


def add_own_elo(m: pd.DataFrame) -> pd.DataFrame:
    """Додає до відсортованої історії стан, що залежить від УСІЄЇ історії (а не лише від двох команд):
    власний Elo, pi-рейтинги, скориговані на суперника голи."""
    m = m.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    return pd.concat([m, own_elo_pass(m), pi_rating_pass(m), adjusted_goal_observations(m)], axis=1)


def pi_features(m: pd.DataFrame) -> pd.DataFrame:
    """Ознаки pi-рейтингів: домашній рейтинг господаря, виїзний гостя, різниця, очікувана різниця голів."""
    exp_gd = m["pi_home"].map(_pi_goals) - m["pi_away"].map(_pi_goals)
    return pd.DataFrame({"f_pi_home": m["pi_home"], "f_pi_away": m["pi_away"],
                         "f_pi_diff": m["pi_home"] - m["pi_away"], "f_pi_exp_gd": exp_gd}, index=m.index)


def fit_own_home_advantage(m: pd.DataFrame) -> float:
    """Перевага поля власного Elo: мінімум MSE очікуваного очка на train (весь прохід для кожного значення)."""
    m = m.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    train = (m["date"] < config.VAL_START).to_numpy()
    actual = m.loc[train, "y"].map({0: 1.0, 1: 0.5, 2: 0.0})
    errors = []
    for adv in OWN_ELO_HFA_GRID:
        elo = own_elo_pass(m, adv)[train]
        expected = elo_expected_home(elo["own_elo_home"] - elo["own_elo_away"], adv)
        errors.append(((expected - actual) ** 2).mean())
    return float(OWN_ELO_HFA_GRID[int(np.argmin(errors))])


def own_elo_features(m: pd.DataFrame) -> pd.DataFrame:
    """Ознаки власного Elo (без пропусків): рейтинги, різниця, її модуль, очікуване очко господаря."""
    diff = m["own_elo_home"] - m["own_elo_away"]
    return pd.DataFrame({"f_own_elo_home": m["own_elo_home"], "f_own_elo_away": m["own_elo_away"],
                         "f_own_elo_diff": diff, "f_own_elo_absdiff": diff.abs(),
                         "f_own_elo_p_home": elo_expected_home(diff, OWN_ELO_HOME_ADV)}, index=m.index)


LEAGUE_RATE_SOURCES = {
    "draw": lambda h: (h["target"] == "D").astype(float),
    "home_goals": lambda h: h["home_goals"].astype(float),
    "away_goals": lambda h: h["away_goals"].astype(float),
}


@lru_cache(maxsize=None)
def league_rate_table(kind: str) -> pd.Series:
    """Середнє показника `kind` у лізі за LEAGUE_DRAW_SEASONS ПОПЕРЕДНІХ сезонів, індекс (division, season).

    Рахується з повної історії й включає наступний після останнього сезон (для демо).
    """
    h = _history()
    values = LEAGUE_RATE_SOURCES[kind](h)
    rates = values.groupby([h["division"].astype(str), h["season"]]).mean().unstack()
    rates = rates.reindex(columns=range(rates.columns.min(), rates.columns.max() + 2))
    prev = rates.T.shift(1).rolling(LEAGUE_DRAW_SEASONS, min_periods=1).mean().T
    return prev.stack()


def league_draw_features(m: pd.DataFrame) -> pd.DataFrame:
    """Частка нічиїх у лізі за попередні сезони (поточний сезон не використовується)."""
    return pd.DataFrame({"f_league_draw_rate": _league_rate(m, "draw")}, index=m.index)


def _league_rate(m: pd.DataFrame, kind: str) -> np.ndarray:
    """Значення league_rate_table(kind) для кожного матчу `m`."""
    keys = pd.MultiIndex.from_arrays([m["division"].astype(str), m["season"].astype(int)])
    return league_rate_table(kind).reindex(keys).to_numpy()


def poisson_features(m: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Очікувані голи = атака × оборона суперника × середнє ліги; з них Пуассон-ймовірності H/D/A."""
    lam_home = wide["f_home_adj_att"] * wide["f_away_adj_def"] * _league_rate(m, "home_goals")
    lam_away = wide["f_away_adj_att"] * wide["f_home_adj_def"] * _league_rate(m, "away_goals")
    goals = np.arange(POISSON_MAX_GOALS + 1)
    log_fact = np.cumsum(np.log(np.maximum(goals, 1)))

    def pmf(lam: pd.Series) -> np.ndarray:
        """Пуассонівські ймовірності 0..POISSON_MAX_GOALS голів для кожного рядка."""
        lam = lam.to_numpy(dtype=float)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.exp(goals * np.log(lam) - lam - log_fact)

    joint = pmf(lam_home)[:, :, None] * pmf(lam_away)[:, None, :]
    diff = goals[:, None] - goals[None, :]
    p = {"home": (joint * (diff > 0)).sum((1, 2)), "draw": (joint * (diff == 0)).sum((1, 2)),
         "away": (joint * (diff < 0)).sum((1, 2))}
    total = p["home"] + p["draw"] + p["away"]
    out = {"f_pois_xg_home": lam_home.to_numpy(), "f_pois_xg_away": lam_away.to_numpy()}
    out.update({f"f_pois_p_{k}": np.where(np.isnan(lam_home) | np.isnan(lam_away), np.nan, v / total)
                for k, v in p.items()})
    return pd.DataFrame(out, index=m.index)


# ---------------------------------------------------------------- блоки ознак

def elo_features(m: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Elo обох команд, різниця і очікуване очко господаря."""
    home = m["HomeElo"].fillna(_elo_before(m["date"], m["home"], snapshots))
    away = m["AwayElo"].fillna(_elo_before(m["date"], m["away"], snapshots))
    diff = home - away
    return pd.DataFrame({"f_elo_home": home, "f_elo_away": away, "f_elo_diff": diff,
                         "f_elo_p_home": elo_expected_home(diff)}, index=m.index)


def _long_market_and_context(m: pd.DataFrame) -> pd.DataFrame:
    """Для to_long (спершу господарі, потім гості): ринкові ймовірності, очікуване очко за власним Elo,
    ліга, скориговані голи. Ринкові величини матчу потрапляють в ознаки ЛИШЕ через shift(1) (f_hodds_*)."""
    probs, _ = _implied_probs(m[list(ODDS_1X2[""])])
    p_home, p_draw, p_away = (probs[c].to_numpy() for c in ODDS_1X2[""])
    if "own_elo_home" in m:
        elo_home = elo_expected_home(m["own_elo_home"] - m["own_elo_away"], OWN_ELO_HOME_ADV).to_numpy()
    else:
        elo_home = np.full(len(m), np.nan)
    sides = []
    for side, p_win, elo_score in (("home", p_home, elo_home), ("away", p_away, 1.0 - elo_home)):
        sides.append(pd.DataFrame({
            "division": m["division"].astype(str).values,
            "mkt_pwin": p_win, "mkt_xpts": POINTS["W"] * p_win + POINTS["D"] * p_draw,
            "mkt_vs_elo": p_win + 0.5 * p_draw - elo_score,
            "adj_att": m[f"adj_att_{side}"].values if f"adj_att_{side}" in m else np.nan,
            "adj_def": m[f"adj_def_{side}"].values if f"adj_def_{side}" in m else np.nan,
        }))
    return pd.concat(sides, ignore_index=True)


def to_long(m: pd.DataFrame) -> pd.DataFrame:
    """Матчі → по рядку на (матч, команда) зі статистикою з погляду команди."""
    parts = []
    for is_home, team_col in ((1, "home"), (0, "away")):
        side = pd.DataFrame({"match_id": m["match_id"].values, "date": m["date"].values,
                             "team": m[team_col].astype(str).values, "is_home": is_home,
                             "season": m["season"].astype(int).values})
        for stat, (home_col, away_col) in TEAM_STATS.items():
            side[stat] = m[home_col if is_home else away_col].values
        parts.append(side)
    long = pd.concat(parts, ignore_index=True)
    long = long.join(_long_market_and_context(m))
    outcome = np.sign(long["gf"] - long["ga"])
    long["points"] = outcome.map({1: POINTS["W"], 0: POINTS["D"], -1: POINTS["L"]})
    return long.sort_values(["date", "match_id", "is_home"], kind="mergesort").reset_index(drop=True)


def team_form_features(long: pd.DataFrame) -> pd.DataFrame:
    """Ковзні ознаки команди: форма, статистика за всі матчі й окремо за матчі на тому ж полі, відпочинок."""
    stats = list(TEAM_STATS)
    feats = {}
    for w in FORM_WINDOWS:
        feats[f"form{w}"] = _prev_rolling_mean(long, ["team"], ["points"], w)["points"]
    for w in ROLL_WINDOWS:
        overall = _prev_rolling_mean(long, ["team"], stats, w)
        venue = _prev_rolling_mean(long, ["team", "is_home"], stats + ["points"], w)
        for s in stats:
            feats[f"{s}_all{w}"] = overall[s]
            feats[f"{s}_venue{w}"] = venue[s]
        feats[f"form_venue{w}"] = venue["points"]
    rest = long.groupby("team", sort=False)["date"].diff().dt.days
    feats["rest_days"] = rest.clip(upper=REST_CAP_DAYS)
    long = long.assign(draw=(long["points"] == POINTS["D"]).where(long["points"].notna()).astype(float))
    feats[f"draw_share{DRAW_WINDOW}"] = _prev_rolling_mean(long, ["team"], ["draw"], DRAW_WINDOW)["draw"]
    feats["form_ewm"] = (long.groupby("team", sort=False)["points"].shift(1)
                         .groupby(long["team"], sort=False)
                         .transform(lambda s: s.ewm(halflife=EWM_HALFLIFE, ignore_na=True).mean()))
    feats.update(season_context(long))
    feats.update(market_history(long))
    feats.update(context_features(long))
    adj = _prev_rolling_mean(long, ["team"], ["adj_att", "adj_def"], ADJ_WINDOW)
    feats["adj_att"], feats["adj_def"] = adj["adj_att"], adj["adj_def"]
    out = pd.DataFrame(feats, index=long.index)
    out[["match_id", "is_home"]] = long[["match_id", "is_home"]]
    return out


def _prev_ewm(long: pd.DataFrame, col: str) -> pd.Series:
    """Експоненційно зважене середнє за ПОПЕРЕДНІ матчі команди (shift(1) → ewm)."""
    prev = long.groupby("team", sort=False)[col].shift(1)
    return prev.groupby(long["team"], sort=False).transform(
        lambda s: s.ewm(halflife=HODDS_EWM_HALFLIFE, ignore_na=True).mean())


def market_history(long: pd.DataFrame) -> dict[str, pd.Series]:
    """Історія ринку (hodds_*): як ринок оцінював команду в її ПОПЕРЕДНІХ матчах."""
    long = long.assign(mkt_overperf=long["points"] - long["mkt_xpts"])
    cols = ["mkt_pwin", "mkt_xpts", "mkt_vs_elo"]
    feats = {}
    for w in ROLL_WINDOWS:
        rolled = _prev_rolling_mean(long, ["team"], cols, w)
        for c in cols:
            feats[f"hodds_{c.removeprefix('mkt_')}{w}"] = rolled[c]
    for c in ("mkt_xpts", "mkt_vs_elo"):
        feats[f"hodds_{c.removeprefix('mkt_')}_ewm"] = _prev_ewm(long, c)
    feats["hodds_overperf10"] = _prev_rolling_mean(long, ["team"], ["mkt_overperf"], 10)["mkt_overperf"]
    return feats


def _streak(long: pd.DataFrame, keep: pd.Series) -> pd.Series:
    """Довжина поточної серії матчів, де виконується `keep`, ДО матчу (shift(1))."""
    keep = keep.astype(float)
    breaks = (keep == 0).groupby(long["team"], sort=False).cumsum()
    run = keep.groupby([long["team"], breaks], sort=False).cumsum()
    return run.groupby(long["team"], sort=False).shift(1)


def context_features(long: pd.DataFrame) -> dict[str, pd.Series]:
    """Календар і серії: щільність матчів, безпрограшна/безвиграшна серія, новачок ліги."""
    feats = {}
    by_team = long.sort_values(["team", "date"], kind="mergesort")
    dated = by_team.assign(one=1.0).set_index("date")
    for days in DENSITY_DAYS:
        cnt = dated.groupby("team", sort=False)["one"].rolling(f"{days}D", closed="left").count()
        cnt = pd.Series(cnt.to_numpy(), index=by_team.index).reindex(long.index)
        feats[f"matches_{days}d"] = cnt.fillna(0.0)
    played = long["points"].notna()
    feats["unbeaten_run"] = _streak(long, (long["points"] > 0) | ~played)
    feats["winless_run"] = _streak(long, (long["points"] < POINTS["W"]) | ~played)
    keys = ["team", "division", "season"]
    seen = long[keys].drop_duplicates()
    recent = pd.concat([seen.assign(season=seen["season"] + k) for k in range(1, NEWCOMER_SEASONS + 1)])
    recent = recent.drop_duplicates().assign(_seen=1.0)
    flag = long[keys].merge(recent, how="left", on=keys)
    feats["league_newcomer"] = pd.Series(flag["_seen"].isna().astype(float).to_numpy(), index=long.index)
    feats.update(previous_season(long))
    return feats


def previous_season(long: pd.DataFrame) -> dict[str, pd.Series]:
    """Сила команди в ПОПЕРЕДНЬОМУ сезоні (очки й різниця голів за гру) — підказка на старті нового сезону."""
    per_season = (long.assign(gd=long["gf"] - long["ga"])
                  .groupby(["team", "season"], sort=False)[["points", "gd"]].mean().reset_index())
    per_season["season"] += 1                       # значення сезону s стає ознакою сезону s+1
    prev = long[["team", "season"]].merge(per_season, how="left", on=["team", "season"])
    return {"prev_season_ppg": pd.Series(prev["points"].to_numpy(), index=long.index),
            "prev_season_gdpg": pd.Series(prev["gd"].to_numpy(), index=long.index)}


def season_context(long: pd.DataFrame) -> dict[str, pd.Series]:
    """Турнірний контекст поточного сезону строго ДО матчу: зіграно матчів, очки й різниця голів за гру."""
    keys = [long["team"], long["season"]]
    stats = pd.DataFrame({"points": long["points"], "gd": long["gf"] - long["ga"]})
    played = stats.groupby(keys, sort=False).cumcount().astype(float)
    prev = stats.groupby(keys, sort=False).shift(1)
    cum = prev.groupby(keys, sort=False).cumsum()
    per_game = cum.div(played.replace(0, np.nan), axis=0)
    return {"season_played": played, "season_ppg": per_game["points"], "season_gdpg": per_game["gd"]}


def side_features(team_feats: pd.DataFrame) -> pd.DataFrame:
    """Розгортає ознаки команд у широкий формат f_home_* / f_away_* по match_id."""
    sides = []
    for is_home, prefix in ((1, "f_home_"), (0, "f_away_")):
        part = team_feats[team_feats["is_home"] == is_home].drop(columns="is_home").set_index("match_id")
        sides.append(part.add_prefix(prefix))
    wide = sides[0].join(sides[1], how="outer")
    wide["f_rest_diff"] = wide["f_home_rest_days"] - wide["f_away_rest_days"]
    wide["f_form5_diff"] = wide["f_home_form5"] - wide["f_away_form5"]
    wide["f_season_ppg_diff"] = wide["f_home_season_ppg"] - wide["f_away_season_ppg"]
    for c in ("xpts10", "vs_elo10", "xpts_ewm", "vs_elo_ewm", "overperf10"):
        wide[f"f_hodds_diff_{c}"] = wide[f"f_home_hodds_{c}"] - wide[f"f_away_hodds_{c}"]
    wide = wide.rename(columns=lambda c: c.replace("f_home_hodds_", "f_hodds_home_")
                       .replace("f_away_hodds_", "f_hodds_away_"))
    return wide


def h2h_features(m: pd.DataFrame) -> pd.DataFrame:
    """H2H за останні H2H_WINDOW очних зустрічей ДО матчу, з погляду поточного господаря."""
    home, away = m["home"].astype(str), m["away"].astype(str)
    first_is_home = home <= away
    pair = np.where(first_is_home, home + "|" + away, away + "|" + home)
    gd_home = m["home_goals"] - m["away_goals"]
    gd_first = np.where(first_is_home, gd_home, -gd_home)        # різниця голів «першої» команди пари
    h = pd.DataFrame({"pair": pair, "gd": gd_first, "win": (gd_first > 0).astype(float),
                      "loss": (gd_first < 0).astype(float), "draw": (gd_first == 0).astype(float),
                      "one": 1.0, "date": m["date"].values}, index=m.index)
    shifted = h.groupby("pair", sort=False)[["gd", "win", "loss", "draw", "one"]].shift(1)
    rolled = shifted.groupby(h["pair"], sort=False).rolling(H2H_WINDOW, min_periods=1)
    mean = rolled.mean().reset_index(level=0, drop=True).reindex(m.index)
    count = rolled.sum()["one"].reset_index(level=0, drop=True).reindex(m.index)
    sign = np.where(first_is_home, 1.0, -1.0)
    return pd.DataFrame({
        "f_h2h_n": count.fillna(0.0),
        "f_h2h_home_win": np.where(first_is_home, mean["win"], mean["loss"]),
        "f_h2h_draw": mean["draw"],
        "f_h2h_gd": mean["gd"] * sign,
    }, index=m.index)


def _implied_probs(odds: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Безмаржові ймовірності (нормалізовані 1/odds) і маржа букмекера."""
    inv = 1.0 / odds
    total = inv.sum(axis=1, min_count=odds.shape[1])
    return inv.div(total, axis=0), total - 1.0


def odds_features(m: pd.DataFrame) -> pd.DataFrame:
    """Усе, що походить від передматчевих коефіцієнтів: 1X2 (середні й максимальні), тотал 2.5, фора."""
    out = {}
    for prefix, cols in ODDS_1X2.items():
        probs, margin = _implied_probs(m[list(cols)])
        for outcome, col in zip(("home", "draw", "away"), cols):
            out[f"f_odds_{prefix}p_{outcome}"] = probs[col]
        out[f"f_odds_{prefix}margin"] = margin
    over, _ = _implied_probs(m[["Over25", "Under25"]])
    out["f_odds_p_over25"] = over["Over25"]
    handi, _ = _implied_probs(m[["HandiHome", "HandiAway"]])
    out["f_odds_handi_size"] = m["HandiSize"]
    out["f_odds_p_handi_home"] = handi["HandiHome"]
    return pd.DataFrame(out, index=m.index).astype(float)


# ---------------------------------------------------------------- збірка

@lru_cache(maxsize=1)
def division_categories() -> tuple[str, ...]:
    """Фіксований набір категорій ліги (однаковий для навчання і демо)."""
    return tuple(sorted(pd.read_parquet(MATCHES_FILE, columns=["division"])["division"].astype(str).unique()))


def compute_features(m: pd.DataFrame, snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Будує таблицю ознак за схемою контракту для набору матчів `m` (формат clubs_matches.parquet).

    Ознаки рядка залежать лише від попередніх матчів у `m`, тому для демо достатньо передати
    історію двох команд + фіктивний рядок майбутнього матчу (результат NaN). Виняток — власний Elo:
    він залежить від усієї історії, тому `m` має вже містити колонки own_elo_home/own_elo_away
    (див. add_own_elo; для демо — стан команди після її останнього матчу до дати).
    """
    if snapshots is None:
        snapshots = load_elo_snapshots()
    m = m.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    meta = pd.DataFrame({
        "match_id": m["match_id"].astype("int64"), "date": m["date"],
        "home": m["home"].astype(str), "away": m["away"].astype(str),
        "competition": m["division"].astype(str), "season": m["season"].astype(int),
        "target": m["target"], "y": m["y"], "home_goals": m["home_goals"], "away_goals": m["away_goals"],
    })
    sides = side_features(team_form_features(to_long(m))).reindex(m["match_id"].values)
    sides.index = m.index
    calendar = pd.DataFrame({"f_month": m["date"].dt.month.astype(float),
                             "f_weekday": m["date"].dt.day_of_week.astype(float)}, index=m.index)
    parts = [meta, elo_features(m, snapshots), own_elo_features(m), pi_features(m), sides,
             poisson_features(m, sides), calendar, h2h_features(m),
             league_draw_features(m), odds_features(m)]
    feats = pd.concat(parts, axis=1)
    feats["f_cat_division"] = pd.Categorical(m["division"].astype(str), categories=division_categories())
    num = [c for c in feats.columns if c.startswith("f_") and c != "f_cat_division"]
    feats[num] = feats[num].astype(float)
    return feats


def fit_home_advantage(feats: pd.DataFrame) -> float:
    """Підбирає перевагу поля в Elo на train: мінімум MSE очікуваного очка проти фактичного."""
    train = feats[(feats["date"] < config.VAL_START) & feats["f_elo_diff"].notna()]
    actual = train["y"].map({0: 1.0, 1: 0.5, 2: 0.0})
    grid = np.arange(0, 151, 5)
    errors = [((elo_expected_home(train["f_elo_diff"], adv) - actual) ** 2).mean() for adv in grid]
    return float(grid[int(np.argmin(errors))])


# ---------------------------------------------------------------- перевірки

def check_dataset_form(m: pd.DataFrame) -> None:
    """Перевіряє, чи Form5Home у джерелі передматчевий: порівнює з власною сумою очок за 5 попередніх матчів."""
    long = to_long(m)
    pts = long.groupby("team", sort=False)["points"]
    prev5 = pts.transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())
    incl5 = pts.transform(lambda s: s.rolling(5, min_periods=5).sum())
    home = long[long["is_home"] == 1].set_index("match_id")
    src = m.set_index("match_id")["Form5Home"]
    for name, own in (("передматчева (shift 1)", prev5), ("з поточним матчем", incl5)):
        own_home = pd.Series(own.values, index=long["match_id"])[home.index]
        ok = (own_home - src.reindex(home.index)).abs().lt(0.5)
        print(f"  Form5Home збігається з власною формою [{name}]: {ok[own_home.notna()].mean():.1%}")


def report(feats: pd.DataFrame) -> None:
    """Санітарні перевірки таблиці ознак, включно з тестом на витік."""
    fcols = [c for c in feats.columns if c.startswith("f_")]
    print(f"Розмір: {feats.shape}, ознак f_: {len(fcols)}")
    print(f"match_id унікальний: {feats['match_id'].is_unique}")
    print(f"Діапазон дат: {feats['date'].min().date()} … {feats['date'].max().date()}")
    print("Розподіл target:", feats["target"].value_counts(normalize=True).round(3).to_dict())
    na = feats[fcols].isna().mean().round(3)
    print("Частка NaN по f_-колонках:\n" + na.to_string())
    gd = feats["home_goals"] - feats["away_goals"]
    num = [c for c in fcols if c != "f_cat_division"]
    corr = feats[num].corrwith(gd).abs().sort_values(ascending=False)
    print("Тест на витік — |corr(ознака, home_goals − away_goals)|, топ-10:\n" + corr.head(10).round(3).to_string())
    new = [c for c in corr.index if c.startswith(V3_PREFIXES) or c.endswith(V3_SUFFIXES)]
    print("Нові ознаки v3 — |corr| з різницею голів:\n" + corr[new].round(3).to_string())
    suspicious = corr[corr > LEAK_CORR_LIMIT]
    print(f"Підозрілих (> {LEAK_CORR_LIMIT}): {list(suspicious.index) or 'немає'}")
    if len(suspicious):
        raise ValueError(f"Підозра на витік у клубних ознаках: {list(suspicious.index)}")


# ---------------------------------------------------------------- API для демо

@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    """Очищені матчі з власним Elo (кешуються в модулі)."""
    return add_own_elo(pd.read_parquet(MATCHES_FILE))


def _team_state_before(team: str, date: pd.Timestamp, competition: str) -> dict[str, float]:
    """Стан команди після її останнього матчу до `date`: власний Elo і обидва pi-рейтинги.

    Для новачка — як у проходах: Elo = середнє ліги − OWN_ELO_NEW_TEAM_GAP, pi-рейтинги = 0.
    """
    h = _history()
    past = h[h["date"] < date]
    mine = past[(past["home"] == team) | (past["away"] == team)]
    if len(mine):
        last = mine.iloc[-1]
        side = "home" if last["home"] == team else "away"
        pi = side[0]
        return {"elo": float(last[f"own_elo_{side}_post"]),
                "pi_home": float(last[f"pi_{pi}_home_post"]), "pi_away": float(last[f"pi_{pi}_away_post"])}
    league = past[past["division"] == competition].tail(LEAGUE_RECENT_MATCHES)
    elo = (OWN_ELO_START if league.empty else
           float(pd.concat([league["own_elo_home"], league["own_elo_away"]]).mean() - OWN_ELO_NEW_TEAM_GAP))
    return {"elo": elo, "pi_home": 0.0, "pi_away": 0.0}


def with_tm_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Додає f_tm_* (Transfermarkt, підчат TM). Якщо модуля чи файлів даних немає — попередження
    й продовження без них; будь-яка інша помилка не ковтається, щоб баг не лишився непоміченим."""
    try:
        from src.features_tm import add_tm_features
        return add_tm_features(feats)
    except (ImportError, FileNotFoundError) as err:
        print(f"Попередження: ознаки Transfermarkt не додано ({type(err).__name__}: {err})")
        return feats


def list_competitions() -> list[str]:
    """Коди ліг (Division)."""
    return list(division_categories())


def list_teams(competition: str | None = None) -> list[str]:
    """Команди; якщо задано лігу — ті, що грали в ній за останні RECENT_SEASONS сезони."""
    h = _history()
    if competition is not None:
        h = h[(h["division"] == competition) & (h["season"] > h["season"].max() - RECENT_SEASONS)]
    return sorted(pd.concat([h["home"], h["away"]]).astype(str).unique())


def build_match_row(home: str, away: str, date, competition: str, odds=None, **extra) -> pd.DataFrame:
    """1 рядок з усіма f_-колонками для гіпотетичного матчу home–away у день `date`.

    Береться історія обох команд строго ДО date, дописується фіктивний матч (результат NaN,
    Elo — останній знімок до date, коефіцієнти — `odds`=(home, draw, away) або NaN), і
    проганяється та сама `compute_features`, що й для навчальної таблиці.
    """
    date = pd.Timestamp(date)
    h = _history()
    hist = h[(h["date"] < date) & (h["home"].isin([home, away]) | h["away"].isin([home, away]))]
    state_home = _team_state_before(home, date, competition)
    state_away = _team_state_before(away, date, competition)
    fake = {col: np.nan for col in h.columns}
    fake.update({"match_id": int(h["match_id"].max()) + 1, "date": date, "home": home, "away": away,
                 "division": competition, "season": int(date.year - (date.month < 7)),
                 "target": pd.NA, "y": -1,
                 "own_elo_home": state_home["elo"], "own_elo_away": state_away["elo"],
                 "pi_home": state_home["pi_home"], "pi_away": state_away["pi_away"]})
    if odds is not None:
        fake.update(dict(zip(ODDS_1X2[""], map(float, odds))))
    data = pd.concat([hist, pd.DataFrame([fake])], ignore_index=True)
    snaps = load_elo_snapshots()
    snaps = snaps[snaps["club"].isin([home, away])]
    row = with_tm_features(compute_features(data, snaps).iloc[[-1]])
    fcols = [c for c in row.columns if c.startswith("f_")]
    return row[fcols].reset_index(drop=True)


def check_build_match_row(feats: pd.DataFrame, n: int = 20) -> None:
    """Порівнює build_match_row для реальних матчів з рядками parquet (має збігатися, крім odds і ClubElo)."""
    sample = feats[feats["date"] >= config.TEST_START].sample(n, random_state=config.RANDOM_STATE)
    fcols = [c for c in feats.columns if c.startswith("f_") and c != "f_cat_division"
             and not c.startswith(("f_odds_", "f_elo_"))]
    times, n_bad = [], 0
    for _, r in sample.iterrows():
        t0 = time.perf_counter()
        row = build_match_row(r["home"], r["away"], r["date"], r["competition"])
        dt = time.perf_counter() - t0
        diff = (row[fcols].iloc[0] - r[fcols]).abs()
        both_nan = row[fcols].iloc[0].isna() & r[fcols].isna()
        bad = diff[(diff > 1e-9) | (diff.isna() & ~both_nan)]
        times.append(dt)
        if len(bad):
            print(f"  {r['date'].date()} {r['home']}–{r['away']}: розбіжності {list(bad.index)[:5]}")
        n_bad += bool(len(bad))
    print(f"  Матчів із розбіжностями: {n_bad}/{n}, час виклику: макс {max(times):.2f} с")


def main() -> None:
    """Точка входу: будує clubs_features.parquet і друкує перевірки."""
    t0 = time.perf_counter()
    m = _history()
    feats = with_tm_features(compute_features(m))
    print(f"Ознаки пораховано за {time.perf_counter() - t0:.1f} с")
    feats = feats[feats["date"] >= config.MIN_DATE].reset_index(drop=True)
    report(feats)
    feats.to_parquet(OUT_FILE, index=False)
    print(f"Збережено: {OUT_FILE}")
    if RUN_DIAGNOSTICS:
        run_diagnostics(m, feats)


def run_diagnostics(m: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Повільніші разові перевірки (результати зафіксовано в DECISIONS.md)."""
    print("Перевірка Form3/Form5 джерела:")
    check_dataset_form(m)
    print(f"Перевага поля ClubElo, підібрана на train: {fit_home_advantage(feats):.0f} (використано {ELO_HOME_ADV:.0f})")
    print(f"Перевага поля власного Elo, підібрана на train: {fit_own_home_advantage(m):.0f} "
          f"(використано {OWN_ELO_HOME_ADV:.0f})")
    print("Перевірка build_match_row проти parquet:")
    check_build_match_row(feats)


if __name__ == "__main__":
    main()
