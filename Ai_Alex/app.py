"""Демо-застосунок: прогноз результату футбольного матчу (H/D/A).

Запуск із кореня проєкту:
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src import predict as pr

OUTCOME_LABELS = {"H": "Перемога господарів", "D": "Нічия", "A": "Перемога гостей"}
OUTCOME_COLORS = {"H": "#2e7d32", "D": "#9e9e9e", "A": "#c62828"}
TRACK_LABELS = {"clubs": "Клуби", "intl": "Збірні"}
MIN_DATE, MAX_DATE = dt.date(2000, 1, 1), dt.date(2030, 12, 31)

# Людські назви кодів клубних ліг (топ-5 — із config.TOP5)
LEAGUE_NAMES = {
    **config.TOP5,
    "E1": "Англія, Чемпіоншип", "E2": "Англія, Ліга 1", "E3": "Англія, Ліга 2", "EC": "Англія, Національна ліга",
    "SP2": "Іспанія, Сегунда", "D2": "Німеччина, Бундесліга 2", "I2": "Італія, Серія B", "F2": "Франція, Ліга 2",
    "N1": "Нідерланди, Ередивізі", "P1": "Португалія, Прімейра", "B1": "Бельгія, Про-ліга",
    "T1": "Туреччина, Суперліга", "G1": "Греція, Суперліга", "SC0": "Шотландія, Прем'єршип",
    "SC1": "Шотландія, Чемпіоншип", "SC2": "Шотландія, Ліга 1", "SC3": "Шотландія, Ліга 2",
    "ARG": "Аргентина", "AUT": "Австрія", "BRA": "Бразилія", "CHN": "Китай", "DEN": "Данія",
    "FIN": "Фінляндія", "IRL": "Ірландія", "JAP": "Японія", "MEX": "Мексика", "NOR": "Норвегія",
    "POL": "Польща", "ROM": "Румунія", "RUS": "Росія", "SUI": "Швейцарія", "SWE": "Швеція", "USA": "США",
}

# Ключові ознаки: (людська назва, шаблон колонки; {side} → home/away). Відсутні колонки пропускаються.
TEAM_FEATURES = {
    "clubs": [
        ("Власний рейтинг Elo", "f_own_elo_{side}"),
        ("Рейтинг Elo (ClubElo)", "f_elo_{side}"),
        ("Pi-рейтинг", "f_pi_{side}"),
        ("Очікувані голи (модель Пуассона)", "f_pois_xg_{side}"),
        ("Вартість складу, млн € (Transfermarkt)", "f_tm_{side}_value_log"),
        ("Середній вік складу", "f_tm_{side}_avg_age"),
        ("Стаж тренера, днів", "f_tm_{side}_coach_days"),
        ("Очки за матч, останні 5", "f_{side}_form5"),
        ("Очки за матч, останні 10", "f_{side}_form10"),
        ("Забиті голи (сер., 5 матчів)", "f_{side}_gf_all5"),
        ("Пропущені голи (сер., 5 матчів)", "f_{side}_ga_all5"),
        ("Забиті голи (сер., 10 матчів)", "f_{side}_gf_all10"),
        ("Пропущені голи (сер., 10 матчів)", "f_{side}_ga_all10"),
        ("Очки за матч у цьому сезоні", "f_{side}_season_ppg"),
        ("Різниця м'ячів за матч у сезоні", "f_{side}_season_gdpg"),
        ("Зіграно матчів у сезоні", "f_{side}_season_played"),
        ("Частка нічиїх, останні 10", "f_{side}_draw_share10"),
        ("Днів відпочинку", "f_{side}_rest_days"),
    ],
    "intl": [
        ("Рейтинг Elo", "f_elo_{side}"),
        ("Очки за матч, останні 5", "f_{side}_pts5"),
        ("Очки за матч, останні 10", "f_{side}_pts10"),
        ("Забиті голи (сер., 5 матчів)", "f_{side}_gf5"),
        ("Пропущені голи (сер., 5 матчів)", "f_{side}_ga5"),
        ("Забиті голи (сер., 10 матчів)", "f_{side}_gf10"),
        ("Пропущені голи (сер., 10 матчів)", "f_{side}_ga10"),
        ("Очікувані голи (xG за історією)", "f_xg_{side}"),
        ("Сер. Elo суперників, останні 5", "f_{side}_opp_elo5"),
        ("Зміна Elo за останні 5 матчів", "f_{side}_elo_mom5"),
        ("Частка нічиїх, останні 10", "f_{side}_draw10"),
        ("Днів відпочинку", "f_{side}_rest_days"),
        ("Матчів в історії", "f_{side}_n_matches"),
    ],
}
MATCH_FEATURES = {
    "clubs": [
        ("Різниця власного Elo (господарі − гості)", "f_own_elo_diff"),
        ("Ймовірність перемоги господарів за власним Elo", "f_own_elo_p_home"),
        ("Різниця Elo ClubElo", "f_elo_diff"),
        ("Різниця pi-рейтингів", "f_pi_diff"),
        ("Пуассон: імовірність перемоги господарів", "f_pois_p_home"),
        ("Пуассон: імовірність нічиєї", "f_pois_p_draw"),
        ("Пуассон: імовірність перемоги гостей", "f_pois_p_away"),
        ("Різниця очок за матч у сезоні", "f_season_ppg_diff"),
        ("Частка нічиїх у лізі", "f_league_draw_rate"),
        ("Очні зустрічі: кількість", "f_h2h_n"),
        ("Очні зустрічі: частка перемог господарів", "f_h2h_home_win"),
        ("Очні зустрічі: частка нічиїх", "f_h2h_draw"),
        ("Очні зустрічі: середня різниця голів", "f_h2h_gd"),
        ("Різниця у відпочинку, днів", "f_rest_diff"),
    ],
    "intl": [
        ("Різниця Elo (з бонусом поля)", "f_elo_diff"),
        ("Ймовірність перемоги господарів за Elo", "f_elo_p_home"),
        ("Нейтральне поле", "f_neutral"),
        ("Пуассон: імовірність перемоги господарів", "f_pois_p_home"),
        ("Пуассон: імовірність нічиєї", "f_pois_p_draw"),
        ("Очні зустрічі: кількість", "f_h2h_n"),
        ("Очні зустрічі: баланс", "f_h2h_balance"),
        ("Різниця очок за 5 матчів", "f_pts5_diff"),
        ("Різниця м'ячів за 5 матчів", "f_gd5_diff"),
    ],
}
def _value_mln(log_value: float) -> float:
    """log1p(вартість) → млн € (якщо джерело в євро — ділимо на мільйон)."""
    value = float(np.expm1(log_value))
    return value / 1e6 if value > 1e4 else value


# Перетворення значень перед показом (ключ — шаблон колонки)
VALUE_TRANSFORMS = {"f_tm_{side}_value_log": _value_mln}

# Словник для автоматичного перекладу назв будь-яких (зокрема нових) f_-ознак
TOKENS = {
    "elo": "Elo", "home": "господарі", "away": "гості", "form": "форма", "gf": "забиті", "ga": "пропущені",
    "all": "усі матчі", "venue": "на цьому полі", "shots": "удари", "ag": "проти", "sot": "удари в площину",
    "corners": "кутові", "rest": "відпочинок", "days": "днів", "diff": "різниця", "h2h": "очні",
    "n": "кількість", "win": "перемоги", "draw": "нічиї", "gd": "різниця м'ячів", "p": "ймовірність",
    "pts": "очки", "neutral": "нейтральне поле", "k": "K", "factor": "фактор", "balance": "баланс",
    "matches": "матчів", "odds": "БК", "margin": "маржа", "max": "макс.", "division": "ліга",
    "tournament": "турнір", "type": "тип", "over25": "тотал > 2.5", "handi": "фора", "size": "розмір",
    "own": "власний", "opp": "суперники", "strength": "сила", "rank": "місце", "position": "позиція",
    "season": "сезон", "table": "таблиця", "points": "очки", "rate": "частка", "draws": "нічиї",
    "avg": "сер.", "mean": "сер.", "games": "матчів", "played": "зіграно", "left": "залишилось",
    "progress": "прогрес сезону", "gap": "відрив", "top": "верх", "bottom": "низ", "streak": "серія",
    "mom": "динаміка", "over": "понад очікування", "absdiff": "абс. різниця", "true": "справді",
    "inter": "між", "confed": "конфедерація",
    "cat": "", "f": "",
}
# Ознаки, для яких автоматичний переклад звучить незграбно
EXTRA_NAMES = {
    "f_home_true_home": "Господарі справді грають удома",
    "f_away_true_home": "Гості фактично грають удома",
    "f_inter_confed": "Матч між різними конфедераціями",
    "f_elo_absdiff": "Абсолютна різниця Elo",
    "f_k_factor": "Вага турніру (K-фактор Elo)",
    "f_tm_home_value_log": "Господарі: вартість складу (log)",
    "f_tm_away_value_log": "Гості: вартість складу (log)",
    "f_tm_value_log_diff": "Різниця log-вартості складів",
    "f_pi_exp_gd": "Очікувана різниця голів за pi-рейтингами",
    "f_home_form_ewm": "Господарі: зважена форма (свіжі матчі важать більше)",
    "f_away_form_ewm": "Гості: зважена форма (свіжі матчі важать більше)",
}
MODEL_FAILURE = ("Не вдалося порахувати прогноз ({err}). Ймовірно, моделі саме перенавчаються — "
                 "оновіть сторінку за хвилину.")
DEFAULTS = {
    "clubs": {"competition": "E0", "home": "Arsenal", "away": "Chelsea"},
    "intl": {"competition": "FIFA World Cup qualification", "home": "Ukraine", "away": "England"},
}


# ---------------------------------------------------------------- дані (кешовані)

def ordered_competitions(track: str, comps: list[str]) -> list[str]:
    """Для клубів — топ-5 ліг першими, решта у вихідному порядку."""
    if track != "clubs":
        return comps
    top = [c for c in config.TOP5 if c in comps]
    return top + [c for c in comps if c not in top]


@st.cache_data(show_spinner="Завантажую довідники…")
def get_competitions(track: str) -> list[str]:
    """Ліги/турніри треку для форми прогнозу."""
    return ordered_competitions(track, list(pr.competitions(track)))


@st.cache_data(show_spinner=False)
def get_teams(track: str, competition: str) -> list[str]:
    """Команди для вибору: клуби — що грали в лізі, збірні — усі."""
    return list(pr.teams(track, competition if track == "clubs" else None))


@st.cache_data(show_spinner="Рахую прогноз…")
def get_prediction(home, away, date, competition, track, odds, neutral, strict=False, country=None,
                   version=None) -> pr.Prediction:
    """Кешований прогноз (однаковий ввід → миттєва відповідь); version — відбиток файлів моделей."""
    return pr.predict_details(home, away, date, competition, track, odds, neutral, strict, country)


@st.cache_data(show_spinner=False)
def get_recent(track: str, team: str, date) -> pd.DataFrame:
    """Останні 5 матчів команди до дати."""
    return pr.recent_matches(track, team, date)


@st.cache_data(show_spinner=False)
def get_h2h(track: str, home: str, away: str, date) -> pd.DataFrame:
    """Очні зустрічі до дати."""
    return pr.head_to_head(track, home, away, date)


@st.cache_data(show_spinner="Рахую прогнози для тестових матчів…")
def get_backtest(track: str, competition: str, n: int, variant: str = "hist_odds", version=None) -> pd.DataFrame:
    """Прогнози моделі для останніх n тестових матчів турніру; version — відбиток файлів моделей."""
    return pr.backtest(track, competition, n, variant)


@st.cache_resource(show_spinner=False)
def warm_up() -> bool:
    """Один раз на процес прогріває кеші історії матчів (Elo, pi-рейтинги), щоб перший прогноз був швидким."""
    pr.warmup()
    return True


@st.cache_data(show_spinner=False)
def get_countries() -> list[str]:
    """Країни проведення для матчів збірних."""
    return list(pr.venue_countries())


@st.cache_data(show_spinner=False)
def get_test_competitions(track: str) -> list[str]:
    """Турніри тестового періоду."""
    return ordered_competitions(track, pr.test_competitions(track))


@st.cache_data(show_spinner=False)
def get_metrics() -> pd.DataFrame | None:
    """reports/metrics.csv або None, якщо ще не згенеровано."""
    path = config.REPORTS / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------- дрібні помічники

def competition_label(track: str, code: str) -> str:
    """Людська назва ліги для селектора."""
    if track == "clubs" and code in LEAGUE_NAMES:
        return f"{LEAGUE_NAMES[code]} ({code})"
    return code


def default_index(options: list[str], value: str, fallback: int = 0) -> int:
    """Індекс value у options або fallback."""
    return options.index(value) if value in options else min(fallback, max(len(options) - 1, 0))


def feature_label(col: str) -> str:
    """Людська назва f_-ознаки: з таблиць ключових ознак або автоматичний переклад токенів."""
    for track in TRACK_LABELS:
        for label, tpl in MATCH_FEATURES[track]:
            if tpl == col:
                return label
        for label, tpl in TEAM_FEATURES[track]:
            for side, who in (("home", "господарі"), ("away", "гості")):
                if tpl.format(side=side) == col:
                    return f"{label} — {who}"
    if col in EXTRA_NAMES:
        return EXTRA_NAMES[col]
    words = []
    for tok in col.split("_"):
        m = None if tok in TOKENS else re.fullmatch(r"([a-z]+?)(\d+)", tok)
        word, num = (m.group(1), m.group(2)) if m and m.group(1) in TOKENS else (tok, "")
        words.append(TOKENS.get(word, word) + (f" ({num})" if num else ""))
    text = " ".join(w for w in words if w)
    return text[:1].upper() + text[1:] if text else col


def _fmt(value) -> str:
    """Число для таблиці: '—' для пропуску."""
    try:
        return "—" if pd.isna(value) else f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def proba_chart(proba: dict[str, float], height: int = 300) -> alt.Chart:
    """Стовпчиковий графік ймовірностей H/D/A з підписами у відсотках."""
    df = pd.DataFrame({"Результат": [OUTCOME_LABELS[k] for k in proba],
                       "Ймовірність": list(proba.values()),
                       "Колір": [OUTCOME_COLORS[k] for k in proba]})
    base = alt.Chart(df).encode(
        x=alt.X("Результат:N", sort=list(OUTCOME_LABELS.values()), axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("Ймовірність:Q", axis=alt.Axis(format="%", title=None), scale=alt.Scale(domain=[0, 1])))
    bars = base.mark_bar().encode(color=alt.Color("Колір:N", scale=None))
    text = base.mark_text(dy=-10, fontSize=16, fontWeight="bold").encode(text=alt.Text("Ймовірність:Q", format=".1%"))
    return (bars + text).properties(height=height)


def compare_table(model: dict[str, float], market: dict[str, float]) -> pd.DataFrame:
    """Модель vs ринок: імовірності без маржі поруч і різниця у відсоткових пунктах."""
    return pd.DataFrame({
        "Модель": [f"{model[k]:.1%}" for k in config.CLASSES],
        "Букмекер (без маржі)": [f"{market[k]:.1%}" for k in config.CLASSES],
        "Різниця, п.п.": [f"{(model[k] - market[k]) * 100:+.1f}" for k in config.CLASSES],
    }, index=[OUTCOME_LABELS[k] for k in config.CLASSES])


# ---------------------------------------------------------------- ввід

def match_inputs(track: str) -> dict:
    """Форма вводу матчу в боковій панелі; повертає параметри для predict_details."""
    defaults = DEFAULTS[track]
    comps = get_competitions(track)
    competition = st.sidebar.selectbox(
        "Ліга" if track == "clubs" else "Турнір", comps,
        index=default_index(comps, defaults["competition"]),
        format_func=lambda c: competition_label(track, c), key=f"comp_{track}")
    team_list = get_teams(track, competition)
    if len(team_list) < 2:
        st.sidebar.warning("Для цієї ліги замало команд в історії")
        return {}
    home = st.sidebar.selectbox("Господарі", team_list, key=f"home_{track}_{competition}",
                                index=default_index(team_list, defaults["home"], 0))
    away = st.sidebar.selectbox("Гості", team_list, key=f"away_{track}_{competition}",
                                index=default_index(team_list, defaults["away"], 1))
    date = st.sidebar.date_input("Дата матчу", value=dt.date.today(), min_value=MIN_DATE,
                                 max_value=MAX_DATE, key="date")
    params = {"home": home, "away": away, "date": pd.Timestamp(date), "competition": competition,
              "track": track, "odds": None, "neutral": False, "strict": False, "country": None}
    if track == "clubs":
        params["odds"] = odds_inputs()
        params["strict"] = st.sidebar.radio(
            "Модель без коефіцієнтів", [False, True], key="strict", horizontal=True,
            format_func=lambda v: "сувора" if v else "з історією ринку",
            help="«З історією ринку» враховує коефіцієнти БК у минулих матчах команд; "
                 "«сувора» не використовує жодних коефіцієнтів. Якщо введено коефіцієнти цього матчу — "
                 "використовується модель з коефіцієнтами.")
    else:
        params["neutral"] = st.sidebar.checkbox("Нейтральне поле", value=False)
        if params["neutral"]:
            options = ["(не задано)"] + get_countries()
            pick = st.sidebar.selectbox("Країна проведення (необов'язково)", options, key="country")
            params["country"] = None if pick == options[0] else pick
    return params


def odds_inputs() -> tuple[float, float, float] | None:
    """Необов'язкові коефіцієнти БК; повертає трійку лише якщо заповнено всі три."""
    with st.sidebar.expander("Коефіцієнти букмекера (необов'язково)"):
        st.caption("Залиште 0, щоб прогнозувати без коефіцієнтів")
        cols = st.columns(3)
        odds = tuple(col.number_input(lbl, min_value=0.0, max_value=1000.0, value=0.0, step=0.05,
                                      format="%.2f", key=f"odds_{lbl}")
                     for col, lbl in zip(cols, ("П1", "Х", "П2")))
    if all(o > 1.0 for o in odds):
        return odds
    if any(o > 0 for o in odds):
        st.sidebar.warning("Заповніть усі три коефіцієнти (> 1), інакше вони не враховуються")
    return None


# ---------------------------------------------------------------- вкладка «Прогноз»

def feature_tables(res: pr.Prediction, home: str, away: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Таблиці ключових ознак: по командах і для матчу загалом (лише наявні колонки)."""
    row = res.features.iloc[0]
    def value(tpl: str, side: str) -> str:
        raw = row[tpl.format(side=side)]
        convert = VALUE_TRANSFORMS.get(tpl)
        return _fmt(convert(raw) if convert and pd.notna(raw) else raw)

    team_rows = [{"Показник": lbl, home: value(tpl, "home"), away: value(tpl, "away")}
                 for lbl, tpl in TEAM_FEATURES[res.track]
                 if tpl.format(side="home") in row.index and tpl.format(side="away") in row.index
                 and row[[tpl.format(side="home"), tpl.format(side="away")]].notna().any()]
    match_rows = [{"Показник": lbl, "Значення": _fmt(row[col])}
                  for lbl, col in MATCH_FEATURES[res.track] if col in row.index]
    return pd.DataFrame(team_rows), pd.DataFrame(match_rows)


def all_features_table(res: pr.Prediction) -> pd.DataFrame:
    """Усі ознаки рядка з людськими назвами (для експандера)."""
    row = res.features.iloc[0]
    return pd.DataFrame({"Ознака": [feature_label(c) for c in row.index],
                         "Код": list(row.index), "Значення": [_fmt(v) for v in row.values]})


def history_warning(res: pr.Prediction, home: str, away: str) -> None:
    """Попередження, якщо в команди мало історії (немає форми або Elo)."""
    row = res.features.iloc[0]
    form_tpl = "f_{side}_form5" if res.track == "clubs" else "f_{side}_pts5"
    for side, team in (("home", home), ("away", away)):
        elo_col = f"f_own_elo_{side}" if f"f_own_elo_{side}" in row.index else f"f_elo_{side}"
        cols = [c for c in (elo_col, form_tpl.format(side=side)) if c in row.index]
        if any(pd.isna(row[c]) for c in cols):
            st.warning(f"Для «{team}» бракує історії (немає Elo або форми до цієї дати) — прогноз менш надійний.")


def show_recent(params: dict) -> None:
    """Останні 5 матчів кожної команди та очні зустрічі (строго до дати)."""
    track, home, away, date = params["track"], params["home"], params["away"], params["date"]
    st.markdown("#### Останні 5 матчів (до обраної дати)")
    for col, team in zip(st.columns(2), (home, away)):
        col.markdown(f"**{team}**")
        recent = get_recent(track, team, date)
        if recent.empty:
            col.info("Матчів до цієї дати немає")
        else:
            col.dataframe(recent, hide_index=True, width="stretch")
    st.markdown("#### Очні зустрічі")
    h2h = get_h2h(track, home, away, date)
    if h2h.empty:
        st.info("Команди раніше не зустрічалися")
    else:
        st.dataframe(h2h, hide_index=True, width="stretch")


def show_prediction(params: dict) -> None:
    """Головний блок: графік, висновок, чесні коефіцієнти, ринок, ознаки, історія."""
    home, away = params["home"], params["away"]
    st.subheader(f"{home} — {away}")
    if home == away:
        st.warning("Оберіть дві різні команди.")
        return
    try:
        with st.spinner("Рахую прогноз…"):
            res = get_prediction(**params, version=pr.models_version())
    except ValueError as err:
        st.warning(str(err))
        return
    except Exception as err:  # noqa: BLE001 — демо не має падати через збій моделі
        st.error(MODEL_FAILURE.format(err=err))
        return
    history_warning(res, home, away)
    proba = res.proba
    best = max(proba, key=proba.get)
    left, right = st.columns([3, 2])
    left.altair_chart(proba_chart(proba), width="stretch")
    with right:
        st.metric("Найімовірніше", OUTCOME_LABELS[best], f"{proba[best]:.1%}", delta_color="off")
        st.markdown("**«Чесні» коефіцієнти (1/p, без маржі):**")
        st.dataframe(pd.DataFrame({"Ймовірність": [f"{p:.1%}" for p in proba.values()],
                                   "Коефіцієнт": [f"{1 / p:.2f}" for p in proba.values()]},
                                  index=[OUTCOME_LABELS[k] for k in proba]), width="stretch")
        st.caption(f"Модель: {res.model_label}")
    if params.get("odds"):
        st.markdown("#### Модель vs ринок")
        st.dataframe(compare_table(proba, pr.market_proba(params["odds"])), width="stretch")
        st.caption("Додатна різниця — модель оцінює результат вище, ніж букмекер (потенційна «цінність»).")
    teams, match = feature_tables(res, home, away)
    st.markdown("#### Ключові ознаки (лише з матчів до обраної дати)")
    c1, c2 = st.columns(2)
    c1.dataframe(teams, hide_index=True, width="stretch")
    c2.dataframe(match, hide_index=True, width="stretch")
    with st.expander(f"Усі ознаки моделі ({res.features.shape[1]})"):
        st.dataframe(all_features_table(res), hide_index=True, width="stretch")
    show_recent(params)


# ---------------------------------------------------------------- вкладка «Перевірка на історії»

def _row_proba(row: pd.Series, prefix: str) -> dict[str, float] | None:
    """Імовірності {H,D,A} з колонок prefix_H/D/A рядка бектесту (None, якщо немає)."""
    cols = [f"{prefix}_{k}" for k in config.CLASSES]
    if not all(c in row.index for c in cols) or row[cols].isna().any():
        return None
    return {k: float(row[f"{prefix}_{k}"]) for k in config.CLASSES}


def backtest_summary(bt: pd.DataFrame) -> None:
    """Зведення: скільки вгадала модель, букмекер і наївне «завжди господарі»."""
    n = len(bt)
    cols = st.columns(3)
    cols[0].metric("Модель вгадала", f"{int(bt['hit'].sum())} з {n}", f"{bt['hit'].mean():.0%}", delta_color="off")
    if "m_H" in bt.columns and bt["m_H"].notna().any():
        has = bt.dropna(subset=["m_H", "m_D", "m_A"])
        fav = has[["m_H", "m_D", "m_A"]].to_numpy().argmax(axis=1)
        hits = int((pd.Series(config.CLASSES).iloc[fav].to_numpy() == has["target"].to_numpy()).sum())
        cols[1].metric("Фаворит букмекера вгадав", f"{hits} з {len(has)}", f"{hits / len(has):.0%}", delta_color="off")
    home_hits = int((bt["target"] == "H").sum())
    cols[2].metric("«Завжди господарі»", f"{home_hits} з {n}", f"{home_hits / n:.0%}", delta_color="off")


def backtest_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Таблиця всіх матчів бектесту для показу."""
    return pd.DataFrame({
        "Дата": bt["date"].dt.strftime("%d.%m.%Y"),
        "Матч": bt["home"] + " — " + bt["away"],
        "Рахунок": bt["home_goals"].astype(int).astype(str) + ":" + bt["away_goals"].astype(int).astype(str),
        "П1": bt["p_H"].map("{:.0%}".format), "Х": bt["p_D"].map("{:.0%}".format), "П2": bt["p_A"].map("{:.0%}".format),
        "Прогноз": bt["pred"].map(OUTCOME_LABELS), "Факт": bt["target"].map(OUTCOME_LABELS),
        "Вгадано": bt["hit"].map({True: "✅", False: "❌"}),
    })


def show_backtest_match(row: pd.Series) -> None:
    """Один історичний матч: фактичний рахунок, прогноз моделі, імовірності букмекера."""
    score = f"{int(row['home_goals'])}:{int(row['away_goals'])}"
    st.subheader(f"{row['home']} {score} {row['away']}")
    model = _row_proba(row, "p")
    verdict = "✅ вгадала" if row["hit"] else "❌ не вгадала"
    st.caption(f"{row['date']:%d.%m.%Y} · факт: {OUTCOME_LABELS[row['target']]} · "
               f"модель ({verdict}): {OUTCOME_LABELS[row['pred']]}")
    left, right = st.columns([3, 2])
    left.altair_chart(proba_chart(model, height=260), width="stretch")
    market = _row_proba(row, "m")
    with right:
        if market:
            st.markdown("**Модель vs букмекер (передматчеві коефіцієнти):**")
            st.dataframe(compare_table(model, market), width="stretch")
        else:
            st.markdown("**Прогноз моделі:**")
            st.dataframe(pd.DataFrame({"Ймовірність": [f"{model[k]:.1%}" for k in config.CLASSES]},
                                      index=[OUTCOME_LABELS[k] for k in config.CLASSES]), width="stretch")


def show_backtest(track: str) -> None:
    """Вкладка «Перевірка на історії»: реальні матчі тестового періоду vs прогноз моделі."""
    st.markdown(f"Матчі з {pd.Timestamp(config.TEST_START):%d.%m.%Y} модель **не бачила** під час навчання. "
                "Для кожного матчу ознаки пораховано лише з ігор, що були **до** нього.")
    comps = get_test_competitions(track)
    if not comps:
        st.info("Немає матчів тестового періоду")
        return
    c1, c2 = st.columns([3, 1])
    competition = c1.selectbox("Ліга" if track == "clubs" else "Турнір", comps, key=f"bt_comp_{track}",
                               index=default_index(comps, DEFAULTS[track]["competition"]),
                               format_func=lambda c: competition_label(track, c))
    n = c2.selectbox("Останніх матчів", [10, 20, 50, 100], index=1, key=f"bt_n_{track}")
    variant = "no_odds"
    if track == "clubs":
        variants = [v for v in ("hist_odds", "no_odds", "with_odds") if pr.model_exists(track, v)] or ["no_odds"]
        variant = st.radio("Модель", variants, horizontal=True, key="bt_variant",
                           format_func=lambda v: pr.MODEL_LABELS[("clubs", v)])
    try:
        bt = get_backtest(track, competition, n, variant, version=pr.models_version())
    except ValueError as err:
        st.warning(str(err))
        return
    except Exception as err:  # noqa: BLE001 — демо не має падати через збій моделі
        st.error(MODEL_FAILURE.format(err=err))
        return
    if bt.empty:
        st.info("Для цього турніру немає матчів у тестовому періоді")
        return
    backtest_summary(bt)
    labels = [f"{d:%d.%m.%Y}  {h} {int(hg)}:{int(ag)} {a}"
              for d, h, a, hg, ag in bt[["date", "home", "away", "home_goals", "away_goals"]].itertuples(index=False)]
    pick = st.selectbox("Матч", range(len(bt)), format_func=labels.__getitem__, key=f"bt_match_{track}_{competition}_{n}")
    show_backtest_match(bt.iloc[pick])
    st.markdown("#### Усі матчі вибірки")
    st.dataframe(backtest_table(bt), hide_index=True, width="stretch")


# ---------------------------------------------------------------- вкладка «Якість моделі»

def _test_row(metrics: pd.DataFrame, track: str, **cond) -> pd.Series | None:
    """Рядок metrics.csv для треку на тесті за умовами колонок (None, якщо немає)."""
    m = metrics[(metrics["track"] == track) & (metrics["split"] == "test")]
    for col, val in cond.items():
        m = m[m[col] == val]
    return m.iloc[0] if len(m) else None


def quality_cards(metrics: pd.DataFrame) -> None:
    """Великі показники: точність і log-loss фінальних моделей проти «завжди господарі»."""
    cards = []
    for track, feature_set in (("clubs", "no_odds"), ("clubs", "hist_odds"), ("clubs", "with_odds"),
                               ("intl", "no_odds")):
        final = _test_row(metrics, track, final=True, feature_set=feature_set)
        base = _test_row(metrics, track, model="baseline_home")
        if final is not None and base is not None:
            cards.append((track, feature_set, final, base))
    if not cards:
        st.info("У metrics.csv немає фінальних моделей")
        return
    for (track, feature_set, final, base), col in zip(cards, st.columns(len(cards))):
        title = TRACK_LABELS[track] + {"no_odds": "", "hist_odds": " (історія ринку)",
                                       "with_odds": " + коеф. БК"}[feature_set]
        col.metric(f"{title}: точність", f"{final['accuracy']:.1%}",
                   f"{(final['accuracy'] - base['accuracy']) * 100:+.1f} п.п. до «завжди господарі»")
        col.metric(f"{title}: log-loss", f"{final['log_loss']:.3f}",
                   f"{final['log_loss'] - base['log_loss']:+.3f} до «завжди господарі»", delta_color="inverse")
    st.caption("Тестовий період: матчі з 01.07.2023. Log-loss — менше краще (оцінює якість імовірностей).")


def show_quality() -> None:
    """Вкладка «Якість моделі»: показники, таблиця метрик і графіки з reports/."""
    metrics = get_metrics()
    if metrics is None:
        st.info("Метрики ще не пораховані — запустіть python -m src.run_all")
    else:
        try:
            quality_cards(metrics)
        except (KeyError, TypeError, ValueError):
            st.info("Формат metrics.csv змінився — показую лише таблицю")
        st.markdown("#### Усі моделі на тесті")
        test = metrics[metrics["split"] == "test"] if "split" in metrics.columns else metrics
        st.dataframe(test.round(4), hide_index=True, width="stretch")
    errors = config.REPORTS / "error_analysis.md"
    with st.expander("Аналіз помилок моделі (error_analysis.md)"):
        if errors.exists():
            st.markdown(errors.read_text(encoding="utf-8"))
            for fig in sorted(Path(config.FIGURES).glob("err_*.png")):
                st.image(str(fig), caption=fig.stem, width="stretch")
        else:
            st.info("Аналіз помилок ще не згенеровано")
    md = config.REPORTS / "metrics.md"
    if md.exists():
        with st.expander("Звіт metrics.md"):
            st.markdown(md.read_text(encoding="utf-8"))
    figures = sorted((f for f in Path(config.FIGURES).glob("*.png") if not f.name.startswith("err_")),
                     key=lambda p: (p.name.startswith("eda_"), p.name))
    if not figures:
        st.info("Графіків у reports/figures/ поки немає")
    for fig in figures:
        st.image(str(fig), caption=fig.stem, width="stretch")


# ---------------------------------------------------------------- точка входу

def main() -> None:
    """Точка входу Streamlit."""
    st.set_page_config(page_title="Прогноз футбольного матчу", page_icon="⚽", layout="wide")
    st.title("⚽ Прогноз результату футбольного матчу")
    st.caption("Ймовірності перемоги господарів / нічиєї / перемоги гостей на основі Elo, форми та історії зустрічей")
    with st.spinner("Завантажую історію матчів, рейтинги Elo та pi… (лише під час першого запуску, ~15 с)"):
        warm_up()
    track = st.sidebar.radio("Трек", list(TRACK_LABELS), format_func=TRACK_LABELS.get, horizontal=True, key="track")
    params = match_inputs(track)
    tab_pred, tab_bt, tab_quality = st.tabs(["Прогноз", "Перевірка на історії", "Якість моделі"])
    with tab_pred:
        if params:
            show_prediction(params)
    with tab_bt:
        show_backtest(track)
    with tab_quality:
        show_quality()


main()
