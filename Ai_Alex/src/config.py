"""Спільна конфігурація проєкту: шляхи, дати розбиття, константи."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RANDOM_STATE = 42
MIN_DATE = "2000-01-01"          # фільтр для обох треків
VAL_START = "2022-07-01"         # train: date < VAL_START
TEST_START = "2023-07-01"        # validation: [VAL_START, TEST_START), test: >= TEST_START
LEAK_ACC_LIMIT = 0.60            # test accuracy вище цього = підозра на витік

CLASSES = ["H", "D", "A"]        # порядок класів усюди: 0=H, 1=D, 2=A
TOP5 = {"E0": "АПЛ", "SP1": "Ла Ліга", "D1": "Бундесліга", "I1": "Серія A", "F1": "Ліга 1"}
TRACKS = ["clubs", "intl"]

_GH = "https://raw.githubusercontent.com"
SOURCES = {
    "Matches.csv": [f"{_GH}/xgabora/Club-Football-Match-Data/main/data/Matches.csv",
                    "https://github.com/xgabora/Club-Football-Match-Data/raw/main/data/Matches.csv"],
    "EloRatings.csv": [f"{_GH}/xgabora/Club-Football-Match-Data/main/data/EloRatings.csv",
                       "https://github.com/xgabora/Club-Football-Match-Data/raw/main/data/EloRatings.csv"],
    "results.csv": [f"{_GH}/martj42/international_results/refs/heads/master/results.csv",
                    f"{_GH}/martj42/international_results/master/results.csv"],
    "shootouts.csv": [f"{_GH}/martj42/international_results/refs/heads/master/shootouts.csv",
                      f"{_GH}/martj42/international_results/master/shootouts.csv"],
    "former_names.csv": [f"{_GH}/martj42/international_results/refs/heads/master/former_names.csv",
                         f"{_GH}/martj42/international_results/master/former_names.csv"],
}

for _d in (RAW, PROCESSED, MODELS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- Тест на витік (версія 2) ---
# Клуби: абсолютний поріг LEAK_ACC_LIMIT (суперники в лізі приблизно рівні, >60% нереально).
# Збірні: абсолютний поріг не працює — багато нерівних пар (Німеччина — Сан-Марино), і навіть
# базова модель «лише Elo» дає ~60%. Тому перевіряємо приріст над нею: справжній витік дав би
# стрибок на 10–20 п.п., чесні ознаки додають 1–2 п.п.
INTL_LEAK_MAX_GAIN_OVER_ELO = 0.05   # макс. дозволений приріст test accuracy над baseline_elo
INTL_LEAK_ABS_LIMIT = 0.70           # запасний абсолютний поріг для збірних
