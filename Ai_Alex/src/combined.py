"""Об'єднана модель: наші ознаки + ознаки стартових складів із моделі учасника (model2_final.py).

model2_final.py будує для кожного матчу Transfermarkt Elo гравців і з нього — силу СТАРТОВОГО складу,
відсутніх гравців, ротацію. Стартовий склад оголошують приблизно за годину до гри, тож це передматчева
інформація, але пізніша за ту, якою користуються інші наші моделі. Тому це окремий набір ознак.

Скрипт чесно порівнює на ОДНИХ І ТИХ САМИХ матчах тесту (де є обидва джерела):
наші ознаки, ознаки учасника, обидва разом. Запуск: python -m src.combined
"""
import importlib.util
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src import config
from src.features_tm import TM_DIR, _map_clubs

LINEUP_PREFIX = "f_lu_"          # ознаки з моделі учасника
MODEL2_FILE = "model2_final.py"  # лежить у корені репозиторію команди (на рівень вище) або поруч
COMBINED = "ОБ'ЄДНАНА: наші + склади"
LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.03, "num_leaves": 15,
              "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 5.0, "max_bin": 127, "num_threads": 8, "seed": config.RANDOM_STATE,
              "deterministic": True, "force_col_wise": True, "verbose": -1}
MAX_ROUNDS, EARLY_STOPPING = 2000, 100
ODDS_COLS = ["f_odds_p_home", "f_odds_p_draw", "f_odds_p_away"]


def load_model2():
    """Імпортує модуль учасника за шляхом (він не входить у пакет src)."""
    for folder in (config.ROOT, config.ROOT.parent):
        path = folder / MODEL2_FILE
        if path.exists():
            spec = importlib.util.spec_from_file_location("model2_final", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["model2_final"] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"{MODEL2_FILE} не знайдено поруч із проєктом або на рівень вище")


def load_tm_games_with_lineups() -> pd.DataFrame:
    """Таблиця матчів Transfermarkt зі списками стартових і запасних — те саме, що SQL у model2_final."""
    games = pd.read_csv(TM_DIR / "games.csv.gz", low_memory=False, parse_dates=["date"])
    clubs = pd.read_csv(TM_DIR / "clubs.csv.gz", usecols=["club_id", "name"])
    lineups = pd.read_csv(TM_DIR / "game_lineups.csv.gz", usecols=["game_id", "club_id", "player_id", "type"],
                          low_memory=False).dropna(subset=["player_id"])
    lineups["player_id"] = lineups["player_id"].astype(str)
    lists = (lineups.drop_duplicates(["game_id", "club_id", "player_id", "type"])
             .groupby(["game_id", "club_id", "type"])["player_id"].agg(list).unstack("type"))
    lists = lists.rename(columns={"starting_lineup": "starters"}).reindex(columns=["starters", "substitutes"])
    lists = lists.reset_index()
    names = clubs.set_index("club_id")["name"]
    known = set(names.index)
    g = games[games["home_club_id"].isin(known) & games["away_club_id"].isin(known) & games["date"].notna()]
    g = g.assign(home_club_name=g["home_club_id"].map(names), away_club_name=g["away_club_id"].map(names))
    for side in ("home", "away"):
        side_lists = lists.rename(columns={"club_id": f"{side}_club_id", "starters": f"{side}_starters",
                                           "substitutes": f"{side}_substitutes"})
        g = g.merge(side_lists, on=["game_id", f"{side}_club_id"], how="left")
    keep = ["game_id", "date", "competition_id", "season", "home_club_id", "away_club_id", "home_club_name",
            "away_club_name", "home_club_goals", "away_club_goals", "home_starters", "home_substitutes",
            "away_starters", "away_substitutes"]
    return g[keep].sort_values("date").reset_index(drop=True)


def lineup_features(model2, tm: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    """Ознаки учасника для кожного матчу, з ключами (date, home, away) у наших назвах команд."""
    feats, *_ = model2.build_features(model2.prepare(tm))
    cols = [c for c in feats.columns if c not in model2.META]
    games = pd.read_csv(TM_DIR / "games.csv.gz", low_memory=False, parse_dates=["date"])
    club_map = _map_clubs(games, ours)
    out = feats[feats["has_lineups"]].copy()
    out["home"] = out["home"].map(club_map)
    out["away"] = out["away"].map(club_map)
    out = out.dropna(subset=["home", "away"]).rename(columns={c: LINEUP_PREFIX + c for c in cols})
    out = out[["date", "home", "away"] + [LINEUP_PREFIX + c for c in cols]]
    return out.drop_duplicates(["date", "home", "away"])


def scores(y: np.ndarray, proba: np.ndarray) -> dict:
    """Accuracy і log-loss для таблиці порівняння."""
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {"accuracy": accuracy_score(y, proba.argmax(axis=1)), "log_loss": log_loss(y, proba, labels=[0, 1, 2])}


def fit_predict(train, val, test, features):
    """LightGBM з ранньою зупинкою на validation; повертає модель і прогнози на test та validation."""
    data = lgb.Dataset(train[features], train["y"])
    booster = lgb.train(LGB_PARAMS, data, MAX_ROUNDS,
                        valid_sets=[lgb.Dataset(val[features], val["y"], reference=data)],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    best = booster.best_iteration
    return booster, booster.predict(test[features], num_iteration=best), booster.predict(val[features], num_iteration=best)


def build_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Наші ознаки клубів, з'єднані з ознаками складів за (дата, господарі, гості)."""
    ours = pd.read_parquet(config.PROCESSED / "clubs_matches.parquet")
    feats = pd.read_parquet(config.PROCESSED / "clubs_features.parquet")
    lineup = lineup_features(load_model2(), load_tm_games_with_lineups(), ours)
    for col in ("home", "away"):
        lineup[col] = lineup[col].astype(feats[col].dtype)
    df = feats.merge(lineup, on=["date", "home", "away"], how="inner").sort_values(["date", "match_id"])
    lineup_cols = [c for c in df.columns if c.startswith(LINEUP_PREFIX)]
    own_cols = [c for c in feats.columns if c.startswith("f_") and not c.startswith("f_odds_")]
    return df, own_cols, lineup_cols


def write_report(table: pd.DataFrame, sizes: tuple[int, int, int], gain: pd.Series) -> None:
    """Зберігає таблицю порівняння та пояснення в reports/combined_metrics.md."""
    share = gain[[c for c in gain.index if c.startswith(LINEUP_PREFIX)]].sum() / gain.sum()
    top = (gain / gain.sum()).head(15).round(4).rename("частка gain").to_frame()
    text = [
        "# Об'єднана модель: наші ознаки + стартові склади (модель учасника)", "",
        "Модель учасника (`model2_final.py`) рахує Elo гравців і силу стартового складу за даними Transfermarkt. "
        "Ми взяли її ознаки як є і додали до наших. Усі варіанти навчено однаково (LightGBM, рання зупинка на "
        "validation) і оцінено на тих самих матчах тесту, де є обидва джерела.", "",
        f"Спільних матчів: train {sizes[0]}, validation {sizes[1]}, test {sizes[2]} "
        f"(розбиття за часом, test з {config.TEST_START}).", "",
        table.round(4).to_markdown(index=False), "",
        f"Частка важливості (gain) ознак складів в об'єднаній моделі: {share:.1%}.", "",
        "Найважливіші ознаки об'єднаної моделі:", "", top.to_markdown(), "",
        "**Важливо.** Ознаки складів використовують стартовий склад самого матчу. Його оголошують приблизно за "
        "годину до початку, тож це не витік, але прогноз із ними можливий лише після оголошення складів. "
        "Порівняння стосується тільки ліг і сезонів, де Transfermarkt має склади (переважно топ-ліги).",
    ]
    (config.REPORTS / "combined_metrics.md").write_text("\n".join(text), encoding="utf-8")
    table.to_csv(config.REPORTS / "combined_metrics.csv", index=False)


def main() -> None:
    """Будує об'єднану таблицю, навчає варіанти на однакових матчах і пише звіт."""
    started = time.time()
    df, own_cols, lineup_cols = build_table()
    train = df[df["date"] < config.VAL_START]
    val = df[(df["date"] >= config.VAL_START) & (df["date"] < config.TEST_START)]
    test = df[df["date"] >= config.TEST_START]
    print(f"[combined] спільних матчів: train {len(train)}, val {len(val)}, test {len(test)}; "
          f"ознак складів {len(lineup_cols)} ({time.time() - started:.0f} с)")

    odds_all = [c for c in df.columns if c.startswith("f_odds_")]
    feature_sets = {"наші ознаки (з історією ринку)": own_cols,
                    "ознаки складів (модель учасника)": lineup_cols,
                    COMBINED: own_cols + lineup_cols,
                    "наші + коефіцієнти БК": own_cols + odds_all,
                    "об'єднана + коефіцієнти БК": own_cols + lineup_cols + odds_all}
    y_test, y_val = test["y"].to_numpy(), val["y"].to_numpy()
    rows, boosters = [], {}
    for name, cols in feature_sets.items():
        boosters[name], p_test, p_val = fit_predict(train, val, test, cols)
        rows.append({"модель": name, "ознак": len(cols), **scores(y_test, p_test), "val_log_loss": scores(y_val, p_val)["log_loss"]})

    prior = np.bincount(train["y"], minlength=3) / len(train)
    market = test[ODDS_COLS].to_numpy()
    market = np.where(np.isnan(market).any(axis=1, keepdims=True), prior, market)
    rows.append({"модель": "ринок (коефіцієнти БК)", "ознак": 3, **scores(y_test, market)})
    rows.append({"модель": "завжди господарі", "ознак": 0, **scores(y_test, np.tile(prior, (len(test), 1)))})
    table = pd.DataFrame(rows)

    best = boosters[COMBINED]
    gain = pd.Series(best.feature_importance("gain"), index=best.feature_name()).sort_values(ascending=False)
    joblib.dump({"booster": best, "features": feature_sets[COMBINED], "best_iteration": best.best_iteration},
                config.MODELS / "clubs_combined.joblib")
    write_report(table, (len(train), len(val), len(test)), gain)
    print(table.round(4).to_string(index=False))
    print(f"[combined] готово за {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
