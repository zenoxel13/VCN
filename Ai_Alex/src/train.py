"""Навчання моделей для обох треків (clubs, intl): базові, логрегресія, LightGBM, ансамбль, калібрування.

Схема (версія 3):
1. Validation ділиться за часом навпіл: va_tune (1-ша половина) — early stopping, вибір варіантів
   LightGBM, ваги ансамблю, калібратор; va_select (2-га) — чесний вибір між кандидатами.
2. LightGBM: випадковий пошук на найсвіжішій частині train з високим learning rate; далі на повному train
   порівнюються варіанти (за log-loss на va_tune): ваги свіжості з різним півперіодом, старт від сильного
   прогнозу (init_score = log ймовірностей ринку / Elo-логрегресії), відкидання ознак з мізерним gain.
3. Логрегресія отримує лог-ймовірності того самого стартового прогнозу як ознаки.
4. Ансамбль LightGBM / логрегресія [/ ринок] з вагами, підібраними сіткою на va_tune.
5. Обраний кандидат донавчається на train+val і зберігається в models/*.joblib (рядок final_<набір>).

Результати: models/<файл>.joblib (FinalModel), data/processed/preds_<track>.parquet,
models/train_summary_<track>.json, reports/feature_importance_<track>.csv.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.modeling import (
    AnchorModel, EnsembleModel, FinalModel, IsotonicCalibrator, LGBMModel, LogitCalibrator, LogRegModel,
    OddsModel, PriorModel, category_levels, choose_calibrator, compute_metrics, feature_sets, recency_weights, split_by_time,
)

N_TRIALS = 20                # спроб випадкового пошуку LightGBM на кожен набір (3 набори клубів — бюджет часу)
SEARCH_FRACTION = 0.30       # частка найсвіжіших рядків train для пошуку гіперпараметрів
SEARCH_LR = 0.1              # learning rate під час пошуку (швидко)
SEARCH_MAX_ROUNDS = 400
FINAL_LR = 0.1               # learning rate для варіантів на повному train
FINAL_MAX_ROUNDS = 800
EARLY_STOPPING = 30
REFIT_ROUNDS_FACTOR = 1.1    # дерев при донавчанні на train+val = best_iter × цей множник
TIME_BUDGET_S = 150          # запобіжник на (єдиний) пошук гіперпараметрів треку
TOTAL_BUDGET_S = 480         # загальний бюджет train: коли вичерпано — необов'язкові кроки пропускаються
SEARCH_SET_PRIORITY = ("hist_odds", "no_odds")  # на якому наборі шукати гіперпараметри (один раз на трек)
FAST = os.environ.get("TRAIN_FAST") == "1"      # швидкий режим перевірки коду: лише свіжі дані
_START = time.time()


def time_left() -> float:
    """Скільки секунд лишилося із загального бюджету TOTAL_BUDGET_S."""
    return TOTAL_BUDGET_S - (time.time() - _START)
HALF_LIVES = (None,)         # півперіоди ваг свіжості (роки); 8.0 не дав виграшу на val і вимкнений для часу
MIN_GAIN_SHARE = 0.0         # відбір ознак за gain (0.002) не дав виграшу на val — вимкнено для часу
LOGREG_TRAIN_FRACTION = 1.0  # частка найсвіжіших рядків train для логрегресії (після фіксу BLAS — увесь train)
ENS_GRID_STEP = 0.1          # крок сітки ваг ансамблю
META_COLS = ["match_id", "date", "y", "competition", "season", "f_cat_tournament_type"]
ANCHOR_ELO_FEATURES = ("f_elo_diff", "f_own_elo_diff", "f_neutral")

PARAM_SPACE = {
    "num_leaves": [7, 15, 31, 63],
    "min_data_in_leaf": [50, 100, 200, 400],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.7, 0.85, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 20.0],
    "max_depth": [-1, 4, 6, 8],
}
BASE_PARAMS = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "bagging_freq": 1,
               "max_bin": 127, "seed": config.RANDOM_STATE, "verbose": -1,
               "num_threads": 8, "deterministic": True, "force_col_wise": True}  # >8 потоків не прискорює


def proba_col(model: str, cls: str) -> str:
    """Назва колонки з ймовірністю класу cls для моделі model у preds_<track>.parquet."""
    return f"p__{model}__{cls}"


def load_track(track: str, data_dir: Path) -> pd.DataFrame | None:
    """Читає <track>_features.parquet; None, якщо файлу ще немає."""
    path = Path(data_dir) / f"{track}_features.parquet"
    if not path.exists():
        print(f"[{track}] файл {path} не знайдено — трек пропущено")
        return None
    df = pd.read_parquet(path).sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if FAST:  # перевірка коду: 3 сезони train + val + test
        df = df[df["date"] >= pd.Timestamp("2019-07-01")].reset_index(drop=True)
    return df


def log_loss_of(y: np.ndarray, proba: np.ndarray) -> float:
    """Скорочення: log-loss."""
    return compute_metrics(y, proba)["log_loss"]


# ----------------------------------------------------------------------------- LightGBM
def fit_lgbm(train: pd.DataFrame, features: list[str], params: dict, levels: dict, rounds: int,
             val: pd.DataFrame | None = None, half_life: float | None = None,
             anchor: AnchorModel | None = None) -> lgb.Booster:
    """Навчає Booster (з вагами свіжості та init_score від anchor, якщо задано); з val — early stopping."""
    def dataset(df: pd.DataFrame, weight=None, reference=None) -> lgb.Dataset:
        init = anchor.log_proba(df) if anchor is not None else None
        return lgb.Dataset(LGBMModel.prepare(df, features, levels), df["y"].to_numpy(),
                           weight=weight, init_score=init, reference=reference)

    ds_tr = dataset(train, recency_weights(train["date"], half_life))
    if val is None:
        return lgb.train(params, ds_tr, num_boost_round=rounds)
    return lgb.train(params, ds_tr, num_boost_round=rounds, valid_sets=[dataset(val, reference=ds_tr)],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])


def search_params(train: pd.DataFrame, val: pd.DataFrame, features: list[str], tag: str) -> tuple[dict, list[dict]]:
    """Випадковий пошук гіперпараметрів на свіжій частині train з lr=SEARCH_LR."""
    levels = category_levels(train, features)
    sub = train.iloc[int(len(train) * (1 - SEARCH_FRACTION)):]
    rng = np.random.RandomState(config.RANDOM_STATE)
    best, trials, t0 = None, [], time.time()
    for i in range(N_TRIALS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"  [{tag}] запобіжник часу спрацював після {i} спроб")
            break
        params = {**BASE_PARAMS, **{k: v[rng.randint(len(v))] for k, v in PARAM_SPACE.items()},
                  "learning_rate": SEARCH_LR}
        booster = fit_lgbm(sub, features, params, levels, SEARCH_MAX_ROUNDS, val)
        score = booster.best_score["valid_0"]["multi_logloss"]
        trials.append({**{k: params[k] for k in PARAM_SPACE}, "best_iter": booster.best_iteration, "val_log_loss": score})
        if best is None or score < best[0]:
            best = (score, params)
    print(f"  [{tag}] пошук: {len(trials)} спроб на {len(sub)} рядках за {time.time() - t0:.0f} с, "
          f"найкращий log-loss (va_tune) {best[0]:.4f}")
    return {**best[1], "learning_rate": FINAL_LR}, trials


def build_lgbm(train: pd.DataFrame, va_tune: pd.DataFrame, features: list[str], anchor: AnchorModel,
               params: dict, tag: str) -> tuple[LGBMModel, dict]:
    """Порівняння варіантів на повному train за va_tune: звичайне навчання vs init_score (старт від anchor).
    Ваги свіжості й відбір ознак вимкнено константами (у v3 не дали приросту на val)."""
    y_tune = va_tune["y"].to_numpy()
    log: dict[str, float] = {}

    def variant(name: str, feats: list[str], half_life, use_anchor: bool) -> tuple[float, LGBMModel]:
        levels = category_levels(train, feats)
        anc = anchor if use_anchor else None
        booster = fit_lgbm(train, feats, params, levels, FINAL_MAX_ROUNDS, va_tune, half_life, anc)
        model = LGBMModel(feats, booster, levels, params, anchor=anc, half_life=half_life)
        log[name] = log_loss_of(y_tune, model.predict_proba(va_tune))
        return log[name], model

    t0 = time.time()
    best = min((variant(f"half_life={hl}", features, hl, False) for hl in HALF_LIVES), key=lambda r: r[0])
    hl = best[1].half_life
    if time_left() > 60:
        with_anchor = variant(f"half_life={hl}+init_score", features, hl, True)
        if with_anchor[0] < best[0]:
            best = with_anchor
    imp = best[1].feature_importance()
    share = imp["gain"] / imp["gain"].sum()
    kept = [f for f in features if f in set(imp.loc[share >= MIN_GAIN_SHARE, "feature"])]
    if 0 < len(kept) < len(features):
        reduced = variant(f"відбір ознак ({len(kept)} з {len(features)})", kept, hl, best[1].anchor is not None)
        if reduced[0] <= best[0]:
            best = reduced
    print(f"  [{tag}] варіанти LightGBM (log-loss на va_tune, {time.time() - t0:.0f} с): "
          + "; ".join(f"{k}={v:.4f}" for k, v in log.items()))
    return best[1], log


# ----------------------------------------------------------------------------- моделі
def baseline_models(df: pd.DataFrame, sets: dict[str, list[str]]) -> dict[str, object]:
    """Базові моделі: «завжди господарі», «лише Elo» (+ f_neutral), «коефіцієнти БК» (якщо є)."""
    elo_feats = [c for c in ("f_elo_diff", "f_neutral") if c in df.columns]
    models = {"baseline_home": PriorModel(), "baseline_elo": LogRegModel(elo_feats)}
    if "with_odds" in sets and all(c in df.columns for c in OddsModel().features):
        models["baseline_odds"] = OddsModel()
    return models


def fit_ensemble_weights(members: dict[str, object], va_tune: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Ваги ансамблю сіткою з кроком ENS_GRID_STEP (сума 1) за log-loss на va_tune."""
    y = va_tune["y"].to_numpy()
    probas = {k: np.asarray(m.predict_proba(va_tune), dtype=float) for k, m in members.items()}
    steps = np.round(np.arange(0, 1 + 1e-9, ENS_GRID_STEP), 2)
    best = (np.inf, None)
    for combo in itertools.product(steps, repeat=len(members) - 1):
        last = round(1 - sum(combo), 2)
        if last < -1e-9:
            continue
        w = {k: float(v) for k, v in zip(members, (*combo, last))}
        score = log_loss_of(y, sum(w[k] * probas[k] for k in members))
        if score < best[0]:
            best = (score, w)
    return best[1], best[0]


def refit(model, data: pd.DataFrame):
    """Донавчає модель того самого типу й конфігурації на нових даних (train+val)."""
    y = data["y"].to_numpy()
    if isinstance(model, EnsembleModel):
        return EnsembleModel({k: refit(m, data) if model.weights[k] > 0 else m for k, m in model.members.items()},
                             model.weights)
    if isinstance(model, LGBMModel):
        rounds = max(1, int(round(model.booster.best_iteration * REFIT_ROUNDS_FACTOR)))
        levels = category_levels(data, model.features)
        booster = fit_lgbm(data, model.features, model.params, levels, rounds,
                           half_life=model.half_life, anchor=model.anchor)
        return LGBMModel(model.features, booster, levels, model.params, model.anchor, model.half_life)
    if isinstance(model, LogRegModel):
        recent = data.iloc[int(len(data) * (1 - LOGREG_TRAIN_FRACTION)):]
        return LogRegModel(model.features, model.C, model.anchor, model.half_life, model.max_iter).fit(
            recent, recent["y"].to_numpy())
    return type(model)().fit(data, y)  # PriorModel / OddsModel


def final_file_name(track: str, set_name: str, sets: dict) -> str:
    """Ім'я файлу фінальної моделі: 'intl' для треку з одним набором, інакше 'clubs_no_odds' тощо."""
    return track if len(sets) == 1 else f"{track}_{set_name}"


def has_clubelo(df: pd.DataFrame) -> pd.Series | None:
    """Ознака «для матчу є рейтинг ClubElo» (клуби): f_clubelo_* якщо є, інакше f_elo_home."""
    cols = [c for c in df.columns if c.startswith("f_clubelo")] or \
        (["f_elo_home"] if "season" in df.columns and "f_elo_home" in df.columns else [])
    return df[cols[0]].notna() if cols else None


class PredsTable:
    """Накопичує прогнози моделей на val+test і реєстр моделей для evaluate."""

    def __init__(self, val: pd.DataFrame, test: pd.DataFrame) -> None:
        meta_cols = [c for c in META_COLS if c in val.columns]
        self.n_val = len(val)
        self.y_val = val["y"].to_numpy()
        self.frame = pd.concat([val[meta_cols].assign(split="val"), test[meta_cols].assign(split="test")],
                               ignore_index=True)
        flag = has_clubelo(pd.concat([val, test], ignore_index=True))
        if flag is not None:
            self.frame["has_clubelo"] = flag.to_numpy()
        for c in self.frame.columns:
            if isinstance(self.frame[c].dtype, pd.CategoricalDtype):
                self.frame[c] = self.frame[c].astype(str)
        self.registry: list[dict] = []

    def add(self, name: str, fset: str, kind: str, proba: np.ndarray) -> float:
        """Додає прогнози (рядки val, потім test) і повертає val log-loss."""
        new = {proba_col(name, cls): proba[:, k] for k, cls in enumerate(config.CLASSES)}
        self.frame = self.frame.assign(**new)
        self.registry.append({"model": name, "feature_set": fset, "kind": kind})
        return log_loss_of(self.y_val, proba[: self.n_val])


def train_feature_set(track: str, set_name: str, feats: list[str], parts: dict, table: PredsTable,
                      elo_anchor: LogRegModel, odds_model: OddsModel | None, file_name: str,
                      models_dir: Path, params: dict) -> tuple[dict, pd.DataFrame]:
    """Кандидати → вибір на va_select → калібрування → донавчання на train+val → збереження."""
    tag = f"{track}/{set_name}"
    tr, va, te = parts["train"], parts["val"], parts["test"]
    y_va = va["y"].to_numpy()
    half = len(va) // 2
    va_tune = va.iloc[:half]
    eval_df = pd.concat([va, te], ignore_index=True)
    anchor = AnchorModel(elo_anchor, use_odds=any(f.startswith("f_odds_") for f in feats))

    lgbm, variants = build_lgbm(tr, va_tune, feats, anchor, params, tag)
    t0 = time.time()
    tr_recent = tr.iloc[int(len(tr) * (1 - LOGREG_TRAIN_FRACTION)):]
    logreg = LogRegModel(feats, anchor=anchor, half_life=lgbm.half_life).fit(tr_recent, tr_recent["y"].to_numpy())
    print(f"  [{tag}] логрегресія (+ лог-ймовірності стартового прогнозу) за {time.time() - t0:.0f} с")
    members = {"lgbm": lgbm, "logreg": logreg}
    if set_name == "with_odds" and odds_model is not None:
        members["odds"] = odds_model
    weights, ens_tune_ll = fit_ensemble_weights(members, va_tune)
    print(f"  [{tag}] ваги ансамблю {weights} (va_tune log-loss {ens_tune_ll:.4f})")
    candidates = {f"logreg_{set_name}": logreg, f"lgbm_{set_name}": lgbm,
                  f"ens_{set_name}": EnsembleModel(members, weights)}

    probas, scores = {}, {}
    for name, model in candidates.items():
        probas[name] = model.predict_proba(eval_df)
        ll = table.add(name, set_name, "ensemble" if name.startswith("ens") else "model", probas[name])
        scores[name] = log_loss_of(y_va[half:], probas[name][half: len(va)])
        print(f"  [{tag}] {name}: val log-loss {ll:.4f}; va_select (вибір) {scores[name]:.4f}")
    best_name = min(scores, key=scores.get)

    p_best = probas[best_name]
    cal, cal_scores = choose_calibrator(p_best[: len(va)], y_va)
    for suffix, cls in (("iso", IsotonicCalibrator), ("platt", LogitCalibrator)):
        table.add(f"{best_name}_{suffix}", set_name, "calibration", cls().fit(p_best[: len(va)], y_va).transform(p_best))

    t1 = time.time()
    trva = pd.concat([tr, va], ignore_index=True)
    final = FinalModel(refit(candidates[best_name], trva), cal, name=f"{best_name} (train+val)", track=track)
    final_col = f"final_{set_name}"
    table.add(final_col, set_name, "final", np.vstack([cal.transform(p_best[: len(va)]), final.predict_proba(te)]))
    print(f"  [{tag}] донавчання {best_name} на train+val: {time.time() - t1:.0f} с")

    final.meta = {
        "selected_by": "log-loss на va_select (2-га половина val); кандидати навчені лише на train",
        "selected_candidate": best_name,
        "candidates_va_select_log_loss": scores,
        "lgbm_variants_va_tune_log_loss": variants,
        "lgbm_half_life": lgbm.half_life,
        "lgbm_init_score": lgbm.anchor is not None,
        "lgbm_n_features": len(lgbm.features),
        "ensemble_weights": weights,
        "calibrator": cal.name,
        "calibration_scores_val_half2": cal_scores,
        "val_metrics_train_only": compute_metrics(y_va, p_best[: len(va)]),
        "lgbm_params": {k: lgbm.params[k] for k in (*PARAM_SPACE, "learning_rate")},
        "lgbm_best_iteration": int(lgbm.booster.best_iteration),
        "train_period": [str(trva["date"].min().date()), str(trva["date"].max().date())],
        "n_train": int(len(trva)),
        "preds_column": final_col,
    }
    joblib.dump(final, Path(models_dir) / f"{file_name}.joblib")
    lgbm_file = f"lgbm_{track}_{set_name}.joblib"
    joblib.dump(lgbm, Path(models_dir) / lgbm_file)  # для SHAP в evaluate
    print(f"[{tag}] ФІНАЛЬНА: {final.name}, калібратор '{cal.name}' → models/{file_name}.joblib")
    info = {"file": f"{file_name}.joblib", "final_model": final_col, "selected_candidate": best_name,
            "calibrator": cal.name, "features": feats, "lgbm_file": lgbm_file,
            "lgbm_variants": variants, "lgbm_half_life": lgbm.half_life, "lgbm_init_score": lgbm.anchor is not None,
            "lgbm_n_features": len(lgbm.features), "ensemble_weights": weights,
            "candidates_va_select_log_loss": scores}
    return info, lgbm.feature_importance().assign(feature_set=set_name)


def train_track(track: str, df: pd.DataFrame, models_dir: Path, reports_dir: Path, data_dir: Path) -> dict:
    """Повний цикл навчання для одного треку; зберігає моделі, прогнози, підсумок. Повертає підсумок."""
    t_track = time.time()
    parts = split_by_time(df)
    tr = parts["train"]
    print(f"\n[{track}] train={len(tr)}, val={len(parts['val'])}, test={len(parts['test'])}")
    sets = feature_sets(df)
    for s, cols in sets.items():
        print(f"[{track}] набір '{s}': {len(cols)} ознак")

    table = PredsTable(parts["val"], parts["test"])
    eval_df = pd.concat([parts["val"], parts["test"]], ignore_index=True)
    baselines = baseline_models(df, sets)
    for name, model in baselines.items():
        model.fit(tr, tr["y"].to_numpy())
        print(f"[{track}] {name}: val log-loss {table.add(name, 'baseline', 'baseline', model.predict_proba(eval_df)):.4f}")
    elo_anchor = LogRegModel([c for c in ANCHOR_ELO_FEATURES if c in df.columns]).fit(tr, tr["y"].to_numpy())

    # Гіперпараметри шукаємо один раз (на hist_odds, інакше no_odds) і перевикористовуємо для всіх наборів.
    search_set = next((s for s in SEARCH_SET_PRIORITY if s in sets), next(iter(sets)))
    va_tune = parts["val"].iloc[: len(parts["val"]) // 2]
    params, trials = search_params(tr, va_tune, sets[search_set], f"{track}/{search_set}")

    summary_sets, importances = {}, []
    for set_name, feats in sets.items():
        info, imp = train_feature_set(track, set_name, feats, parts, table, elo_anchor,
                                      baselines.get("baseline_odds"), final_file_name(track, set_name, sets),
                                      models_dir, params)
        summary_sets[set_name] = info
        importances.append(imp)

    table.frame.to_parquet(Path(data_dir) / f"preds_{track}.parquet", index=False)
    pd.concat(importances, ignore_index=True).to_csv(Path(reports_dir) / f"feature_importance_{track}.csv", index=False)
    summary = {"track": track, "models": table.registry, "sets": summary_sets,
               "search": {"feature_set": search_set, "trials": trials},
               "seconds": round(time.time() - t_track, 1)}
    (Path(models_dir) / f"train_summary_{track}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[{track}] готово за {summary['seconds']:.0f} с")
    return summary


def main(data_dir: Path = config.PROCESSED, models_dir: Path = config.MODELS,
         reports_dir: Path = config.REPORTS) -> None:
    """Навчає всі треки, для яких є таблиці ознак."""
    t0 = time.time()
    for d in (models_dir, reports_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    for track in config.TRACKS:
        df = load_track(track, data_dir)
        if df is not None:
            train_track(track, df, models_dir, reports_dir, data_dir)
    print(f"\nНавчання завершено за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
