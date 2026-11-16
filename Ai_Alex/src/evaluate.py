"""Оцінка всіх моделей на TEST (і val для довідки), графіки, тест на витік.

Читає data/processed/preds_<track>.parquet і models/train_summary_<track>.json (пише src.train).
Пише reports/metrics.csv, metrics.md, metrics_by_league.csv, metrics_by_season.csv, reports/figures/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from src import config  # noqa: E402
from src.modeling import LGBMModel, compute_metrics  # noqa: E402
from src.train import proba_col  # noqa: E402

METRIC_COLS = ["accuracy", "log_loss", "brier", "rps", "macro_f1"]
CLASS_UA = {"H": "Перемога господарів", "D": "Нічия", "A": "Перемога гостей"}
TRACK_UA = {"clubs": "Клуби", "intl": "Збірні"}
SHAP_SAMPLE = 5000


class LeakageError(RuntimeError):
    """Test accuracy вище config.LEAK_ACC_LIMIT — ймовірний витік даних."""


# ----------------------------------------------------------------------------- завантаження
def load_results(track: str, data_dir: Path, models_dir: Path) -> tuple[pd.DataFrame, dict] | None:
    """Прогнози і підсумок навчання треку; None, якщо трек не навчався."""
    preds_path = Path(data_dir) / f"preds_{track}.parquet"
    summary_path = Path(models_dir) / f"train_summary_{track}.json"
    if not (preds_path.exists() and summary_path.exists()):
        print(f"[{track}] немає результатів навчання — пропущено")
        return None
    return pd.read_parquet(preds_path), json.loads(summary_path.read_text(encoding="utf-8"))


def proba_of(preds: pd.DataFrame, model: str) -> np.ndarray:
    """Матриця ймовірностей (n, 3) моделі з таблиці прогнозів."""
    return preds[[proba_col(model, c) for c in config.CLASSES]].to_numpy(dtype=float)


def final_models(summary: dict) -> dict[str, str]:
    """{набір ознак: назва фінальної моделі в preds}."""
    return {s: info["final_model"] for s, info in summary["sets"].items()}


def _by_values(series: pd.Series) -> list[tuple[str, np.ndarray]]:
    """Сегменти «кожне значення серії окремо»."""
    return [(str(v), (series == v).to_numpy()) for v in sorted(series.dropna().unique())]


def segments(track: str, preds: pd.DataFrame) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Розрізи для окремих метрик: {розріз: [(назва сегмента, булева маска рядків preds)]}.

    league — клуби: топ-5 ліг окремо й разом, решта ліг; збірні: тип турніру.
    season — клуби: сезон; збірні: рік. clubelo — клуби: матчі з рейтингом ClubElo / без.
    """
    out = {}
    if track == "clubs":
        top5 = preds["competition"].map(config.TOP5)
        out["league"] = _by_values(top5) + [("Топ-5 разом", top5.notna().to_numpy()),
                                            ("Інші ліги", top5.isna().to_numpy())]
        if "season" in preds:
            out["season"] = _by_values(preds["season"].astype("Int64").astype(str))
        if "has_clubelo" in preds:
            flag = preds["has_clubelo"].astype(bool)
            out["clubelo"] = [("з ClubElo", flag.to_numpy()), ("без ClubElo", (~flag).to_numpy())]
    else:
        if "f_cat_tournament_type" in preds:
            out["league"] = _by_values(preds["f_cat_tournament_type"])
        out["season"] = _by_values(pd.to_datetime(preds["date"]).dt.year.astype(str))
    return out


# ----------------------------------------------------------------------------- метрики
def metrics_table(track: str, preds: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Метрики кожної моделі на val і test."""
    finals = set(final_models(summary).values())
    rows = []
    for m in summary["models"]:
        for split in ("val", "test"):
            part = preds[preds["split"] == split]
            rows.append({"track": track, "feature_set": m["feature_set"], "model": m["model"],
                         "kind": m["kind"], "split": split, "final": m["model"] in finals,
                         **compute_metrics(part["y"].to_numpy(), proba_of(part, m["model"]))})
    return pd.DataFrame(rows)


def grouped_metrics(track: str, preds: pd.DataFrame, summary: dict, by: str) -> pd.DataFrame:
    """Метрики на test у розрізі (by: 'league' / 'season' / 'clubelo') для фінальних і базових моделей."""
    segs = segments(track, preds).get(by)
    if not segs:
        return pd.DataFrame()
    test_mask = (preds["split"] == "test").to_numpy()
    models = [m["model"] for m in summary["models"] if m["kind"] == "baseline"] + \
        list(final_models(summary).values())
    rows = []
    for value, seg_mask in segs:
        mask = test_mask & seg_mask
        if mask.sum() < 30:
            continue
        part = preds[mask]
        for model in models:
            rows.append({"track": track, by: value, "model": model,
                         **compute_metrics(part["y"].to_numpy(), proba_of(part, model))})
    return pd.DataFrame(rows)


def check_leakage(metrics: pd.DataFrame) -> list[str]:
    """Тест на витік (test accuracy); повертає порушників.

    Клуби: accuracy > config.LEAK_ACC_LIMIT. Збірні: baseline_elo — лише абсолютний поріг
    config.INTL_LEAK_ABS_LIMIT; інші моделі — абсолютний поріг АБО приріст над baseline_elo
    > config.INTL_LEAK_MAX_GAIN_OVER_ELO (правило «вищий Elo» саме дає ~60% через нерівні пари).
    """
    test = metrics[metrics["split"] == "test"]
    elo = test[(test["track"] == "intl") & (test["model"] == "baseline_elo")]["accuracy"]
    elo_acc = elo.iloc[0] if len(elo) else None
    bad = []
    for r in test.itertuples():
        if r.track != "intl":
            if r.accuracy > config.LEAK_ACC_LIMIT:
                bad.append(f"{r.track}/{r.model} (accuracy={r.accuracy:.3f} > {config.LEAK_ACC_LIMIT})")
            continue
        gain = r.accuracy - elo_acc if elo_acc is not None and r.model != "baseline_elo" else 0.0
        if r.accuracy > config.INTL_LEAK_ABS_LIMIT or gain > config.INTL_LEAK_MAX_GAIN_OVER_ELO:
            bad.append(f"intl/{r.model} (accuracy={r.accuracy:.3f}, приріст над baseline_elo={gain:+.3f})")
    return bad


def leakage_text(metrics: pd.DataFrame, leaks: list[str]) -> str:
    """Текст висновку про тест на витік для metrics.md і консолі."""
    rule = (f"Правило: клуби — test accuracy ≤ {config.LEAK_ACC_LIMIT:.0%}; збірні — не більше ніж "
            f"на {config.INTL_LEAK_MAX_GAIN_OVER_ELO * 100:.0f} п.п. краще за baseline_elo і ≤ "
            f"{config.INTL_LEAK_ABS_LIMIT:.0%} (сам baseline_elo — лише ≤ {config.INTL_LEAK_ABS_LIMIT:.0%}). "
            "Для збірних абсолютний поріг 60% не працює: майже половина матчів test — явно нерівні пари "
            "(|різниця Elo| > 200, напр. Німеччина — Сан-Марино), і просте правило «перемагає вищий Elo» "
            "саме дає ~60% accuracy. Справжній витік дав би стрибок на 10–20 п.п. над Elo, чесні ознаки — 1–2 п.п. "
            "Elo збірних перевірено як передматчевий: кореляція результату зі зміною Elo до матчу ≈ 0.03, "
            "після матчу ≈ 0.77.")
    test = metrics[metrics["split"] == "test"]
    stats = "; ".join(f"{TRACK_UA.get(t, t)}: max accuracy {g['accuracy'].max():.3f}"
                      for t, g in test.groupby("track"))
    intl = test[test["track"] == "intl"]
    elo = intl[intl["model"] == "baseline_elo"]
    if len(elo):
        gain = intl[intl["model"] != "baseline_elo"]["accuracy"].max() - elo["accuracy"].iloc[0]
        stats += f" (baseline_elo {elo['accuracy'].iloc[0]:.3f}, макс. приріст {gain * 100:+.1f} п.п.)"
    verdict = f"ПРОВАЛЕНО: {', '.join(leaks)}." if leaks else "Пройдено."
    return f"{verdict} {stats}. {rule}"


def check_beats_baseline(metrics: pd.DataFrame) -> list[str]:
    """Перевіряє, що кожна фінальна модель краща за «завжди господарі» і за accuracy, і за log-loss."""
    lines = []
    test = metrics[metrics["split"] == "test"]
    for track, part in test.groupby("track"):
        base = part[part["model"] == "baseline_home"].iloc[0]
        for r in part[part["final"]].itertuples():
            ok = r.accuracy > base["accuracy"] and r.log_loss < base["log_loss"]
            lines.append(f"{'OK ' if ok else 'НІ '} {track}/{r.model}: accuracy {r.accuracy:.3f} vs "
                         f"{base['accuracy']:.3f}, log-loss {r.log_loss:.4f} vs {base['log_loss']:.4f}")
    return lines


# ----------------------------------------------------------------------------- графіки
def plot_confusion(track: str, preds: pd.DataFrame, summary: dict, fig_dir: Path) -> None:
    """Матриці помилок (нормовані за рядками) фінальних моделей треку на test."""
    finals = final_models(summary)
    test = preds[preds["split"] == "test"]
    fig, axes = plt.subplots(1, len(finals), figsize=(5.2 * len(finals), 4.6), squeeze=False)
    for ax, (fset, model) in zip(axes[0], finals.items()):
        pred = proba_of(test, model).argmax(axis=1)
        cm = confusion_matrix(test["y"], pred, labels=[0, 1, 2])
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), config.CLASSES)
        ax.set_yticks(range(3), config.CLASSES)
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Факт")
        ax.set_title(f"{TRACK_UA[track]} / {fset}\n{model}", fontsize=10)
    fig.suptitle(f"Матриця помилок на test — {TRACK_UA[track]}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"confusion_{track}.png", dpi=120)
    plt.close(fig)


def plot_reliability(track: str, preds: pd.DataFrame, summary: dict, fig_dir: Path, n_bins: int = 10) -> None:
    """Калібрувальні криві по класах H/D/A для фінальних моделей на test (квантильні біни)."""
    test = preds[preds["split"] == "test"]
    y = test["y"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for fset, model in final_models(summary).items():
        proba = proba_of(test, model)
        for k, ax in enumerate(axes):
            bins = pd.qcut(proba[:, k], n_bins, labels=False, duplicates="drop")
            frame = pd.DataFrame({"p": proba[:, k], "o": (y == k).astype(float), "b": bins})
            agg = frame.groupby("b").mean()
            ax.plot(agg["p"], agg["o"], marker="o", label=f"{fset}: {model}")
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ідеальне калібрування")
        ax.set_title(f"{CLASS_UA[config.CLASSES[k]]} ({config.CLASSES[k]})")
        ax.set_xlabel("Прогнозована ймовірність")
        ax.set_ylabel("Фактична частота")
        ax.legend(fontsize=8)
    fig.suptitle(f"Калібрувальні криві на test — {TRACK_UA[track]}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"reliability_{track}.png", dpi=120)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, fig_dir: Path) -> None:
    """Горизонтальні стовпчики test log-loss усіх моделей по треках."""
    test = metrics[metrics["split"] == "test"]
    tracks = list(test["track"].unique())
    fig, axes = plt.subplots(1, len(tracks), figsize=(7 * len(tracks), 6), squeeze=False)
    for ax, track in zip(axes[0], tracks):
        part = test[test["track"] == track].sort_values("log_loss", ascending=False)
        colors = ["tab:orange" if f else ("tab:gray" if k == "baseline" else "tab:blue")
                  for f, k in zip(part["final"], part["kind"])]
        ax.barh(part["model"], part["log_loss"], color=colors)
        lo = part["log_loss"].min()
        ax.set_xlim(lo - 0.05, part["log_loss"].max() + 0.02)
        ax.set_xlabel("Log-loss на test (менше — краще)")
        ax.set_title(f"{TRACK_UA[track]}: помаранчевий — фінальні, сірий — базові")
    fig.suptitle("Порівняння моделей за log-loss")
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_logloss.png", dpi=120)
    plt.close(fig)


def plot_importance(track: str, reports_dir: Path, fig_dir: Path, top: int = 20) -> None:
    """Топ ознак LightGBM за gain для кожного набору ознак треку."""
    path = Path(reports_dir) / f"feature_importance_{track}.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path)
    sets = list(imp["feature_set"].unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(7 * len(sets), 7), squeeze=False)
    for ax, fset in zip(axes[0], sets):
        part = imp[imp["feature_set"] == fset].nlargest(top, "gain").iloc[::-1]
        ax.barh(part["feature"], part["gain"], color="tab:green")
        ax.set_xlabel("Сумарний gain")
        ax.set_title(f"{TRACK_UA[track]} / {fset}: топ-{top} ознак LightGBM")
    fig.tight_layout()
    fig.savefig(fig_dir / f"feature_importance_{track}.png", dpi=120)
    plt.close(fig)


def plot_shap(track: str, summary: dict, data_dir: Path, models_dir: Path, fig_dir: Path) -> None:
    """SHAP (вбудований TreeSHAP LightGBM, pred_contrib) — середній |SHAP| по класах на вибірці test."""
    feats_path = Path(data_dir) / f"{track}_features.parquet"
    if not feats_path.exists():
        return
    df = pd.read_parquet(feats_path)
    test = df[pd.to_datetime(df["date"]) >= pd.Timestamp(config.TEST_START)]
    test = test.sample(min(SHAP_SAMPLE, len(test)), random_state=config.RANDOM_STATE)
    for fset, info in summary["sets"].items():
        model_path = Path(models_dir) / info.get("lgbm_file", "")
        if not info.get("lgbm_file") or not model_path.exists():
            continue
        model: LGBMModel = joblib.load(model_path)
        X = LGBMModel.prepare(test, model.features, model.cat_levels)
        contrib = model.booster.predict(X, pred_contrib=True, num_iteration=model.booster.best_iteration or None)
        n_f = len(model.features)
        per_class = np.stack([np.abs(contrib[:, k * (n_f + 1): k * (n_f + 1) + n_f]).mean(axis=0)
                              for k in range(3)], axis=1)
        order = np.argsort(per_class.sum(axis=1))[-20:]
        fig, ax = plt.subplots(figsize=(8, 7))
        left = np.zeros(len(order))
        for k, cls in enumerate(config.CLASSES):
            ax.barh(np.array(model.features)[order], per_class[order, k], left=left, label=CLASS_UA[cls])
            left += per_class[order, k]
        ax.set_xlabel("Середній |SHAP| (лог-шанси)")
        ax.set_title(f"SHAP: вплив ознак LightGBM — {TRACK_UA[track]} / {fset}\n(вибірка test, n={len(X)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"shap_{track}_{fset}.png", dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------------- звіт
def to_markdown(df: pd.DataFrame) -> str:
    """Проста markdown-таблиця (без залежності від tabulate)."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def split_table(metrics: pd.DataFrame, track: str, split: str) -> str:
    """Таблиця метрик треку на одному спліті, відсортована за log-loss."""
    part = metrics[(metrics["track"] == track) & (metrics["split"] == split)].sort_values("log_loss")
    part = part.assign(model=np.where(part["final"], "**" + part["model"] + "** ★", part["model"]))
    return to_markdown(part[["feature_set", "model", "n", *METRIC_COLS]])


SET_UA = {"no_odds": "без коефіцієнтів (суворий)", "hist_odds": "з історією коефіцієнтів",
          "with_odds": "з коефіцієнтами матчу"}

# Метрики фінальних моделей v2 на test (accuracy, log-loss) — для таблиці «v2 → v3».
V2_TEST = {("clubs", "no_odds"): (0.4910, 1.0154), ("clubs", "with_odds"): (0.5026, 1.0023),
           ("intl", "no_odds"): (0.6153, 0.8524)}


def version_table(metrics: pd.DataFrame) -> str:
    """Таблиця «v2 → v3» для фінальних моделей на test."""
    test = metrics[(metrics["split"] == "test") & metrics["final"]]
    rows = []
    for r in test.itertuples():
        v2 = V2_TEST.get((r.track, r.feature_set))
        rows.append({"трек": TRACK_UA[r.track], "набір": r.feature_set,
                     "v2 accuracy": f"{v2[0]:.4f}" if v2 else "—", "v3 accuracy": r.accuracy,
                     "v2 log-loss": f"{v2[1]:.4f}" if v2 else "—", "v3 log-loss": r.log_loss,
                     "Δ log-loss": f"{r.log_loss - v2[1]:+.4f}" if v2 else "—"})
    return to_markdown(pd.DataFrame(rows))


def tuning_text(results: dict) -> str:
    """Що обрали на train/val: ваги свіжості, init_score, відбір ознак, ваги ансамблю, переможець."""
    lines = []
    for track, (_, summary) in results.items():
        for fset, info in summary["sets"].items():
            if "lgbm_variants" not in info:
                continue
            variants = "; ".join(f"{k} = {v:.4f}" for k, v in info["lgbm_variants"].items())
            lines.append(f"- {TRACK_UA[track]} / {fset}: варіанти LightGBM (log-loss на 1-й половині val): {variants}. "
                         f"Обрано: півперіод {info['lgbm_half_life']}, init_score = {info['lgbm_init_score']}, "
                         f"{info['lgbm_n_features']} з {len(info['features'])} ознак; ваги ансамблю "
                         f"{info['ensemble_weights']}; фінальний кандидат — {info['selected_candidate']}.")
    return "\n".join(lines)


def odds_comparison(metrics: pd.DataFrame) -> str:
    """Текстовий висновок: порівняння наборів ознак клубів (без коеф. / історія коеф. / коеф. матчу)."""
    test = metrics[(metrics["split"] == "test") & (metrics["track"] == "clubs")]
    if test.empty:
        return "Трек клубів не навчався."
    get = lambda cond: test[cond].iloc[0] if cond.any() else None  # noqa: E731
    bk = get(test["model"] == "baseline_odds")
    parts = []
    for fset, label in SET_UA.items():
        r = get(test["final"] & (test["feature_set"] == fset))
        if r is not None:
            parts.append(f"Фінальна модель {label}: log-loss {r.log_loss:.4f}, accuracy {r.accuracy:.3f}.")
    parts.append("Набір hist_odds чесно придатний для прогнозу ДО появи коефіцієнтів: коефіцієнтів цього матчу в ньому "
                 "немає, є лише ринкова оцінка команд у їхніх попередніх матчах (f_hodds_*).")
    if bk is not None:
        parts.append(f"Ринок (коефіцієнти БК напряму): log-loss {bk.log_loss:.4f}, accuracy {bk.accuracy:.3f}.")
    parts.append("Коефіцієнти концентрують інформацію, якої немає в публічній статистиці (склади, травми, мотивація), "
                 "тому модель без них очікувано трохи гірша; модель з ними здебільшого відтворює ринок.")
    return " ".join(parts)


def draw_share(metrics_preds: dict[str, tuple[pd.DataFrame, dict]]) -> str:
    """Частка прогнозів «нічия» фінальних моделей vs фактична частка нічиїх на test."""
    out = []
    for track, (preds, summary) in metrics_preds.items():
        test = preds[preds["split"] == "test"]
        for fset, model in final_models(summary).items():
            p = proba_of(test, model)
            out.append(f"{TRACK_UA[track]}/{fset}: модель прогнозує нічию в {np.mean(p.argmax(1) == 1):.1%} матчів "
                       f"(фактично нічиїх {np.mean(test['y'] == 1):.1%}; макс. P(нічия) = {p[:, 1].max():.2f})")
    return "; ".join(out) + "."


def write_markdown(metrics: pd.DataFrame, by_league: pd.DataFrame, by_season: pd.DataFrame,
                   by_clubelo: pd.DataFrame, results: dict, leaks: list[str], beats: list[str], reports_dir: Path) -> None:
    """Збирає reports/metrics.md українською."""
    md = ["# Метрики моделей прогнозу H / D / A", "",
          f"Розбиття за часом: train < {config.VAL_START} ≤ val < {config.TEST_START} ≤ test.", "",
          "**Набори ознак клубів:** `no_odds` — суворий, без жодної ринкової інформації (без f_odds_ і f_hodds_); "
          "`hist_odds` — без коефіцієнтів ЦЬОГО матчу, але з історією ринкової оцінки команд у попередніх матчах "
          "(f_hodds_); `with_odds` — усе, включно з коефіцієнтами матчу.", "",
          "**Схема.** Кандидати (логрегресія `logreg_*`, LightGBM `lgbm_*`, ансамбль `ens_*` — зважене середнє "
          "LightGBM + логрегресії, для with_odds ще й ринку; ваги — сітка з кроком 0.1) навчаються лише на train. "
          "LightGBM: випадковий пошук (20 спроб) на найсвіжіших 30% train з lr 0.1 — один раз на трек (на наборі "
          "hist_odds), конфігурація перевикористовується для всіх наборів; далі на всьому train порівнюються звичайне "
          "навчання і старт від сильного прогнозу (init_score = log ймовірностей ринку для with_odds або "
          "Elo-логрегресії). Ваги свіжості (півперіод 5/8 років) і відкидання ознак з малим gain пробували — на val "
          "приросту не дали, тому вимкнені. Логрегресія отримує лог-ймовірності стартового прогнозу як ознаки. "
          "Серед калібраторів є температура, що стискає крайні ймовірності. Validation поділено за часом навпіл: 1-ша половина — early "
          "stopping LightGBM і калібратор, 2-га — вибір моделі (щоб LightGBM не мав переваги від підгонки під той "
          "самий val). Фінальна модель (★ `final_*`) — кандидат із найменшим log-loss на 2-й половині val, ДОНАВЧЕНИЙ на train+val (LightGBM: best_iter × 1.1 дерев). "
          "У рядку `final_*` val-метрики — від версії, навченої лише на train (чесна val-оцінка), test-метрики — "
          "від донавченої моделі, яку зберігає `models/*.joblib`. Рядки `*_iso` / `*_platt` — обраний кандидат з "
          "ізотонічним / Platt-калібруванням, навченим на val (їхні val-метрики оптимістичні; дивитися на test). "
          "Калібратор для фінальної моделі обирається так само: навчання на 1-й половині val, оцінка на 2-й. "
          "Val-метрики все одно трохи оптимістичні, тож чесне порівняння — лише test.", ""]
    for track in metrics["track"].unique():
        md += [f"## {TRACK_UA[track]} — TEST", "", split_table(metrics, track, "test"), "",
               f"<details><summary>{TRACK_UA[track]} — validation (довідково)</summary>", "",
               split_table(metrics, track, "val"), "", "</details>", ""]
    md += ["## Висновки", "",
           "**Набори ознак клубів.** " + odds_comparison(metrics), "",
           "**v2 → v3 (test, фінальні моделі).**", "", version_table(metrics), "",
           "**Що обрано на train/val (test не використовувався).**", "", tuning_text(results), "",
           "**Чому нічиї майже не прогнозуються.** Нічия майже ніколи не є найімовірнішим результатом: навіть для "
           "рівних суперників P(нічия) зазвичай ≈ 0.27–0.32, а P(H) або P(A) вища. Argmax тому обирає H/A, і це "
           "оптимально для accuracy; ймовірність нічиєї при цьому відкалібрована, що видно на log-loss/RPS. "
           + draw_share(results), "",
           "**Тест на витік.** " + leakage_text(metrics, leaks), "",
           "**Фінальні моделі vs «завжди господарі» (test):**", "", *[f"- {b}" for b in beats], ""]
    for title, table, col in (("Окремі розрізи: ліги / типи турнірів (test)", by_league, "league"),
                              ("Окремі розрізи: сезони / роки (test)", by_season, "season"),
                              ("Окремі розрізи: клуби з рейтингом ClubElo / без (test)", by_clubelo, "clubelo")):
        md += [f"## {title}", ""]
        if not table.empty:
            for track in table["track"].unique():
                part = table[table["track"] == track][[col, "model", "n", *METRIC_COLS]]
                md += [f"### {TRACK_UA[track]}", "", to_markdown(part), ""]
    md += ["## Графіки", "", "- `figures/model_comparison_logloss.png` — порівняння моделей за log-loss",
           "- `figures/confusion_<трек>.png` — матриці помилок фінальних моделей",
           "- `figures/reliability_<трек>.png` — калібрувальні криві",
           "- `figures/feature_importance_<трек>.png`, `figures/shap_<трек>_<набір>.png` — важливість ознак", ""]
    (Path(reports_dir) / "metrics.md").write_text("\n".join(md), encoding="utf-8")


def main(data_dir: Path = config.PROCESSED, models_dir: Path = config.MODELS,
         reports_dir: Path = config.REPORTS) -> None:
    """Рахує метрики, будує графіки і звіти; падає з LeakageError, якщо test accuracy підозріло висока."""
    reports_dir = Path(reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {t: r for t in config.TRACKS if (r := load_results(t, data_dir, models_dir)) is not None}
    if not results:
        raise FileNotFoundError("Немає жодного треку з результатами навчання — спершу запустіть src.train")

    metrics = pd.concat([metrics_table(t, p, s) for t, (p, s) in results.items()], ignore_index=True)
    by_league = pd.concat([grouped_metrics(t, p, s, "league") for t, (p, s) in results.items()], ignore_index=True)
    by_season = pd.concat([grouped_metrics(t, p, s, "season") for t, (p, s) in results.items()], ignore_index=True)
    by_clubelo = pd.concat([grouped_metrics(t, p, s, "clubelo") for t, (p, s) in results.items()], ignore_index=True)
    by_clubelo.to_csv(reports_dir / "metrics_by_clubelo.csv", index=False)
    metrics.to_csv(reports_dir / "metrics.csv", index=False)
    by_league.to_csv(reports_dir / "metrics_by_league.csv", index=False)
    by_season.to_csv(reports_dir / "metrics_by_season.csv", index=False)

    for track, (preds, summary) in results.items():
        plot_confusion(track, preds, summary, fig_dir)
        plot_reliability(track, preds, summary, fig_dir)
        plot_importance(track, reports_dir, fig_dir)
        try:
            plot_shap(track, summary, data_dir, models_dir, fig_dir)
        except Exception as exc:  # SHAP — необов'язковий графік, не валимо весь звіт
            print(f"[{track}] SHAP пропущено: {exc}")
    plot_model_comparison(metrics, fig_dir)

    leaks, beats = check_leakage(metrics), check_beats_baseline(metrics)
    write_markdown(metrics, by_league, by_season, by_clubelo, results, leaks, beats, reports_dir)

    test = metrics[metrics["split"] == "test"]
    print(test[["track", "feature_set", "model", "final", *METRIC_COLS]].to_string(index=False))
    print("\nФінальні моделі vs «завжди господарі»:\n" + "\n".join(beats))
    if leaks:
        raise LeakageError(f"ТЕСТ НА ВИТІК ПРОВАЛЕНО: {', '.join(leaks)}")
    print("\n" + leakage_text(metrics, leaks))


if __name__ == "__main__":
    main()
