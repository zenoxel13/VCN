"""
Football Match Winner Prediction: графіки точності моделей.

Вхід:
    - За замовчуванням читає збережену модель з out/predictor.pkl та out/features.parquet (або .csv),
      навчає варіанти на 5 і 3 останніх сезонах до test-періоду й оцінює на test-матчах.
    - Або приймає CSV з колонками: date, correct_5y, correct_3y

Запуск:
    python prediction_dashboard.py                 # реальні дані з папки out/
    python prediction_dashboard.py out             # вказати іншу папку з моделлю
    python prediction_dashboard.py results.csv     # готовий CSV
    python prediction_dashboard.py --fake          # фейкові дані
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

C5, C3 = "#2E86AB", "#F18F01"   # кольори моделей: 5 років / 3 роки
ROLL = 200                      # вікно ковзної точності (у матчах)


# ---------- дані ----------
def make_fake_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-08-01", periods=n, freq="D")
    base = rng.random(n)
    return pd.DataFrame({
        "date": dates,
        "correct_5y": base < 0.54,
        "correct_3y": np.where(rng.random(n) < 0.6, base < 0.54, rng.random(n) < 0.49),
    })


def load_from_model(out_dir="out"):
    model_path = os.path.join(out_dir, "predictor.pkl")
    pq_path = os.path.join(out_dir, "features.parquet")
    csv_path = os.path.join(out_dir, "features.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Не знайдено збережену модель: {model_path}")

    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)

    if os.path.exists(pq_path):
        feats = pd.read_parquet(pq_path)
    elif os.path.exists(csv_path):
        feats = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        raise FileNotFoundError(f"Не знайдено features.parquet або features.csv у {out_dir}")

    feats["date"] = pd.to_datetime(feats["date"])
    cols = bundle["cols"]
    base_model = bundle["model"]

    pre_test = feats[feats["split"].isin(["train", "val"])]
    test = feats[feats["split"] == "test"].sort_values(["date", "game_id"]).reset_index(drop=True)
    if test.empty:
        raise ValueError("У features відсутні матчі зі split == 'test'.")

    years = sorted(pre_test["f_year"].unique())
    yrs_5 = years[-5:] if len(years) >= 5 else years
    yrs_3 = years[-3:] if len(years) >= 3 else years

    tr_5 = pre_test[pre_test["f_year"].isin(yrs_5)]
    tr_3 = pre_test[pre_test["f_year"].isin(yrs_3)]

    print(f"Навчання 5y ({min(yrs_5)}–{max(yrs_5)}, {len(tr_5)} матчів) і 3y ({min(yrs_3)}–{max(yrs_3)}, {len(tr_3)} матчів)...")
    m5 = clone(base_model).fit(tr_5[cols].astype(float), tr_5["y"].values)
    m3 = clone(base_model).fit(tr_3[cols].astype(float), tr_3["y"].values)

    X_test = test[cols].astype(float)
    y_test = test["y"].values

    p5 = m5.predict_proba(X_test).argmax(axis=1)
    p3 = m3.predict_proba(X_test).argmax(axis=1)

    df = pd.DataFrame({
        "date": test["date"],
        "correct_5y": p5 == y_test,
        "correct_3y": p3 == y_test,
    })
    print(f"Точність на test ({len(df)} матчів): 5y = {df['correct_5y'].mean():.2%}, 3y = {df['correct_3y'].mean():.2%}")
    return df


def load(arg=None):
    if arg == "--fake":
        return make_fake_data()
    if arg is not None and arg.endswith(".csv"):
        df = pd.read_csv(arg, parse_dates=["date"])
        for c in ("correct_5y", "correct_3y"):
            df[c] = df[c].astype(bool)
        return df.sort_values("date").reset_index(drop=True)
    out_dir = arg if arg is not None else "out"
    return load_from_model(out_dir)


# ---------- допоміжне ----------
def wilson_ci(k, n, z=1.96):
    """95% довірчий інтервал для частки (Вілсон)."""
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def style(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")

ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


# ---------- графіки ----------
def plot_accuracy(ax, df):
    names, vals, errs = ["5 років", "3 роки"], [], []
    for col in ("correct_5y", "correct_3y"):
        k, n = int(df[col].sum()), len(df)
        lo, hi = wilson_ci(k, n)
        vals.append(k / n)
        errs.append([k / n - lo, hi - k / n])
    bars = ax.bar(names, vals, color=[C5, C3], width=0.5,
                  yerr=np.array(errs).T, capsize=6, ecolor="#444")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{v:.1%}",
                ha="center", color="white", fontsize=15, fontweight="bold")
    ax.axhline(1 / 3, ls="--", c="gray", lw=1, label="Випадкове вгадування (1/3)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    style(ax, "Загальна точність (з 95% ДІ)")


def plot_rolling(ax, df):
    win = min(ROLL, max(20, len(df) // 10))
    tmp = df.copy()
    for col, color, label in (("correct_5y", C5, "5 років"), ("correct_3y", C3, "3 роки")):
        tmp["roll"] = tmp[col].rolling(win, min_periods=win // 2).mean()
        # Беремо останнє значення на дату, щоб уникнути вертикальних стрибків у дні з багатьма матчами
        daily = tmp.groupby("date")["roll"].last().dropna()
        ax.plot(daily.index, daily.values, color=color, lw=2, label=label)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(frameon=False)
    style(ax, f"Ковзна точність (вікно {win} матчів)")


# ---------- збірка ----------
def build_dashboard(df, out="dashboard.png"):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_accuracy(axs[0], df)
    plot_rolling(axs[1], df)
    fig.suptitle(f"Прогноз переможця футбольного матчу: 5 років vs 3 роки  (матчів: {len(df)})",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    print(f"Збережено: {out}")
    return fig


if name == "__main__":
    data = load(sys.argv[1] if len(sys.argv) > 1 else None)
    build_dashboard(data)
    plt.show()
