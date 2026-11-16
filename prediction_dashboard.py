"""
Football Match Winner Prediction: графіки точності моделей.

Вхід: таблиця, де один рядок = один матч, колонки:
    date        - дата матчу
    correct_5y  - True/False: модель (навчена на 5 роках) вгадала результат
    correct_3y  - True/False: модель (навчена на 3 роках) вгадала результат

Запуск:
    python prediction_dashboard.py                 # фейкові дані
    python prediction_dashboard.py results.csv     # реальні дані
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C5, C3 = "#2E86AB", "#F18F01"   # кольори моделей: 5 років / 3 роки
ROLL = 50                       # вікно ковзної точності (у матчах)


# ---------- дані ----------
def make_fake_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-08-01", periods=n, freq="D")
    base = rng.random(n)
    return pd.DataFrame({
        "date": dates,
        "correct_5y": base < 0.54,
        # 3y модель частково корелює з 5y, але трохи гірша
        "correct_3y": np.where(rng.random(n) < 0.6, base < 0.54, rng.random(n) < 0.49),
    })


def load(path=None):
    if path is None:
        return make_fake_data()
    df = pd.read_csv(path, parse_dates=["date"])
    for c in ("correct_5y", "correct_3y"):
        df[c] = df[c].astype(bool)
    return df.sort_values("date").reset_index(drop=True)


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
        k, n = df[col].sum(), len(df)
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
    for col, color, label in (("correct_5y", C5, "5 років"), ("correct_3y", C3, "3 роки")):
        ax.plot(df["date"], df[col].rolling(ROLL, min_periods=ROLL // 2).mean(),
                color=color, lw=2, label=label)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(frameon=False)
    style(ax, f"Ковзна точність (вікно {ROLL} матчів)")


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


if __name__ == "__main__":
    data = load(sys.argv[1] if len(sys.argv) > 1 else None)
    build_dashboard(data)
    plt.show()
