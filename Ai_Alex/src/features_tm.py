"""Ознаки Transfermarkt (f_tm_*) для клубних матчів: вартість складу, вік, стаж тренера.

Принципи проти витоку:
- склад команди перед матчем = гравці, що виходили на поле в ПОПЕРЕДНІХ N матчах клубу;
- оцінка гравця — остання, датована СТРОГО раніше дня матчу (merge_asof, allow_exact_matches=False);
- жодних «поточних» полів-знімків (clubs.total_market_value, players.market_value_in_eur тощо);
- тренер поточного матчу відомий до гри, стаж рахується лише за попередніми матчами.

API:
    main()                      -> будує data/processed/tm_team_state.parquet
    add_tm_features(matches)    -> matches + f_tm_* колонки
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TM_DIR = ROOT / "data" / "raw" / "transfermarkt"
STATE_PATH = ROOT / "data" / "processed" / "tm_team_state.parquet"
MATCHES_PATH = ROOT / "data" / "processed" / "clubs_matches.parquet"

N_PREV = 10          # скільки попередніх матчів клубу формують «склад»
NEW_COACH_GAMES = 5  # менше стількох матчів під тренером — «новий тренер»
STALE_DAYS = 120    # стан старший за це (клуб випав з покриття TM) не використовуємо
MIN_MATCHES = 5      # мінімум збігів для зіставлення club_id ↔ назва

STATE_COLS = ["team", "date", "value", "n_players", "avg_age", "spell_start", "spell_games"]
TM_FEATURES = [
    "f_tm_home_value_log", "f_tm_away_value_log", "f_tm_value_log_diff", "f_tm_home_value_share",
    "f_tm_home_avg_age", "f_tm_away_avg_age", "f_tm_age_diff",
    "f_tm_home_coach_days", "f_tm_away_coach_days",
    "f_tm_home_new_coach", "f_tm_away_new_coach",
]

_STATE_CACHE: pd.DataFrame | None = None


def _read(name: str, usecols=None) -> pd.DataFrame:
    """Читає csv.gz із data/raw/transfermarkt."""
    return pd.read_csv(TM_DIR / f"{name}.csv.gz", usecols=usecols, low_memory=False)


def _club_games(games: pd.DataFrame) -> pd.DataFrame:
    """Перетворює games у довгий формат: один рядок = (клуб, матч) з тренером."""
    cols = ["game_id", "date"]
    h = games[cols + ["home_club_id", "home_club_manager_name"]].rename(
        columns={"home_club_id": "club_id", "home_club_manager_name": "manager"})
    a = games[cols + ["away_club_id", "away_club_manager_name"]].rename(
        columns={"away_club_id": "club_id", "away_club_manager_name": "manager"})
    cg = pd.concat([h, a], ignore_index=True).dropna(subset=["club_id", "date"])
    cg["club_id"] = cg["club_id"].astype(int)
    cg = cg.drop_duplicates(["club_id", "game_id"]).sort_values(["club_id", "date", "game_id"])
    cg["idx"] = cg.groupby("club_id").cumcount()
    return cg.reset_index(drop=True)


def _map_clubs(games: pd.DataFrame, ours: pd.DataFrame) -> dict[int, str]:
    """Зіставляє club_id Transfermarkt з нашими назвами через збіг матчів (дата, рахунок).

    Для кожної пари (club_id, назва) рахуємо збіги; беремо найчастішу назву з ≥ MIN_MATCHES збігами
    яка втричі частіша за другу; одну назву віддаємо лише одному club_id.
    """
    g = games.dropna(subset=["home_club_goals", "away_club_goals"]).copy()
    o = ours.dropna(subset=["home_goals", "away_goals"])[["date", "home", "away", "home_goals", "away_goals"]].copy()
    for df, hg, ag in ((g, "home_club_goals", "away_club_goals"), (o, "home_goals", "away_goals")):
        df["k"] = df["date"].dt.strftime("%Y%m%d") + "_" + df[hg].astype(int).astype(str) + "_" + df[ag].astype(int).astype(str)
    m = g[["k", "home_club_id", "away_club_id"]].merge(o[["k", "home", "away"]], on="k")
    pairs = pd.concat([
        m[["home_club_id", "home"]].set_axis(["club_id", "team"], axis=1),
        m[["away_club_id", "away"]].set_axis(["club_id", "team"], axis=1),
    ])
    cnt = pairs.value_counts().rename("n").reset_index()
    cnt = cnt.sort_values(["club_id", "n"], ascending=[True, False])
    cnt["second"] = cnt.groupby("club_id")["n"].shift(-1).fillna(0)
    best = cnt.drop_duplicates("club_id")
    # найчастіша назва, ≥ MIN_MATCHES збігів і втричі частіше за другу кандидатуру
    best = best[(best["n"] >= MIN_MATCHES) & (best["n"] >= 3 * best["second"])]
    best = best.sort_values("n", ascending=False).drop_duplicates("team")
    return dict(zip(best["club_id"].astype(int), best["team"]))


def _empty_state() -> pd.DataFrame:
    """Порожня таблиця стану з правильними колонками."""
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in ("date", "spell_start") else
                                      ("object" if c == "team" else "float64")) for c in STATE_COLS})


def build_state() -> pd.DataFrame:
    """Будує передматчевий стан кожного клубу на кожен його матч у Transfermarkt (у наших назвах)."""
    games = _read("games", ["game_id", "date", "home_club_id", "away_club_id", "home_club_goals",
                            "away_club_goals", "home_club_manager_name", "away_club_manager_name"])
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    ours = pd.read_parquet(MATCHES_PATH, columns=["date", "home", "away", "home_goals", "away_goals"])
    mapping = _map_clubs(games, ours)
    print(f"Зіставлено клубів Transfermarkt ↔ наші назви: {len(mapping)}")

    cg = _club_games(games)
    cg = cg[cg["club_id"].isin(mapping)].copy()

    # --- тренер: стаж за попередніми матчами під поточним тренером ---
    cg["manager"] = cg["manager"].fillna("?")
    new_spell = (cg["manager"] != cg.groupby("club_id")["manager"].shift()) | (cg["idx"] == 0)
    cg["spell"] = new_spell.cumsum()
    cg["spell_start"] = cg.groupby("spell")["date"].transform("min")
    cg["spell_games"] = cg.groupby("spell").cumcount().astype(float)  # матчів ДО поточного
    unknown = cg["manager"] == "?"
    cg.loc[unknown, ["spell_games"]] = np.nan
    cg.loc[unknown, "spell_start"] = pd.NaT

    have_squad = all((TM_DIR / f"{n}.csv.gz").exists() for n in ("appearances", "player_valuations", "players"))
    if not have_squad:
        print("ПОПЕРЕДЖЕННЯ: немає appearances/player_valuations/players — лише ознаки тренера")
        st = cg.assign(value=np.nan, n_players=np.nan, avg_age=np.nan)
        st["team"] = st["club_id"].map(mapping)
        st = st[STATE_COLS].sort_values(["team", "date"])
        return st.drop_duplicates(["team", "date"], keep="last").reset_index(drop=True)

    # --- склад: гравці з попередніх N матчів клубу ---
    app = _read("appearances", ["game_id", "player_id", "player_club_id"])
    app = app.rename(columns={"player_club_id": "club_id"})
    app = app.merge(cg[["club_id", "game_id", "idx"]], on=["club_id", "game_id"])
    pool = pd.concat([app.assign(t=app["idx"] + s) for s in range(1, N_PREV + 1)], ignore_index=True)
    pool = pool[["club_id", "t", "player_id"]].drop_duplicates()
    pool = pool.merge(cg[["club_id", "idx", "date"]].rename(columns={"idx": "t"}), on=["club_id", "t"])

    # оцінки строго до дня матчу
    val = _read("player_valuations", ["player_id", "date", "market_value_in_eur"])
    val["date"] = pd.to_datetime(val["date"], errors="coerce")
    val = val.dropna().sort_values("date")
    pool = pool.sort_values("date")
    pool = pd.merge_asof(pool, val, on="date", by="player_id", allow_exact_matches=False)

    players = _read("players", ["player_id", "date_of_birth"])
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    pool = pool.merge(players, on="player_id", how="left")
    pool["age"] = (pool["date"] - pool["date_of_birth"]).dt.days / 365.25

    agg = pool.groupby(["club_id", "t"]).agg(
        value=("market_value_in_eur", "sum"), n_valued=("market_value_in_eur", "count"),
        n_players=("player_id", "nunique"), avg_age=("age", "mean")).reset_index()
    agg.loc[agg["n_valued"] == 0, "value"] = np.nan
    st = cg.merge(agg.rename(columns={"t": "idx"}), on=["club_id", "idx"], how="left")
    st.loc[st["idx"] < 3, ["value", "avg_age"]] = np.nan  # замало попередніх матчів
    st["team"] = st["club_id"].map(mapping)
    st = st[STATE_COLS].sort_values(["team", "date"])
    # якщо в один день кілька записів — лишаємо останній
    return st.drop_duplicates(["team", "date"], keep="last").reset_index(drop=True)


def _load_state() -> pd.DataFrame:
    """Повертає закешовану таблицю стану (або порожню)."""
    global _STATE_CACHE
    if _STATE_CACHE is None:
        _STATE_CACHE = pd.read_parquet(STATE_PATH) if STATE_PATH.exists() else _empty_state()
        _STATE_CACHE["date"] = pd.to_datetime(_STATE_CACHE["date"]).astype("datetime64[ns]")
        _STATE_CACHE = _STATE_CACHE.sort_values("date").reset_index(drop=True)
    return _STATE_CACHE


def add_tm_features(matches: pd.DataFrame) -> pd.DataFrame:
    """matches має колонки match_id, date, home, away (назви як у clubs_matches.parquet).

    Повертає ту саму таблицю + колонки f_tm_* (NaN, де даних немає). Лише інформація, відома ДО date:
    стан береться з останнього запису клубу з датою ≤ date (запис на саму дату — це передматчевий стан
    цього ж матчу, побудований лише з попередніх ігор і оцінок до дня матчу).
    """
    out = matches.copy()
    st = _load_state()
    key = pd.DataFrame({"_i": np.arange(len(out)), "date": pd.to_datetime(out["date"]).astype("datetime64[ns]").values})
    for side in ("home", "away"):
        k = key.assign(team=out[side].astype(str).values).sort_values("date")
        if len(st):
            r = pd.merge_asof(k, st, on="date", by="team", direction="backward",
                              tolerance=pd.Timedelta(days=STALE_DAYS))
            # дати після кінця даних Transfermarkt (демо, майбутнє) — останній відомий стан команди
            fut = (k["date"] > st["date"].max()).values
            if fut.any():
                last = st.drop_duplicates("team", keep="last").set_index("team")
                rf = k[fut].join(last.drop(columns="date"), on="team")
                r = r.set_index("_i")
                r.loc[rf["_i"].values, last.columns.drop("date")] = rf[last.columns.drop("date")].values
                r = r.reset_index()
        else:
            r = k.assign(**{c: np.nan for c in STATE_COLS if c not in ("team", "date")})
            r["spell_start"] = pd.NaT
        r = r.sort_values("_i")
        out[f"f_tm_{side}_value_log"] = np.log1p(r["value"].astype(float)).values
        out[f"f_tm_{side}_avg_age"] = r["avg_age"].astype(float).values
        days = (pd.to_datetime(r["date"]) - pd.to_datetime(r["spell_start"])).dt.days
        out[f"f_tm_{side}_coach_days"] = days.astype(float).values
        out[f"f_tm_{side}_new_coach"] = (r["spell_games"].astype(float) < NEW_COACH_GAMES).astype(float).where(
            r["spell_games"].notna()).values
    out["f_tm_value_log_diff"] = out["f_tm_home_value_log"] - out["f_tm_away_value_log"]
    hv, av = np.expm1(out["f_tm_home_value_log"]), np.expm1(out["f_tm_away_value_log"])
    out["f_tm_home_value_share"] = hv / (hv + av)
    out["f_tm_age_diff"] = out["f_tm_home_avg_age"] - out["f_tm_away_avg_age"]
    return out


def _report(st: pd.DataFrame) -> None:
    """Друкує перевірки: покриття, кореляції, топ-10 клубів за вартістю."""
    m = pd.read_parquet(MATCHES_PATH)
    t0 = time.time()
    f = add_tm_features(m[["match_id", "date", "home", "away"]])
    print(f"add_tm_features на {len(m)} матчах: {time.time() - t0:.2f} с")
    t0 = time.time()
    add_tm_features(m[["match_id", "date", "home", "away"]].tail(1))
    print(f"add_tm_features на 1 рядку: {time.time() - t0:.3f} с")
    ok = f["f_tm_value_log_diff"].notna()
    print(f"Покриття f_tm_value_log_diff: {ok.mean():.1%}")
    print("По лігах:\n", ok.groupby(m["division"]).mean().round(3).sort_values(ascending=False).head(25).to_string())
    print("По роках:\n", ok.groupby(m["date"].dt.year).mean().round(3).to_string())
    for c in TM_FEATURES:
        print(f"  {c}: заповнено {f[c].notna().mean():.1%}")
    gd = m["home_goals"] - m["away_goals"]
    elo = m["HomeElo"] - m["AwayElo"]
    print(f"corr(value_log_diff, різниця голів) = {f['f_tm_value_log_diff'].corr(gd):.3f}")
    print(f"corr(value_log_diff, різниця Elo)   = {f['f_tm_value_log_diff'].corr(elo):.3f}")
    print(f"corr(різниця Elo, різниця голів) на тих самих матчах = {elo[ok].corr(gd[ok]):.3f}")
    if st["value"].isna().all():
        print("Вартості немає (сирі файли складу відсутні)")
        return
    last = st.sort_values("date").drop_duplicates("team", keep="last").nlargest(10, "value")
    print("Топ-10 клубів за вартістю (останній стан):")
    print((last[["team", "date"]].assign(value_meur=(last["value"] / 1e6).round(0))).to_string(index=False))


def main() -> None:
    """Будує data/processed/tm_team_state.parquet; не падає, якщо сирих даних немає."""
    global _STATE_CACHE
    t0 = time.time()
    need = ["games"]
    missing = [n for n in need if not (TM_DIR / f"{n}.csv.gz").exists()]
    if missing:
        print(f"ПОПЕРЕДЖЕННЯ: немає сирих файлів {missing}; пишу порожню таблицю стану (f_tm_* = NaN)")
        st = _empty_state()
    else:
        st = build_state()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(STATE_PATH, index=False)
    _STATE_CACHE = None
    print(f"Збережено {STATE_PATH} ({len(st)} рядків) за {time.time() - t0:.1f} с")
    if len(st):
        _report(st)


if __name__ == "__main__":
    main()
