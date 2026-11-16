"""Побудова ознак обох треків (клуби та збірні)."""
import importlib

from src import features_clubs, features_intl


def build_optional_tm_state() -> None:
    """Будує стан команд за Transfermarkt, якщо модуль і дані є.

    Це додаткове джерело: без нього пайплайн має працювати (ознаки f_tm_ будуть порожні).
    """
    try:
        importlib.import_module("src.features_tm").main()
    except ImportError:
        print("[features] модуль features_tm відсутній — ознаки Transfermarkt пропущено")
    except Exception as err:  # noqa: BLE001 — додаткове джерело не повинно валити пайплайн
        print(f"[features] Transfermarkt недоступний ({type(err).__name__}: {err}) — ознаки f_tm_ будуть порожні")


def main() -> None:
    """Запускає етап для обох треків (Transfermarkt → клуби → збірні)."""
    build_optional_tm_state()
    features_clubs.main()
    features_intl.main()


if __name__ == "__main__":
    main()
