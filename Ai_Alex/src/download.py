"""Завантаження сирих даних у data/raw/. Сирі файли після завантаження не змінюються."""
import importlib
import sys

import requests

from src.config import RAW, SOURCES


def download_file(name: str, urls: list[str], force: bool = False) -> bool:
    """Завантажує файл за першим робочим посиланням. Повертає True, якщо файл є на диску."""
    path = RAW / name
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"[download] {name}: вже є ({path.stat().st_size / 1e6:.1f} МБ), пропускаю")
        return True
    for url in urls:
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            print(f"[download] {name}: завантажено ({len(resp.content) / 1e6:.1f} МБ) з {url}")
            return True
        except requests.RequestException as err:
            print(f"[download] {name}: помилка {err} для {url}")
    print(f"[download] {name}: НЕДОСТУПНЕ — джерело пропущено")
    return False


def main(force: bool = False) -> None:
    """Завантажує всі джерела з config.SOURCES; падає, якщо немає ключових файлів."""
    results = {name: download_file(name, urls, force) for name, urls in SOURCES.items()}
    missing = [n for n, ok in results.items() if not ok]
    if "Matches.csv" in missing or "results.csv" in missing:
        raise RuntimeError(f"Не вдалося завантажити ключові файли: {missing}")
    download_optional_sources()


def download_optional_sources() -> None:
    """Завантажує додаткові джерела (Transfermarkt, REEP); їх відсутність не зупиняє пайплайн."""
    try:
        importlib.import_module("src.download_extra").main()
    except ImportError:
        print("[download] модуль download_extra відсутній — додаткові джерела пропущено")
    except Exception as err:  # noqa: BLE001 — додаткове джерело не повинно валити пайплайн
        print(f"[download] додаткові джерела недоступні ({type(err).__name__}: {err}) — пропущено")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
