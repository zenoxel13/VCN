"""Завантаження додаткових відкритих даних: Transfermarkt (dcaribou/transfermarkt-datasets) і REEP.

Файли зберігаються в data/raw/transfermarkt/ і data/raw/reep/. Уже завантажені файли пропускаються.
Якщо джерело недоступне — друкується попередження, функція не падає.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TM_DIR = ROOT / "data" / "raw" / "transfermarkt"
REEP_DIR = ROOT / "data" / "raw" / "reep"

TM_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TM_FILES = ["games.csv.gz", "clubs.csv.gz", "player_valuations.csv.gz",
            "players.csv.gz", "appearances.csv.gz"]
REEP_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data/teams.csv"


def _download(url: str, dest: Path, tries: int = 3) -> bool:
    """Потоково завантажує url у dest (через тимчасовий файл). Повертає True при успіху."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  вже є: {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            print(f"  завантажено: {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  спроба {attempt}/{tries} для {url} невдала: {exc}")
            time.sleep(2)
    print(f"  ПОПЕРЕДЖЕННЯ: не вдалося завантажити {url}; ознаки f_tm_ будуть NaN")
    return False


def _download_parallel(url: str, dest: Path, workers: int = 48, chunk: int = 1 << 20) -> bool:
    """Завантаження діапазонами (HTTP Range) у багато потоків — джерело обмежує швидкість на з'єднання."""
    from concurrent.futures import ThreadPoolExecutor

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  вже є: {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
        return True
    try:
        size = int(requests.head(url, timeout=30).headers["Content-Length"])
    except Exception as exc:  # noqa: BLE001
        print(f"  не вдалося дізнатися розмір {url}: {exc}; пробую звичайне завантаження")
        return _download(url, dest)
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]

    def get(rng):
        for _ in range(5):
            try:
                r = requests.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"}, timeout=120)
                if r.status_code in (200, 206) and len(r.content) == rng[1] - rng[0] + 1:
                    return r.content
            except Exception:  # noqa: BLE001
                time.sleep(1)
        raise RuntimeError(f"діапазон {rng} не завантажено")

    try:
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(get, ranges))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for p in parts:
                fh.write(p)
        tmp.replace(dest)
        print(f"  завантажено: {dest.name} ({size / 1e6:.1f} МБ)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ПОПЕРЕДЖЕННЯ: {url} не завантажено ({exc}); ознаки f_tm_ будуть NaN")
        return False


def main() -> None:
    """Завантажує всі потрібні файли; не падає при недоступності джерел."""
    print("Transfermarkt:")
    for f in TM_FILES:
        _download_parallel(TM_BASE + f, TM_DIR / f, chunk=256 * 1024)
    print("REEP:")
    _download(REEP_URL, REEP_DIR / "teams.csv")


if __name__ == "__main__":
    main()
