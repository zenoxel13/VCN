"""Повний пайплайн однією командою: python -m src.run_all"""
import time

from src import clean, download, evaluate, features, train


def main() -> None:
    """Послідовно виконує всі етапи пайплайна й друкує час кожного."""
    for step in (download, clean, features, train, evaluate):
        t0 = time.time()
        print(f"\n===== Етап: {step.__name__} =====")
        step.main()
        print(f"===== {step.__name__}: готово за {time.time() - t0:.0f} с =====")


if __name__ == "__main__":
    main()
