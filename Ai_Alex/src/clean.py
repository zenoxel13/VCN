"""Очищення даних обох треків (клуби та збірні)."""
from src import clean_clubs, clean_intl


def main() -> None:
    """Запускає етап для обох треків (клуби, потім збірні)."""
    clean_clubs.main()
    clean_intl.main()


if __name__ == "__main__":
    main()
