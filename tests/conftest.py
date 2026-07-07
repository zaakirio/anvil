import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from anvil import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Keep backend state and the cost ledger out of the repo during tests."""
    monkeypatch.setattr(config, "BACKEND_DATA_DIR", tmp_path / "backend")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    yield


def db_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(config.DATABASE_URL, connect_timeout=2):
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not db_available(), reason="postgres not reachable")
