from types import SimpleNamespace

from sqlalchemy import text

from valuecell.server.db.connection import DatabaseManager


def test_database_manager_enables_sqlite_foreign_keys(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'foreign-keys.db'}"
    monkeypatch.setattr(
        "valuecell.server.db.connection.get_settings",
        lambda: SimpleNamespace(
            get_database_config=lambda: {"url": database_url}
        ),
    )

    manager = DatabaseManager()
    with manager.get_engine().connect() as connection:
        enabled = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert enabled == 1
