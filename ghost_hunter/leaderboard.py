"""Local capture records, grouped by mode and difficulty."""

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

MODES = ("Solo", "AI vs Human", "Human vs Human")
MAX_NAME_LENGTH = 20


class Leaderboard:
    def __init__(self, path=None):
        root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
                    or Path.home() / ".local" / "share")
        self.path = Path(path) if path is not None else root / "GhostHunter" / "leaderboard.sqlite3"

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS captures (
                        id INTEGER PRIMARY KEY, mode TEXT NOT NULL, difficulty TEXT NOT NULL,
                        name TEXT NOT NULL, milliseconds INTEGER NOT NULL, recorded_at TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE INDEX IF NOT EXISTS capture_rank
                    ON captures(mode, difficulty, milliseconds, id)
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS last_players (
                        mode TEXT NOT NULL, player INTEGER NOT NULL, name TEXT NOT NULL,
                        PRIMARY KEY (mode, player)
                    )
                """)
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def last_names(self, mode):
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT name FROM last_players WHERE mode = ? ORDER BY player", (mode,))
            return tuple(name for (name,) in rows)

    def remember_names(self, mode, names):
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM last_players WHERE mode = ?", (mode,))
            connection.executemany("INSERT INTO last_players VALUES (?, ?, ?)",
                                   [(mode, index, name) for index, name in enumerate(names)])

    def record(self, mode, difficulty, names, elapsed):
        self.record_times(mode, difficulty, [(name, elapsed) for name in names])

    def record_times(self, mode, difficulty, captures):
        """Store a round's successful captures together, with each hunter's own time."""
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                "INSERT INTO captures(mode, difficulty, name, milliseconds, recorded_at) VALUES (?, ?, ?, ?, ?)",
                [(mode, difficulty, name, max(1, round(elapsed * 1000)), timestamp) for name, elapsed in captures],
            )

    def top(self, mode, difficulty, limit=10):
        with closing(self._connect()) as connection:
            return connection.execute("""
                SELECT name, milliseconds, recorded_at FROM captures
                WHERE mode = ? AND difficulty = ? ORDER BY milliseconds, id LIMIT ?
            """, (mode, difficulty, limit)).fetchall()
