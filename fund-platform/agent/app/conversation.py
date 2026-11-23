"""SQLite conversation storage: whole turns only, ten turns, isolated IDs."""
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


def validate_thread_id(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("thread_id 仅允许1到80位字母、数字、下划线和连字符")
    return value


class ConversationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.turn_lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS messages (thread_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, mode TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(thread_id,seq))")

    @contextmanager
    def turn(self):
        # One worker; serializing complete turns also prevents clear/generation races.
        with self.turn_lock:
            yield

    def history(self, thread_id):
        validate_thread_id(thread_id)
        with self.lock:
            return [dict(row) for row in self.db.execute(
                "SELECT * FROM messages WHERE thread_id=? ORDER BY seq", (thread_id,))]

    def append_turn(self, thread_id, question, result):
        import json
        validate_thread_id(thread_id)
        stamp = datetime.now(timezone.utc).isoformat()
        with self.lock, self.db:
            seq = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM messages WHERE thread_id=?", (thread_id,)).fetchone()[0]
            for offset, role, content in ((1, 'user', question), (2, 'assistant', result['answer'])):
                self.db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?)",
                    (thread_id, seq+offset, role, content, result['mode'], stamp,
                     json.dumps(result if role == 'assistant' else {}, ensure_ascii=False)))
            self.db.execute("DELETE FROM messages WHERE thread_id=? AND seq<=?", (thread_id, seq+2-20))

    def clear(self, thread_id):
        validate_thread_id(thread_id)
        with self.turn_lock, self.lock, self.db:
            self.db.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))

    def close(self):
        with self.lock:
            self.db.close()
