import json
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).with_name("reports.json")


class LocalCollection:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        return data if isinstance(data, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def insert_one(self, document: dict[str, Any]) -> None:
        items = self._read()
        items.append(document)
        self._write(items)

    def find(
        self, _query: dict[str, Any] | None = None, _projection: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._read()


collection = LocalCollection(DB_PATH)
