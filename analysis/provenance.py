"""Nguồn gốc của hình: analysis_version, trạng thái git của analysis/, figures/PROVENANCE.json (module
analysis; T16a; SPEC INT-1).

- `analysis_version` là sha256 của mọi file `analysis/*.py`, tính như `code_version` (ADR-008): từng cặp
  (đường dẫn tương đối, nội dung với `\\r\\n` đổi thành `\\n`), mỗi phần có độ dài đứng trước. analysis/ không
  được import dfl.io (contract `analysis-reads-only`), nên cách tính được viết lại ở đây.
- `analysis_dirty` hỏi `git status --porcelain -- analysis`; không hỏi được (không có git) thì coi là bẩn, để
  không bao giờ ghi vào figures/ từ code phân tích không truy được.
- PROVENANCE.json giữ mục của hình không được vẽ lại lần này, nên bỏ qua một hình không xóa nguồn gốc của nó.

hashlib.sha256: https://docs.python.org/3.12/library/hashlib.html
subprocess.run: https://docs.python.org/3.12/library/subprocess.html#subprocess.run
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent


def _part(digest: Any, data: bytes) -> None:
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def analysis_version(directory: Path = ANALYSIS_DIR) -> str:
    digest = hashlib.sha256()
    sources = sorted(
        (path.relative_to(directory.parent).as_posix(), path) for path in directory.rglob("*.py")
    )
    for name, path in sources:
        _part(digest, name.encode())
        _part(digest, path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def analysis_dirty() -> bool:
    status = _git("status", "--porcelain", "--", "analysis")
    return status is None or status.strip() != ""


def git_commit() -> str:
    head = _git("rev-parse", "HEAD")
    return head.strip() if head else "none"


def update(path: Path, *, command: str, draft: bool, figures: dict[str, dict[str, Any]]) -> None:
    """Ghi PROVENANCE.json: mục mới của các hình vừa vẽ, giữ mục cũ của hình khác."""
    version, commit = analysis_version(), git_commit()
    old = json.loads(path.read_text(encoding="utf-8")).get("figures", {}) if path.exists() else {}
    stamped = {
        name: {**entry, "command": command, "analysis_version": version, "commit": commit, "draft": draft}
        for name, entry in figures.items()
    }
    record = {
        "command": command,
        "analysis_version": version,
        "commit": commit,
        "draft": draft,
        "figures": {**old, **stamped},
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
