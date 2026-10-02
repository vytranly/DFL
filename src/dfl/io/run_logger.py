"""Ghi kết quả của một attempt vào <run_dir>/attempt-<k>/ (module metrics-io; SPEC §5.3; ADR-008; INV-6).

- Thư mục attempt được tạo bằng os.mkdir, vốn nguyên tử và raise FileExistsError khi đã có, nên kết quả cũ
  không bao giờ bị ghi đè.
- Mọi bản ghi được kiểm khóa theo TypedDict của contracts.py, rồi mã hóa JSON với allow_nan=False (NaN hay vô
  cực thì ValueError) và sort_keys=True, nên cùng bản ghi cho cùng byte (INV-6). Bản ghi được mã hóa xong mới
  ghi, nên bản ghi bị từ chối không để lại dòng dở.
- manifest.json và summary.json được ghi ra file tạm rồi os.replace, vốn nguyên tử trên POSIX. Attempt có
  summary.json là attempt đã xong, nên summary không bao giờ được nằm dở trên đĩa. Summary bị từ chối thì
  attempt vẫn mở, để lượt ghi SummaryFailed thay thế.

- Đọc cấu trúc results/ cho `run` và sweep: trạng thái từng attempt, số attempt kế tiếp, và khóa flock
  (ADR-008). fcntl chỉ có trên Unix nên chỉ được import bên trong `file_lock`, tức chỉ khi ghi vào results/.

os.mkdir, os.replace: https://docs.python.org/3.12/library/os.html#os.mkdir
https://docs.python.org/3.12/library/os.html#os.replace
json.dumps (allow_nan, sort_keys, ensure_ascii): https://docs.python.org/3.12/library/json.html#json.dumps
TypedDict.__required_keys__: https://docs.python.org/3.12/library/typing.html#typing.TypedDict.__required_keys__
fcntl.flock (thất bại thì OSError; với LOCK_NB, errno là EWOULDBLOCK khi khóa đang bị giữ):
https://docs.python.org/3.12/library/fcntl.html#fcntl.flock
https://man7.org/linux/man-pages/man2/flock.2.html
"""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from dfl.contracts import (
    ManifestRecord,
    RoundRecord,
    SummaryDiverged,
    SummaryFailed,
    SummaryOk,
    SummaryRecord,
    TimingRecord,
)

SUMMARY_SCHEMAS = {"ok": SummaryOk, "diverged": SummaryDiverged, "failed": SummaryFailed}
# Attempt có summary với một trong các status này là xong; mỗi run_id có nhiều nhất một attempt xong.
FINISHED = ("ok", "diverged")


class LockBusy(Exception):
    """Một tiến trình khác đang giữ khóa flock (chỉ khi gọi file_lock với wait=False)."""


@contextmanager
def file_lock(path: Path, *, wait: bool) -> Iterator[None]:
    """Giữ flock(LOCK_EX) trên `path` (tạo nếu chưa có); wait=False thì không đợi mà raise LockBusy.

    Khóa tự nhả khi file descriptor đóng, kể cả khi tiến trình chết, nên không có khóa bỏ quên phải xóa tay.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            if wait or err.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise
            raise LockBusy(f"{path} đang bị một tiến trình khác giữ") from err
        yield
    finally:
        os.close(fd)


def attempt_statuses(run_dir: Path) -> dict[int, str | None]:
    """k → status trong summary.json của attempt-<k>; None với attempt chưa có summary (bị ngắt)."""
    statuses: dict[int, str | None] = {}
    for path in run_dir.glob("attempt-*"):
        number = path.name.removeprefix("attempt-")
        if number.isdigit():
            summary = path / "summary.json"
            status = json.loads(summary.read_text(encoding="utf-8"))["status"] if summary.exists() else None
            statuses[int(number)] = status
    return statuses


def next_attempt(run_dir: Path) -> int:
    """Số attempt kế tiếp trong `run_dir`: 1 + số lớn nhất trong các thư mục attempt-<k> đã có."""
    return max(attempt_statuses(run_dir), default=0) + 1


def _check_keys(record: Mapping[str, object], schema: type, what: str) -> None:
    required = frozenset(schema.__required_keys__)
    missing = sorted(required - set(record))
    extra = sorted(set(record) - required)
    if missing or extra:
        raise ValueError(
            f"{what}: thiếu khóa {missing}, thừa khóa {extra} so với contracts.{schema.__name__}"
        )


def _encode(record: Mapping[str, object], *, indent: int | None = None) -> str:
    separators = None if indent else (",", ":")
    return json.dumps(
        record, allow_nan=False, sort_keys=True, ensure_ascii=False, indent=indent, separators=separators
    )


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


class JsonRunLogger:
    """RunLogger ghi JSON/JSONL; mỗi đối tượng ghi đúng một attempt, theo thứ tự start → log → finish."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._attempt_dir: Path | None = None
        self._finished = False

    @property
    def attempt_dir(self) -> Path | None:
        """Thư mục attempt sau start; None trước đó."""
        return self._attempt_dir

    def _open_attempt(self, what: str) -> Path:
        if self._attempt_dir is None:
            raise RuntimeError(f"{what} trước start")
        if self._finished:
            raise RuntimeError(f"{what} sau finish; attempt đã xong")
        return self._attempt_dir

    def start(self, manifest: ManifestRecord) -> None:
        if self._attempt_dir is not None:
            raise RuntimeError("start đã được gọi; mỗi logger ghi đúng một attempt")
        _check_keys(manifest, ManifestRecord, "manifest")
        attempt = manifest["attempt"]
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError(f"attempt phải là số nguyên ≥ 1, nhận {attempt!r}")
        text = _encode(manifest, indent=2)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = self._run_dir / f"attempt-{attempt}"
        os.mkdir(attempt_dir)
        _write_atomic(attempt_dir / "manifest.json", text)
        for name in ("rounds.jsonl", "timing.jsonl"):
            (attempt_dir / name).open("x", encoding="utf-8").close()
        self._attempt_dir = attempt_dir

    def _append(self, name: str, record: Mapping[str, object], schema: type, what: str) -> None:
        attempt_dir = self._open_attempt(what)
        _check_keys(record, schema, what)
        line = _encode(record)
        with (attempt_dir / name).open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def log_round(self, record: RoundRecord) -> None:
        self._append("rounds.jsonl", record, RoundRecord, "log_round")

    def log_timing(self, record: TimingRecord) -> None:
        self._append("timing.jsonl", record, TimingRecord, "log_timing")

    def finish(self, summary: SummaryRecord) -> None:
        attempt_dir = self._open_attempt("finish")
        status = summary.get("status")
        schema = SUMMARY_SCHEMAS.get(status) if isinstance(status, str) else None
        if schema is None:
            raise ValueError(
                f"status của summary phải là một trong {sorted(SUMMARY_SCHEMAS)}, nhận {status!r}"
            )
        _check_keys(summary, schema, "summary")
        _write_atomic(attempt_dir / "summary.json", _encode(summary, indent=2))
        self._finished = True
