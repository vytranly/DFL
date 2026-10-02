"""Nguồn gốc của một attempt: code_version, run_id, máy, trạng thái git, môi trường, và ManifestRecord
(module metrics-io; KH §4.7; SPEC §5.3; ADR-008; G13, G20).

Module này không import torch: sweep tính run_id cho mọi ô của lưới trước khi chạy, nên việc đó phải rẻ.

- code_version: sha256 của dãy cặp (đường dẫn tính từ thư mục chứa gói, nội dung) của mọi file *.py trong gói
  `dfl` đang được nạp (dfl.__file__), không phải của thư mục đang đứng, nên lượt chạy từ một worktree mang
  đúng code của worktree đó. CRLF được đổi thành LF trước khi hash. Phần cuối là requirements.lock: các dòng
  bỏ khoảng trắng thừa, bỏ dòng trống, rồi sắp xếp. Mỗi phần được hash kèm độ dài của nó, nên ranh giới giữa
  đường dẫn và nội dung không mơ hồ.
- run_id: 16 ký tự hex đầu của sha256 của (cấu hình chuẩn hóa, code_version, máy).
- Máy: tên CPU kèm 12 ký tự hex đầu của sha256 danh sách cờ CPU (đã sắp xếp) trong /proc/cpuinfo. Không có
  /proc/cpuinfo (laptop demo) thì dùng tên nền tảng.
- Cây sạch (SPEC §5.3): `git status --porcelain -- src configs requirements.lock` rỗng, tính cả file chưa
  track, và không có file *.py, *.yaml, *.yml nào bị ignore trong src/, configs/ (trừ trong __pycache__ và
  *.egg-info). Không có git thì commit là "none" và cây được coi là bẩn.
- Môi trường: so từng gói của requirements.lock với phiên bản đã cài; gói không có trong file khóa (công cụ
  dev) được bỏ qua.

hashlib.sha256: https://docs.python.org/3.12/library/hashlib.html
importlib.metadata.version, PackageNotFoundError: https://docs.python.org/3.12/library/importlib.metadata.html
git status (--porcelain, -z; --ignored với --untracked-files=all hiện từng file trong thư mục bị ignore):
https://git-scm.com/docs/git-status
git rev-parse (HEAD, --show-toplevel): https://git-scm.com/docs/git-rev-parse
/proc/cpuinfo: https://man7.org/linux/man-pages/man5/proc_cpuinfo.5.html
platform.processor, platform.machine, platform.python_version: https://docs.python.org/3.12/library/platform.html
datetime.isoformat: https://docs.python.org/3.12/library/datetime.html#datetime.datetime.isoformat
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import dfl
from dfl.config import RunConfig, canonical_json
from dfl.contracts import ManifestRecord

LOCK_FILE = "requirements.lock"
CPUINFO = Path("/proc/cpuinfo")
RUN_ID_LENGTH = 16
FLAGS_HASH_LENGTH = 12
NUM_CLASSES = 10
# Cây sạch (SPEC §5.3): đường dẫn theo dõi, và đuôi của file bị ignore nhưng vẫn ảnh hưởng kết quả.
TRACKED_PATHS = ("src", "configs", LOCK_FILE)
SOURCE_SUFFIXES = (".py", ".yaml", ".yml")


def package_dir() -> Path:
    """Thư mục của gói dfl đang được nạp (src/dfl khi cài editable)."""
    return Path(dfl.__file__).resolve().parent


def repo_root() -> Path:
    """Gốc repo chứa src/dfl và requirements.lock."""
    return package_dir().parents[1]


def _hash_part(update: Callable[[bytes], None], data: bytes) -> None:
    update(len(data).to_bytes(8, "big"))
    update(data)


def code_version(package: Path, lock_file: Path) -> str:
    """sha256 hex của mọi file *.py trong `package` cùng file khóa (ADR-008)."""
    if not lock_file.is_file():
        raise FileNotFoundError(f"không thấy {lock_file}; code_version cần file khóa {LOCK_FILE} (QS-1)")
    digest = hashlib.sha256()
    sources = sorted((path.relative_to(package.parent).as_posix(), path) for path in package.rglob("*.py"))
    for name, path in sources:
        _hash_part(digest.update, name.encode())
        _hash_part(digest.update, path.read_bytes().replace(b"\r\n", b"\n"))
    lines = sorted(
        line.strip() for line in lock_file.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    _hash_part(digest.update, LOCK_FILE.encode())
    _hash_part(digest.update, "\n".join(lines).encode())
    return digest.hexdigest()


def current_code_version() -> str:
    return code_version(package_dir(), repo_root() / LOCK_FILE)


def machine(cpuinfo: str | None = None) -> str:
    """Tên CPU kèm hash cờ CPU (G13); `cpuinfo` là nội dung /proc/cpuinfo, None thì tự đọc."""
    if cpuinfo is None:
        cpuinfo = CPUINFO.read_text() if CPUINFO.exists() else ""
    fields: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())  # giữ giá trị của processor đầu tiên
    name = fields.get("model name") or platform.processor() or platform.machine() or "unknown"
    # x86 ghi "flags", ARM ghi "Features".
    flags = " ".join(sorted(set((fields.get("flags") or fields.get("Features") or "").split())))
    return f"{name} (flags {hashlib.sha256(flags.encode()).hexdigest()[:FLAGS_HASH_LENGTH]})"


def run_id(config: RunConfig, code_version: str, machine: str) -> str:
    payload = json.dumps([canonical_json(config), code_version, machine], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:RUN_ID_LENGTH]


def _git(root: Path, *args: str) -> str | None:
    """stdout của một lệnh git, hoặc None khi không có git hay lệnh lỗi."""
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.decode("utf-8", "surrogateescape") if result.returncode == 0 else None


def _is_ignored_source(entry: str) -> bool:
    if not entry.startswith("!! "):
        return False
    path = PurePosixPath(entry[3:])
    in_cache = any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)
    return path.name.endswith(SOURCE_SUFFIXES) and not in_cache


def git_state(root: Path) -> tuple[str, bool]:
    """(commit HEAD, cây có bẩn không) của repo có gốc ở `root`; không có git thì ("none", True)."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.strip()).resolve() != root.resolve():
        return "none", True
    commit = _git(root, "rev-parse", "HEAD")
    changed = _git(root, "status", "--porcelain", "-z", "--", *TRACKED_PATHS)
    ignored = _git(
        root, "status", "--porcelain", "-z", "--ignored", "--untracked-files=all", "--", "src", "configs"
    )
    if commit is None or changed is None or ignored is None:
        return "none", True
    dirty = bool(changed) or any(_is_ignored_source(entry) for entry in ignored.split("\0"))
    return commit.strip(), dirty


def read_lock(lock_file: Path) -> dict[str, str]:
    """Tên → phiên bản của mọi dòng `tên==phiên bản` trong file khóa; dòng khác thì raise ValueError."""
    pins: dict[str, str] = {}
    for raw in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not (sep and name and version) or any(c in name for c in " <>=!~;@"):
            raise ValueError(f"dòng {line!r} của {lock_file.name} không có dạng tên==phiên bản")
        pins[name] = version
    return pins


@dataclass(frozen=True, slots=True)
class EnvCheck:
    env_match: bool
    # "python" và mọi gói của file khóa → phiên bản đã cài ("missing" nếu chưa cài).
    versions: dict[str, str]
    # Mô tả từng gói lệch, để cảnh báo hay từ chối ghi vào results/.
    mismatches: tuple[str, ...]


def check_environment(lock: Mapping[str, str]) -> EnvCheck:
    versions = {"python": platform.python_version()}
    mismatches = []
    for name, pinned in lock.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        versions[name] = installed
        if installed != pinned:
            mismatches.append(f"{name}: file khóa ghi {pinned}, đã cài {installed}")
    return EnvCheck(env_match=not mismatches, versions=versions, mismatches=tuple(mismatches))


@dataclass(frozen=True, slots=True)
class Provenance:
    """Phần của manifest không phụ thuộc cấu hình lượt chạy; tính một lần mỗi tiến trình."""

    code_version: str
    git_commit: str
    dirty: bool
    env_match: bool
    versions: dict[str, str]
    machine: str
    mismatches: tuple[str, ...] = ()


def is_linked_worktree(root: Path) -> bool:
    """True khi `root` là gốc của một worktree liên kết (`git worktree add`), không phải cây chính.

    Ở worktree liên kết, `--git-dir` là .git/worktrees/<tên> của repo chính, khác `--git-common-dir`.
    https://git-scm.com/docs/git-rev-parse#Documentation/git-rev-parse.txt---git-common-dir
    """
    git_dir = _git(root, "rev-parse", "--git-dir")
    common_dir = _git(root, "rev-parse", "--git-common-dir")
    if git_dir is None or common_dir is None:
        return False
    return (root / git_dir.strip()).resolve() != (root / common_dir.strip()).resolve()


def current_provenance() -> Provenance:
    return provenance_of(repo_root())


def provenance_of(root: Path) -> Provenance:
    """Provenance của code trong `root`/src/dfl cùng `root`/requirements.lock, và của môi trường hiện tại."""
    lock_file = root / LOCK_FILE
    commit, dirty = git_state(root)
    env = check_environment(read_lock(lock_file))
    return Provenance(
        code_version=code_version(root / "src" / "dfl", lock_file),
        git_commit=commit,
        dirty=dirty,
        env_match=env.env_match,
        versions=env.versions,
        machine=machine(),
        mismatches=env.mismatches,
    )


def _record(
    config: RunConfig,
    *,
    provenance: Provenance,
    attempt: int,
    child_seeds: Mapping[str, int],
    partition_retries: int,
    cpu_threads: int,
    spectral_gap: float | None,
    objective_weights: Sequence[float],
    objective_shift_tv: float,
    node_class_counts: Sequence[Sequence[int]],
) -> ManifestRecord:
    if attempt < 1:
        raise ValueError(f"attempt đếm từ 1, nhận {attempt}")
    return {
        "run_id": run_id(config, provenance.code_version, provenance.machine),
        "code_version": provenance.code_version,
        "git_commit": provenance.git_commit,
        "dirty": provenance.dirty,
        "env_match": provenance.env_match,
        "attempt": attempt,
        "config": json.loads(canonical_json(config)),
        "seed": config.seed,
        "child_seeds": dict(child_seeds),
        "partition_retries": partition_retries,
        "versions": dict(provenance.versions),
        "machine": provenance.machine,
        "cpu_threads": cpu_threads,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "spectral_gap": spectral_gap,
        "objective_weights": [float(w) for w in objective_weights],
        "objective_shift_tv": objective_shift_tv,
        "node_class_counts": [[int(c) for c in row] for row in node_class_counts],
    }


def build_manifest(
    config: RunConfig,
    *,
    provenance: Provenance,
    attempt: int,
    child_seeds: Mapping[str, int],
    partition_retries: int,
    cpu_threads: int,
    spectral_gap: float | None,
    objective_weights: Sequence[float],
    objective_shift_tv: float,
    node_class_counts: Sequence[Sequence[int]],
) -> ManifestRecord:
    """ManifestRecord của attempt `attempt` (đếm từ 1); các dãy theo node phải có đúng N phần tử."""
    n = config.num_nodes
    if len(objective_weights) != n:
        raise ValueError(f"objective_weights cần {n} phần tử (N), nhận {len(objective_weights)}")
    if len(node_class_counts) != n or any(len(row) != NUM_CLASSES for row in node_class_counts):
        raise ValueError(f"node_class_counts phải là ma trận N × {NUM_CLASSES} = {n} × {NUM_CLASSES}")
    return _record(
        config,
        provenance=provenance,
        attempt=attempt,
        child_seeds=child_seeds,
        partition_retries=partition_retries,
        cpu_threads=cpu_threads,
        spectral_gap=spectral_gap,
        objective_weights=objective_weights,
        objective_shift_tv=objective_shift_tv,
        node_class_counts=node_class_counts,
    )


def unassembled_manifest(
    config: RunConfig,
    *,
    provenance: Provenance,
    attempt: int,
    child_seeds: Mapping[str, int],
    cpu_threads: int,
) -> ManifestRecord:
    """Manifest của attempt hỏng trước khi lắp xong lượt chạy, ví dụ vượt 1,000 lần lấy mẫu lại (REQ-02).

    Các trường chỉ có sau khi lắp (partition_retries, spectral_gap, objective_weights, objective_shift_tv,
    node_class_counts) để trống: 0, None, [], 0.0, []. Lý do hỏng nằm ở SummaryFailed của attempt.
    """
    return _record(
        config,
        provenance=provenance,
        attempt=attempt,
        child_seeds=child_seeds,
        partition_retries=0,
        cpu_threads=cpu_threads,
        spectral_gap=None,
        objective_weights=[],
        objective_shift_tv=0.0,
        node_class_counts=[],
    )
