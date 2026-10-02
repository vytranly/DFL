"""Test cho src/dfl/io/manifest.py: code_version, run_id, máy, trạng thái git, kiểm môi trường và
ManifestRecord (T11a; SPEC §5.3; ADR-008; G13, G20).
"""

import dataclasses
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import dfl
from dfl import config as cfg
from dfl.contracts import ManifestRecord
from dfl.io import manifest as mf

CPUINFO = (
    "processor\t: 0\nmodel name\t: Test CPU @ 2.10GHz\nflags\t\t: sse avx fpu\n\n"
    "processor\t: 1\nmodel name\t: Test CPU @ 2.10GHz\nflags\t\t: sse avx fpu\n"
)


def make_package(root: Path) -> tuple[Path, Path]:
    """Một gói `dfl` giả trong `root`/src cùng file khóa, như cây của repo."""
    package = root / "src" / "dfl"
    (package / "sim").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "sim" / "engine.py").write_text("X = 1\nY = 2\n")
    (package / "notes.md").write_text("không phải code\n")
    lock = root / "requirements.lock"
    lock.write_text("numpy==2.5.3\ntorch==2.14.0+cpu\n")
    return package, lock


# ---------------------------------------------------------------------------------------------- code_version


def test_code_version_ignores_line_endings_files_outside_the_package_and_non_python_files(
    tmp_path: Path,
) -> None:
    package, lock = make_package(tmp_path)
    before = mf.code_version(package, lock)

    (package / "sim" / "engine.py").write_bytes(b"X = 1\r\nY = 2\r\n")
    (package / "notes.md").write_text("đổi tài liệu\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("tài liệu\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("num_nodes: 20\n")
    lock.write_text("  torch==2.14.0+cpu\n\nnumpy==2.5.3  \n")  # thứ tự, khoảng trắng, dòng trống

    assert mf.code_version(package, lock) == before
    assert re.fullmatch(r"[0-9a-f]{64}", before)


@pytest.mark.parametrize("change", ["edit", "rename", "new-file", "lock"])
def test_code_version_changes_with_any_python_source_or_the_lock_file(tmp_path: Path, change: str) -> None:
    package, lock = make_package(tmp_path)
    before = mf.code_version(package, lock)
    engine = package / "sim" / "engine.py"

    if change == "edit":
        engine.write_text("X = 1\nY = 3\n")
    elif change == "rename":
        engine.rename(package / "sim" / "core.py")
    elif change == "new-file":
        (package / "extra.py").write_text("")
    else:
        lock.write_text("numpy==2.5.4\ntorch==2.14.0+cpu\n")

    assert mf.code_version(package, lock) != before


def test_current_code_version_hashes_the_loaded_dfl_package_and_the_repo_lock() -> None:
    package = Path(dfl.__file__).resolve().parent

    assert mf.current_code_version() == mf.code_version(package, package.parents[1] / "requirements.lock")


def test_missing_lock_file_raises_file_not_found(tmp_path: Path) -> None:
    package, lock = make_package(tmp_path)
    lock.unlink()

    with pytest.raises(FileNotFoundError, match="requirements.lock"):
        mf.code_version(package, lock)


# ------------------------------------------------------------------------------------------------ máy, run_id


def test_machine_is_cpu_name_with_a_hash_of_the_sorted_flags() -> None:
    reordered = CPUINFO.replace("sse avx fpu", "fpu sse avx")
    other_flags = CPUINFO.replace("sse avx fpu", "sse avx fpu avx512f")

    name = mf.machine(CPUINFO)

    assert name.startswith("Test CPU @ 2.10GHz (flags ")
    assert mf.machine(reordered) == name
    assert mf.machine(other_flags) != name


def test_machine_without_cpuinfo_still_names_the_platform() -> None:
    assert mf.machine("") and mf.machine("") == mf.machine("")


def test_machine_reads_proc_cpuinfo_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(CPUINFO)
    monkeypatch.setattr(mf, "CPUINFO", cpuinfo)
    present = mf.machine()
    monkeypatch.setattr(mf, "CPUINFO", tmp_path / "missing")

    assert (present, mf.machine()) == (mf.machine(CPUINFO), mf.machine(""))


def test_run_id_depends_on_config_code_version_and_machine() -> None:
    config = cfg.RunConfig()
    base = mf.run_id(config, "c" * 64, "cpu A")

    assert re.fullmatch(r"[0-9a-f]{16}", base)
    assert mf.run_id(cfg.RunConfig(), "c" * 64, "cpu A") == base
    assert mf.run_id(dataclasses.replace(config, seed=1), "c" * 64, "cpu A") != base
    assert mf.run_id(config, "d" * 64, "cpu A") != base
    assert mf.run_id(config, "c" * 64, "cpu B") != base


# ------------------------------------------------------------------------------------------------ git


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    make_package(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("num_nodes: 20\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.egg-info/\nscratch_*\n")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree_reports_the_head_commit_and_not_dirty(repo: Path) -> None:
    # Bộ đệm bytecode, egg-info, và thay đổi ngoài src/, configs/ không làm cây bẩn.
    (repo / "src" / "dfl" / "__pycache__").mkdir()
    (repo / "src" / "dfl" / "__pycache__" / "engine.py").write_text("")
    (repo / "src" / "dfl.egg-info").mkdir()
    (repo / "src" / "dfl.egg-info" / "x.py").write_text("")
    (repo / "docs.md").write_text("ghi chú\n")

    assert mf.git_state(repo) == (git(repo, "rev-parse", "HEAD"), False)


@pytest.mark.parametrize(
    "path",
    [
        "src/dfl/sim/engine.py",
        "src/dfl/new.py",
        "configs/new.yaml",
        "requirements.lock",
        "src/dfl/scratch_x.py",
        "configs/scratch_local.yaml",
        "configs/scratch_dir/x.yml",
    ],
    ids=[
        "modified",
        "untracked-py",
        "untracked-yaml",
        "lock",
        "ignored-py",
        "ignored-yaml",
        "ignored-dir-yml",
    ],
)
def test_changes_that_reach_code_version_or_configs_make_the_tree_dirty(repo: Path, path: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(target.read_text() + "# đổi\n" if target.exists() else "x = 1\n")

    assert mf.git_state(repo)[1] is True


def test_a_subdirectory_of_a_repo_is_not_treated_as_the_repo(repo: Path) -> None:
    # Từ src/, pathspec "src" và "configs" không trỏ tới đâu: nếu chấp nhận thì cây bẩn bị báo là sạch.
    (repo / "src" / "dfl" / "new.py").write_text("x = 1\n")

    assert mf.git_state(repo / "src") == ("none", True)


def test_a_repo_without_commits_is_dirty_with_no_commit(tmp_path: Path) -> None:
    make_package(tmp_path)
    git(tmp_path, "init", "-q", "-b", "main")

    assert mf.git_state(tmp_path) == ("none", True)


def test_a_machine_without_git_is_dirty_with_no_commit(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(repo / "no-bin"))

    assert mf.git_state(repo) == ("none", True)


def test_a_directory_without_git_is_dirty_with_no_commit(tmp_path: Path) -> None:
    make_package(tmp_path)

    assert mf.git_state(tmp_path) == ("none", True)


# ------------------------------------------------------------------------------------------------ môi trường


def test_environment_matches_when_every_locked_package_has_the_locked_version() -> None:
    # pytest, ruff… đã cài nhưng không có trong file khóa: bị bỏ qua (công cụ dev).
    lock = {"numpy": importlib.metadata.version("numpy"), "PyYAML": importlib.metadata.version("pyyaml")}

    check = mf.check_environment(lock)

    assert (check.env_match, check.mismatches) == (True, ())
    assert check.versions == {"python": sys.version.split()[0], **lock}


def test_a_package_at_another_version_or_missing_breaks_the_match() -> None:
    lock = {"numpy": "0.0.1", "no-such-package-xyz": "1.0"}

    check = mf.check_environment(lock)

    assert check.env_match is False
    assert len(check.mismatches) == 2 and "numpy" in check.mismatches[0]
    assert check.versions["no-such-package-xyz"] == "missing"


def test_read_lock_parses_pinned_lines_and_rejects_anything_else(tmp_path: Path) -> None:
    good = tmp_path / "good.lock"
    good.write_text("PyYAML==6.0.3\n\ntorch==2.14.0+cpu\n")
    bad = tmp_path / "bad.lock"
    bad.write_text("torch>=2.0\n")

    assert mf.read_lock(good) == {"PyYAML": "6.0.3", "torch": "2.14.0+cpu"}
    with pytest.raises(ValueError, match="torch>=2.0"):
        mf.read_lock(bad)


def test_the_repo_lock_matches_this_environment() -> None:
    lock = mf.read_lock(Path(dfl.__file__).resolve().parents[2] / "requirements.lock")

    assert mf.check_environment(lock).mismatches == ()


# ------------------------------------------------------------------------------------------------ manifest


PROVENANCE = mf.Provenance(
    code_version="c" * 64,
    git_commit="a" * 40,
    dirty=False,
    env_match=True,
    versions={"python": "3.12.3"},
    machine="Test CPU (flags 0123456789ab)",
)


def build(config: cfg.RunConfig, **overrides: object) -> ManifestRecord:
    n = config.num_nodes
    arguments: dict[str, object] = {
        "provenance": PROVENANCE,
        "attempt": 1,
        "child_seeds": {"init": 7},
        "partition_retries": 0,
        "cpu_threads": 1,
        "spectral_gap": 0.0326,
        "objective_weights": [1 / n] * n,
        "objective_shift_tv": 0.0,
        "node_class_counts": [[1] * 10] * n,
        **overrides,
    }
    return mf.build_manifest(config, **arguments)


def test_manifest_has_every_field_and_records_the_canonical_config() -> None:
    config = cfg.RunConfig()

    record = build(config)

    assert set(record) == set(ManifestRecord.__required_keys__)
    assert record["config"] == json.loads(cfg.canonical_json(config))
    assert record["run_id"] == mf.run_id(config, PROVENANCE.code_version, PROVENANCE.machine)
    assert (record["seed"], record["attempt"], record["child_seeds"]) == (config.seed, 1, {"init": 7})
    assert (record["code_version"], record["dirty"], record["env_match"]) == ("c" * 64, False, True)
    assert datetime.fromisoformat(record["created_at"]).tzinfo is not None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"objective_weights": [0.5, 0.5]}, "objective_weights"),
        ({"node_class_counts": [[1] * 10]}, "node_class_counts"),
        ({"node_class_counts": [[1] * 9] * 20}, "node_class_counts"),
        ({"attempt": 0}, "attempt"),
    ],
    ids=["weights-length", "counts-rows", "counts-columns", "attempt-zero"],
)
def test_manifest_rejects_per_node_fields_of_the_wrong_shape(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build(cfg.RunConfig(), **overrides)


def test_current_provenance_describes_this_checkout_and_environment() -> None:
    root = Path(dfl.__file__).resolve().parents[2]

    provenance = mf.current_provenance()

    assert provenance.code_version == mf.current_code_version()
    assert (provenance.git_commit, provenance.dirty) == mf.git_state(root)
    assert provenance.machine == mf.machine()
    env = mf.check_environment(mf.read_lock(root / "requirements.lock"))
    assert (provenance.env_match, provenance.versions, provenance.mismatches) == (
        env.env_match,
        env.versions,
        env.mismatches,
    )


def test_reading_provenance_does_not_import_torch() -> None:
    # Sweep tính run_id cho mọi ô trước khi chạy; việc đó không được nạp torch. Kiểm trong tiến trình sạch.
    code = (
        "import sys\n"
        "from dfl.io import manifest, run_logger\n"
        "manifest.current_provenance()\n"
        "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_a_run_that_failed_before_assembly_gets_a_manifest_with_empty_assembly_fields() -> None:
    # Ví dụ vượt 1,000 lần lấy mẫu lại (REQ-02): attempt vẫn có manifest đủ khóa, lý do nằm ở summary.json.
    config = cfg.RunConfig()

    record = mf.unassembled_manifest(
        config, provenance=PROVENANCE, attempt=2, child_seeds={"init": 7}, cpu_threads=1
    )

    assert set(record) == set(ManifestRecord.__required_keys__)
    assert record["run_id"] == mf.run_id(config, PROVENANCE.code_version, PROVENANCE.machine)
    assert (record["objective_weights"], record["node_class_counts"], record["spectral_gap"]) == (
        [],
        [],
        None,
    )
    assert (record["partition_retries"], record["objective_shift_tv"], record["attempt"]) == (0, 0.0, 2)
