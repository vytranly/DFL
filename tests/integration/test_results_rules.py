"""Quy tắc ghi results/, phân kỳ và lỗi của `sim.run` (T11d; SPEC §5.3, ENG-10; ADR-008).

Provenance (cây sạch, môi trường khớp) được truyền vào thay vì đọc từ repo thật, để kết quả test không phụ
thuộc trạng thái git của cây làm việc lúc chạy test.
"""

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from dfl.config import AlgorithmConfig, EvalConfig, PartitionConfig, RunConfig, load_config
from dfl.io.manifest import Provenance
from dfl.learning.local_train import SgdTrainer
from dfl.sim import run

ROOT = Path(__file__).resolve().parents[2]
# Profile smoke rút gọn còn 2 round và tập con đánh giá 100 ảnh (vẫn đủ 10 lớp), để các test chạy nhanh;
# test e2e vẫn chạy đúng profile smoke.
SMOKE = dataclasses.replace(
    load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
    local_steps=4,
    eval=EvalConfig(subset_size=100, every_steps=None),
)
CLEAN = Provenance(
    code_version="c" * 64,
    git_commit="a" * 40,
    dirty=False,
    env_match=True,
    versions={"python": "3.12.3"},
    machine="Test CPU (flags 0123456789ab)",
)


def mnist_like(config: RunConfig = SMOKE) -> RunConfig:
    """Cấu hình được phép vào results/: dữ liệu tổng hợp (profile smoke) thì không bao giờ (SPEC §5.3)."""
    return dataclasses.replace(config, dataset="mnist")


@pytest.fixture
def synthetic_as_mnist(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test không có MNIST thật (ENG-8): bộ nạp trả dữ liệu tổng hợp cho cả dataset mnist.
    from dfl.data import datasets

    synthetic = datasets.load("synthetic")
    monkeypatch.setattr(datasets, "load", lambda name, *, root=None: synthetic)


def attempts(root: Path) -> list[Path]:
    return sorted(root.glob("runs/*/attempt-*"))


def summary_of(attempt: Path) -> dict[str, object]:
    return json.loads((attempt / "summary.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ results/


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_a_clean_matching_run_is_written_to_results(tmp_path: Path) -> None:
    attempt = run.run(mnist_like(), results_root=tmp_path, provenance=CLEAN)

    assert attempt.parent.parent == tmp_path / "runs"
    assert summary_of(attempt)["status"] == "ok"


@pytest.mark.usefixtures("synthetic_as_mnist")
@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        (dataclasses.replace(CLEAN, dirty=True), "cây"),
        (
            dataclasses.replace(
                CLEAN, env_match=False, mismatches=("numpy: file khóa ghi 2.5.3, đã cài 2.5.4",)
            ),
            "numpy",
        ),
    ],
    ids=["dirty-tree", "env-mismatch"],
)
def test_results_refuses_a_dirty_tree_or_a_mismatched_environment(
    tmp_path: Path, provenance: Provenance, message: str
) -> None:
    with pytest.raises(run.RunRefused, match=message):
        run.run(mnist_like(), results_root=tmp_path, provenance=provenance)

    assert attempts(tmp_path) == []


def test_results_never_takes_the_synthetic_smoke_data(tmp_path: Path) -> None:
    with pytest.raises(run.RunRefused, match="smoke"):
        run.run(SMOKE, results_root=tmp_path, provenance=CLEAN)

    assert attempts(tmp_path) == []


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_results_refuses_a_run_id_that_already_finished(tmp_path: Path) -> None:
    run.run(mnist_like(), results_root=tmp_path, provenance=CLEAN)

    with pytest.raises(run.RunRefused, match="đã xong"):
        run.run(mnist_like(), results_root=tmp_path, provenance=CLEAN)

    assert len(attempts(tmp_path)) == 1


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_a_diverged_attempt_also_finishes_its_run_id(tmp_path: Path) -> None:
    exploding = mnist_like(dataclasses.replace(SMOKE, algorithm=AlgorithmConfig(tau=2, lr=1.0e4)))
    attempt = run.run(exploding, results_root=tmp_path, provenance=CLEAN)
    assert summary_of(attempt)["status"] == "diverged"

    with pytest.raises(run.RunRefused, match="đã xong"):
        run.run(exploding, results_root=tmp_path, provenance=CLEAN)


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_a_failed_attempt_can_be_retried_by_hand_as_a_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `run` là lệnh tay, nên được chạy lại sau attempt failed; sweep thì cần --retry-failed (T12).
    original = SgdTrainer.__call__

    def broken(self: SgdTrainer, *args: object, **kwargs: object) -> float:
        raise MemoryError("thiếu RAM (giả)")

    monkeypatch.setattr(SgdTrainer, "__call__", broken)
    with pytest.raises(MemoryError):
        run.run(mnist_like(), results_root=tmp_path, provenance=CLEAN)
    monkeypatch.setattr(SgdTrainer, "__call__", original)

    second = run.run(mnist_like(), results_root=tmp_path, provenance=CLEAN)

    assert second.name == "attempt-2" and summary_of(second)["status"] == "ok"
    assert summary_of(second.parent / "attempt-1")["status"] == "failed"


@pytest.mark.parametrize("below", ["", "nested"], ids=["same-directory", "inside"])
def test_out_inside_results_is_refused_so_the_results_rules_cannot_be_bypassed(
    tmp_path: Path, below: str
) -> None:
    # REVIEW-C3-P4.1 I1: --out bỏ qua quy tắc của results/, nên không được trỏ vào chính results/.
    results = tmp_path / "results"

    with pytest.raises(run.RunRefused, match="--out"):
        run.run(
            SMOKE,
            out=results / below,
            results_root=results,
            provenance=dataclasses.replace(CLEAN, dirty=True),
        )

    assert list(tmp_path.rglob("attempt-*")) == []


def test_out_whose_runs_directory_links_into_results_is_refused(tmp_path: Path) -> None:
    # Attempt nằm ở out/runs/<run_id>, nên phép kiểm xét cả đường dẫn đó sau khi resolve, không chỉ out.
    results = tmp_path / "results"
    (results / "runs").mkdir(parents=True)
    out = tmp_path / "build"
    out.mkdir()
    (out / "runs").symlink_to(results / "runs", target_is_directory=True)

    with pytest.raises(run.RunRefused, match="--out"):
        run.run(SMOKE, out=out, results_root=results, provenance=dataclasses.replace(CLEAN, dirty=True))

    assert list(results.rglob("attempt-*")) == []


@pytest.fixture
def linked_worktree(tmp_path: Path) -> Path:
    """Worktree liên kết (`git worktree add -d`, như SPEC §5.3) của một repo tạm có results/ bị ignore."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    repo = tmp_path / "main"
    repo.mkdir()
    (repo / ".gitignore").write_text("results/\n", encoding="utf-8")
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=repo, check=True)
    subprocess.run([*git, "worktree", "add", "-q", "-d", str(tmp_path / "wt")], cwd=repo, check=True)
    return tmp_path / "wt"


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_results_inside_a_linked_worktree_are_refused(linked_worktree: Path) -> None:
    # REVIEW-C3-P4.1 I2: `git worktree remove` xóa cả results/ bị ignore trong worktree.
    with pytest.raises(run.RunRefused, match="nằm trong worktree"):
        run.run(mnist_like(), results_root=linked_worktree / "results", provenance=CLEAN)

    assert not (linked_worktree / "results").exists()


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_results_symlinked_from_a_linked_worktree_to_the_main_repo_are_accepted(
    tmp_path: Path, linked_worktree: Path
) -> None:
    # Quy trình ở tasks/todo.md: results/ của worktree là symlink tới results/ của repo chính.
    shared = tmp_path / "main" / "results"
    shared.mkdir()
    (linked_worktree / "results").symlink_to(shared, target_is_directory=True)

    attempt = run.run(mnist_like(), results_root=linked_worktree / "results", provenance=CLEAN)

    assert attempt.resolve().is_relative_to(shared.resolve())
    assert summary_of(attempt)["status"] == "ok"


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_out_runs_despite_a_dirty_tree_and_records_it(tmp_path: Path) -> None:
    provenance = dataclasses.replace(CLEAN, dirty=True, env_match=False)

    attempt = run.run(SMOKE, out=tmp_path, provenance=provenance)

    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["dirty"], manifest["env_match"]) == (True, False)


@pytest.mark.usefixtures("synthetic_as_mnist")
def test_a_second_process_on_a_locked_run_id_leaves_without_a_new_attempt(tmp_path: Path) -> None:
    # Tiến trình thứ nhất giữ khóa flock của run_id; lượt thứ hai thấy khóa và thoát (SPEC §5.3).
    run_dir = tmp_path / "runs" / run.run_id_of(mnist_like(), CLEAN)
    run_dir.mkdir(parents=True)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time\n"
            f"fd = os.open({str(run_dir / '.lock')!r}, os.O_RDWR | os.O_CREAT)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"

        with pytest.raises(run.RunLocked):
            run.run(mnist_like(), results_root=tmp_path, provenance=CLEAN)
    finally:
        holder.kill()
        holder.wait()

    assert attempts(tmp_path) == []


# ---------------------------------------------------------------------------------------------- phân kỳ, lỗi


def test_a_diverging_run_writes_summary_diverged_and_no_record_of_that_round(tmp_path: Path) -> None:
    exploding = dataclasses.replace(SMOKE, algorithm=AlgorithmConfig(tau=2, lr=1.0e4))

    attempt = run.run(exploding, out=tmp_path, provenance=CLEAN)

    summary = summary_of(attempt)
    lines = (attempt / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["status"] == "diverged" and summary["reason"]
    assert [json.loads(line)["round"] for line in lines] == list(range(1, summary["diverged_at_round"]))
    assert "NaN" not in (attempt / "rounds.jsonl").read_text() and "Infinity" not in json.dumps(summary)
    # Phân kỳ ở Adapt thì round đó chưa gửi gì; ở w^(t+1) thì đã gửi đủ 8 message của ring N = 4.
    rounds_done = summary["diverged_at_round"] - 1
    assert summary["messages_total"] in (8 * rounds_done, 8 * (rounds_done + 1))


def test_an_exception_during_the_run_writes_summary_failed_and_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    original = SgdTrainer.__call__

    def flaky(self: SgdTrainer, *args: object, **kwargs: object) -> float:
        calls["n"] += 1
        if calls["n"] == 5:  # round thứ hai của 4 node
            raise RuntimeError("đĩa đầy (giả)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SgdTrainer, "__call__", flaky)

    with pytest.raises(RuntimeError, match="đĩa đầy"):
        run.run(SMOKE, out=tmp_path, provenance=CLEAN)

    (attempt,) = attempts(tmp_path)
    summary = summary_of(attempt)
    assert summary["status"] == "failed" and "đĩa đầy" in summary["error"]
    assert summary["messages_total"] == 8  # một round xong, 8 message mỗi round
    assert all(math.isfinite(json.loads(line)["acc_mean"]) for line in (attempt / "rounds.jsonl").open())


def test_a_partition_that_cannot_be_made_fails_before_training(tmp_path: Path) -> None:
    # Vượt 1,000 lần lấy mẫu lại thì lượt ghi failed trước khi huấn luyện (SPEC REQ-02). 4 node × 451 ảnh
    # cần ít nhất 1,804 ảnh, mà phần train tổng hợp chỉ có 1,800, nên không lần thử nào đạt min_size.
    partition = PartitionConfig(scheme="dirichlet", alpha=0.5, min_size=451)
    impossible = dataclasses.replace(SMOKE, partition=partition)

    with pytest.raises(ValueError, match="1,000"):
        run.run(impossible, out=tmp_path, provenance=CLEAN)

    (attempt,) = attempts(tmp_path)
    assert summary_of(attempt)["status"] == "failed" and "1,000" in summary_of(attempt)["error"]
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["node_class_counts"], manifest["objective_weights"]) == ([], [])
    assert (attempt / "rounds.jsonl").read_text() == ""
