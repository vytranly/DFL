"""Sweep: YAML thí nghiệm → lưới → run_id; chỉ mục theo code_version; bỏ qua lượt đã xong, chạy lại lượt bị
ngắt, `--retry-failed`, `--only`, `--dry-run`; khóa và worktree khi ghi vào results/ (T12; SPEC §5.2, §5.3,
ENG-9; ADR-008).

Provenance của tiến trình cha được truyền vào (cây coi như sạch), để test không phụ thuộc trạng thái git của
cây làm việc. Tiến trình con là `python -m dfl run` thật, nạp code hiện tại qua PYTHONPATH.
"""

import dataclasses
import json
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml

from dfl import cli
from dfl.config import load_config, to_yaml
from dfl.contracts import ExperimentIndex
from dfl.io import manifest
from dfl.sim import sweep

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "profiles" / "smoke.yaml"
# Mỗi lượt con chỉ một round và tập con đánh giá 100 ảnh (đủ 10 lớp), để test nhanh.
TINY = {"local_steps": 2, "eval": {"subset_size": 100, "every_steps": None}}
RING_AND_FC = {"topology": [{"kind": "ring"}, {"kind": "fully_connected"}]}


def write_experiment(
    directory: Path, name: str, blocks: list[dict[str, object]], profile: Path = SMOKE
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"experiment": name, "profile": str(profile), "blocks": blocks}, sort_keys=False)
    )
    return path


@pytest.fixture(scope="module")
def here() -> manifest.Provenance:
    """Provenance của cây hiện tại, coi như sạch: code_version trùng với tiến trình con."""
    return dataclasses.replace(manifest.current_provenance(), dirty=False)


def attempts(root: Path) -> list[Path]:
    return sorted(root.glob("runs/*/attempt-*"))


def status(attempt: Path) -> str | None:
    summary = attempt / "summary.json"
    return json.loads(summary.read_text())["status"] if summary.exists() else None


# --------------------------------------------------------------------------------------------- lưới


def test_blocks_expand_to_the_axis_product_times_their_seeds(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    path = write_experiment(
        tmp_path,
        "grid",
        [
            {"seeds": [0, 1], "set": TINY, "grid": {**RING_AND_FC, "algorithm": [{"tau": 1}, {"tau": 2}]}},
            {"seeds": [3], "set": {**TINY, "mixing": "size_weighted", "topology": {"kind": "server"}}},
        ],
    )

    experiment = sweep.load_experiment(path)
    cells = experiment.cells

    assert experiment.name == "grid" and len(cells) == 2 * 2 * 2 + 1
    # seed ngoài cùng trong một block: seed 0 của mọi ô chạy trước
    assert [(c.seed, c.config.topology.kind, c.config.algorithm.tau) for c in cells[:4]] == [
        (0, "ring", 1),
        (0, "ring", 2),
        (0, "fully_connected", 1),
        (0, "fully_connected", 2),
    ]
    assert cells[-1].config.topology.kind == "server" and cells[-1].config.seed == 3
    assert all(c.config.local_steps == 2 and c.config.algorithm.lr == 0.05 for c in cells)  # set + profile
    assert len({sweep.cell_run_id(c, here) for c in cells}) == len(cells)


def test_only_keeps_the_cells_whose_config_has_that_value(tmp_path: Path) -> None:
    path = write_experiment(tmp_path, "only", [{"seeds": [0, 1], "set": TINY, "grid": RING_AND_FC}])
    experiment = sweep.load_experiment(path)

    kept = sweep.select(experiment.cells, ["topology.kind=ring", "seed=1"])

    assert [(c.config.topology.kind, c.seed) for c in kept] == [("ring", 1)]
    with pytest.raises(ValueError, match="topology.knd"):
        sweep.select(experiment.cells, ["topology.knd=ring"])
    with pytest.raises(ValueError, match="khóa=giá trị"):
        sweep.select(experiment.cells, ["topology.kind"])


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"experiment": "x", "profile": str(SMOKE), "blocks": [], "extra": 1}, "extra"),
        ({"experiment": "x", "profile": str(SMOKE), "blocks": []}, "blocks"),
        ({"experiment": "x", "profile": str(SMOKE), "blocks": [{"set": TINY}]}, "seeds"),
        ({"experiment": "x", "profile": str(SMOKE), "blocks": [{"seeds": [0, 0]}]}, "seeds"),
        (
            {
                "experiment": "x",
                "profile": str(SMOKE),
                "blocks": [{"seeds": [0], "grid": {"topology": "ring"}}],
            },
            "danh sách",
        ),
        (
            {
                "experiment": "x",
                "profile": str(SMOKE),
                "blocks": [{"seeds": [0], "grid": {"topologies": [1]}}],
            },
            "topologies",
        ),
        ({"experiment": "x", "profile": str(SMOKE), "blocks": [{"seeds": [0], "set": {"seed": 3}}]}, "seed"),
        ({"experiment": "x", "profile": str(SMOKE), "blocks": [{"seeds": [0]}, {"seeds": [0]}]}, "trùng"),
        (
            {"experiment": "x", "profile": str(SMOKE), "blocks": [{"seeds": [0], "set": {"num_nodes": 0}}]},
            "num_nodes",
        ),
        ({"experiment": "x/y", "profile": str(SMOKE), "blocks": [{"seeds": [0]}]}, "experiment"),
        ({"experiment": "x", "profile": str(SMOKE), "blocks": [3]}, "mapping"),
        ({"experiment": "x", "profile": 3, "blocks": [{"seeds": [0]}]}, "profile"),
        (
            {"experiment": "x", "profile": str(SMOKE), "blocks": [{"seeds": [0], "grid": {"topology": []}}]},
            "không rỗng",
        ),
    ],
    ids=[
        "unknown-key",
        "no-blocks",
        "no-seeds",
        "duplicate-seeds",
        "axis-not-list",
        "unknown-axis",
        "seed-in-set",
        "duplicate-cells",
        "invalid-cell",
        "bad-name",
        "block-not-mapping",
        "profile-not-path",
        "empty-axis",
    ],
)
def test_malformed_experiments_raise_value_error(
    tmp_path: Path, document: dict[str, object], message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match=message):
        sweep.load_experiment(path)


def test_a_cell_config_survives_the_yaml_given_to_the_child(tmp_path: Path) -> None:
    config = dataclasses.replace(
        load_config(SMOKE), algorithm=dataclasses.replace(load_config(SMOKE).algorithm, lr=1.0e-5)
    )
    path = tmp_path / "cell.yaml"

    path.write_text(to_yaml(config))

    assert load_config(path) == config


# --------------------------------------------------------------------------------------------- kế hoạch


def make_attempt(root: Path, run_id: str, k: int, outcome: str | None) -> None:
    attempt = root / "runs" / run_id / f"attempt-{k}"
    attempt.mkdir(parents=True)
    if outcome is not None:
        (attempt / "summary.json").write_text(json.dumps({"status": outcome}))


def test_plan_skips_finished_reruns_interrupted_and_reruns_failed_only_on_request(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    path = write_experiment(tmp_path, "plan", [{"seeds": [0, 1, 2, 3, 4, 5], "set": TINY}])
    cells = sweep.load_experiment(path).cells
    ids = [sweep.cell_run_id(c, here) for c in cells]
    make_attempt(tmp_path, ids[0], 1, "ok")
    make_attempt(tmp_path, ids[1], 1, "diverged")
    make_attempt(tmp_path, ids[2], 1, "failed")
    make_attempt(tmp_path, ids[3], 1, None)  # bị ngắt
    make_attempt(tmp_path, ids[5], 1, "failed")
    make_attempt(tmp_path, ids[5], 2, None)  # lần chạy lại sau failed cũng bị ngắt

    plan = sweep.plan(cells, here, tmp_path, retry_failed=False)
    retry = sweep.plan(cells, here, tmp_path, retry_failed=True)

    assert (plan.total, plan.done, plan.failed, len(plan.to_run)) == (6, 2, 1, 3)
    assert [ids[cells.index(c)] for c in plan.to_run] == [ids[3], ids[4], ids[5]]
    assert (retry.failed, len(retry.to_run)) == (1, 4)


def test_dry_run_reports_counts_and_writes_nothing(tmp_path: Path, here: manifest.Provenance) -> None:
    path = write_experiment(tmp_path / "exp", "dry", [{"seeds": [0, 1], "set": TINY}])

    report = sweep.run_sweep(path, out=tmp_path / "out", dry_run=True, provenance=here, root=ROOT)

    assert (report.plan.total, report.plan.done, report.plan.failed, len(report.plan.to_run)) == (2, 0, 0, 2)
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------------------------- từ chối


def test_sweep_refuses_a_dirty_tree_even_with_out(tmp_path: Path, here: manifest.Provenance) -> None:
    path = write_experiment(tmp_path, "dirty", [{"seeds": [0], "set": TINY}])

    with pytest.raises(sweep.SweepRefused, match="cây"):
        sweep.run_sweep(
            path, out=tmp_path / "out", provenance=dataclasses.replace(here, dirty=True), root=ROOT
        )


def test_sweep_out_inside_results_is_refused(tmp_path: Path, here: manifest.Provenance) -> None:
    # REVIEW-C3-P4.1 I1: --out bỏ qua quy tắc của results/, nên không được trỏ vào chính results/.
    path = write_experiment(tmp_path / "exp", "inside", [{"seeds": [0], "set": TINY}])
    results = tmp_path / "results"

    with pytest.raises(sweep.SweepRefused, match="--out"):
        sweep.run_sweep(path, out=results, results_root=results, provenance=here, root=ROOT)

    assert not results.exists()


def test_results_need_a_linked_worktree_a_matching_environment_and_real_data(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    path = write_experiment(tmp_path, "res", [{"seeds": [0], "set": {**TINY, "dataset": "mnist"}}])
    synthetic = write_experiment(tmp_path, "syn", [{"seeds": [0], "set": TINY}])
    synthetic_cifar10 = write_experiment(
        tmp_path,
        "syn10",
        [{"seeds": [0], "set": {**TINY, "dataset": "synthetic_cifar10", "model": "cnn_niidbench"}}],
    )
    mismatched = dataclasses.replace(
        here, env_match=False, mismatches=("numpy: file khóa ghi 2.5.3, đã cài 2.5.4",)
    )

    with pytest.raises(sweep.SweepRefused, match="worktree liên kết"):
        sweep.run_sweep(path, results_root=tmp_path / "results", provenance=here, root=ROOT)
    with pytest.raises(sweep.SweepRefused, match="numpy"):
        sweep.run_sweep(path, results_root=tmp_path / "results", provenance=mismatched, root=ROOT)
    for smoke_data in (synthetic, synthetic_cifar10):
        with pytest.raises(sweep.SweepRefused, match="smoke"):
            sweep.run_sweep(smoke_data, results_root=tmp_path / "results", provenance=here, root=ROOT)
    assert not (tmp_path / "results").exists()


# --------------------------------------------------------------------------------------------- chạy, --out


def test_sweeps_are_idempotent_index_their_run_ids_and_do_not_depend_on_workers(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    # Arrange: chỉ mục đã có một mục của code_version khác, phải được giữ nguyên.
    path = write_experiment(tmp_path / "exp", "workers", [{"seeds": [0], "set": TINY, "grid": RING_AND_FC}])
    (tmp_path / "two" / "experiments").mkdir(parents=True)
    old: ExperimentIndex = {
        "experiment": "workers",
        "versions": {
            "f" * 64: {"run_ids": ["0" * 16], "git_commit": "b" * 40, "machine": "cũ", "updated_at": "x"}
        },
    }
    (tmp_path / "two" / "experiments" / "workers.json").write_text(json.dumps(old))

    # Act: hai worker cho cả lưới; một worker cho riêng ô ring (--only); rồi chạy lại cả lưới.
    two = sweep.run_sweep(path, out=tmp_path / "two", workers=2, provenance=here, root=ROOT)
    one = sweep.run_sweep(
        path, out=tmp_path / "one", workers=1, only=["topology.kind=ring"], provenance=here, root=ROOT
    )
    again = sweep.run_sweep(path, out=tmp_path / "two", workers=2, provenance=here, root=ROOT)

    # Assert
    assert [o.outcome for o in two.outcomes] == ["ok", "ok"] and not two.stopped
    (ring,) = attempts(tmp_path / "one")
    twin = tmp_path / "two" / ring.relative_to(tmp_path / "one")
    assert (ring / "rounds.jsonl").read_bytes() == (twin / "rounds.jsonl").read_bytes()
    assert (one.plan.total, len(again.plan.to_run), again.outcomes) == (1, 0, [])
    assert len(attempts(tmp_path / "two")) == 2
    index = json.loads((tmp_path / "two" / "experiments" / "workers.json").read_text())
    expected = sorted(sweep.cell_run_id(cell, here) for cell in sweep.load_experiment(path).cells)
    assert index["versions"]["f" * 64] == old["versions"]["f" * 64]
    assert index["versions"][here.code_version]["run_ids"] == expected


def test_interrupted_attempts_rerun_and_failed_ones_only_with_retry_failed(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    impossible = {**TINY, "partition": {"scheme": "dirichlet", "alpha": 0.5, "min_size": 451}}
    path = write_experiment(
        tmp_path / "exp", "retry", [{"seeds": [0], "set": TINY}, {"seeds": [0], "set": impossible}]
    )
    out = tmp_path / "out"

    first = sweep.run_sweep(path, out=out, workers=2, provenance=here, root=ROOT)
    good = next(a for a in attempts(out) if status(a) == "ok")
    (good / "summary.json").unlink()  # như tiến trình bị ngắt trước khi ghi summary
    second = sweep.run_sweep(path, out=out, provenance=here, root=ROOT)
    third = sweep.run_sweep(path, out=out, retry_failed=True, provenance=here, root=ROOT)

    assert [o.outcome for o in first.outcomes] == ["ok", "failed"]
    assert [o.outcome for o in second.outcomes] == ["ok"] and second.plan.failed == 1
    assert [o.outcome for o in third.outcomes] == ["failed"]
    assert sorted((status(a) for a in attempts(out)), key=lambda s: s or "") == [
        None,
        "failed",
        "failed",
        "ok",
    ]


def test_a_child_with_another_code_version_writes_failed_and_stops_the_sweep(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    path = write_experiment(tmp_path / "exp", "changed", [{"seeds": [0, 1, 2], "set": TINY}])
    stale = dataclasses.replace(here, code_version="0" * 64)  # sweep bắt đầu ở một code_version khác

    report = sweep.run_sweep(path, out=tmp_path / "out", workers=1, provenance=stale, root=ROOT)

    assert report.stopped and [o.outcome for o in report.outcomes] == ["code_changed"]
    (attempt,) = attempts(tmp_path / "out")
    assert status(attempt) == "failed"
    assert "code_version" in json.loads((attempt / "summary.json").read_text())["error"]


# ----------------------------------------------------------------------------------------- results/, worktree


def write_idx(path: Path, array: np.ndarray) -> None:
    path.write_bytes(struct.pack(f">I{array.ndim}I", 0x0800 | array.ndim, *array.shape) + array.tobytes())


@pytest.fixture(scope="module")
def worktree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Repo tạm chứa bản sao code hiện tại, đã commit, cùng một worktree liên kết (nhánh grid) như ở E0a.

    Worktree có MNIST giả (5,400 ảnh train, 1,000 ảnh test) trong data/, vì results/ không nhận dữ liệu smoke.
    """
    base = tmp_path_factory.mktemp("repo")
    repo = base / "main"
    repo.mkdir()
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "configs", repo / "configs")
    for name in ("requirements.lock", ".gitignore"):
        shutil.copy(ROOT / name, repo / name)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "lưới"], cwd=repo, check=True)
    wt = base / "wt"
    subprocess.run([*git, "worktree", "add", "-q", "-b", "grid", str(wt)], cwd=repo, check=True)
    rng = np.random.default_rng(0)
    raw = wt / "data" / "MNIST" / "raw"
    raw.mkdir(parents=True)
    for prefix, n in (("train", 5400), ("t10k", 1000)):
        write_idx(raw / f"{prefix}-images-idx3-ubyte", rng.integers(0, 256, (n, 28, 28), dtype=np.uint8))
        write_idx(raw / f"{prefix}-labels-idx1-ubyte", (np.arange(n) % 10).astype(np.uint8))
    return repo, wt


def commit_experiments(wt: Path, documents: dict[str, list[dict[str, object]]]) -> dict[str, Path]:
    paths = {
        name: write_experiment(
            wt / "configs" / "experiments", name, blocks, wt / "configs" / "profiles" / "smoke.yaml"
        )
        for name, blocks in documents.items()
    }
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "add", "configs"], cwd=wt, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "thẻ thí nghiệm"], cwd=wt, check=True)
    return paths


def test_two_sweeps_sharing_a_run_id_in_results_run_it_once(worktree: tuple[Path, Path]) -> None:
    _, wt = worktree
    cell = {**TINY, "dataset": "mnist"}
    paths = commit_experiments(
        wt,
        {
            "first": [{"seeds": [0], "set": cell}],
            "second": [{"seeds": [0], "set": cell, "grid": RING_AND_FC}],
        },
    )
    # Như quy trình thí nghiệm: results/ của worktree là symlink tới thư mục kết quả nằm ngoài worktree.
    outside = wt.parent / "shared-results"
    outside.mkdir()
    results = wt / "results"
    results.symlink_to(outside, target_is_directory=True)
    reports: dict[str, sweep.SweepReport] = {}

    def go(name: str) -> None:
        reports[name] = sweep.run_sweep(
            paths[name], results_root=results, data_root=wt / "data", workers=2, root=wt
        )

    threads = [threading.Thread(target=go, args=(name,)) for name in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {name: report.stopped for name, report in reports.items()} == {"first": False, "second": False}
    ring = [
        o.outcome
        for report in reports.values()
        for o in report.outcomes
        if o.cell.config.topology.kind == "ring"
    ]
    assert sorted(ring)[-1] == "ok" and sorted(ring)[:-1] in ([], ["done"], ["locked"])  # chạy đúng một nơi
    assert len(attempts(results)) == 2  # ring dùng chung, fully_connected riêng
    assert all(status(a) == "ok" for a in attempts(results))
    for name in paths:
        assert (results / "experiments" / f"{name}.json").exists()

    # Commit chỉ đổi docs/ không làm đổi code_version, nên chạy lại không chạy lượt nào (ENG-9).
    (wt / "docs").mkdir(exist_ok=True)
    (wt / "docs" / "E1.md").write_text("ghi chú\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "add", "docs"], cwd=wt, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "tài liệu"], cwd=wt, check=True)
    rerun = sweep.run_sweep(paths["second"], results_root=results, data_root=wt / "data", root=wt)
    assert (rerun.plan.done, rerun.plan.to_run, rerun.outcomes) == (2, (), [])


def test_results_are_refused_from_the_main_working_tree(worktree: tuple[Path, Path]) -> None:
    # File thí nghiệm và profile đều nằm trong repo, nên chỉ điều kiện worktree liên kết từ chối được.
    repo, _ = worktree
    blocks = [{"seeds": [0], "set": {**TINY, "dataset": "mnist"}}]
    path = write_experiment(repo / "build", "main", blocks, repo / "configs" / "profiles" / "smoke.yaml")

    with pytest.raises(sweep.SweepRefused, match="worktree liên kết"):
        sweep.run_sweep(path, results_root=repo / "results", root=repo)


def test_results_refuse_an_experiment_file_outside_the_worktree(
    worktree: tuple[Path, Path], tmp_path: Path
) -> None:
    _, wt = worktree
    path = write_experiment(tmp_path, "outside", [{"seeds": [0], "set": {**TINY, "dataset": "mnist"}}])

    with pytest.raises(sweep.SweepRefused, match="nằm ngoài worktree"):
        sweep.run_sweep(path, results_root=wt / "results", root=wt)


def test_a_child_that_imports_dfl_from_another_checkout_refuses_to_run(worktree: tuple[Path, Path]) -> None:
    _, wt = worktree

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dfl",
            "run",
            "--config",
            str(SMOKE),
            "--out",
            "build/x",
            "--expect-root",
            str(wt),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 5 and "worktree" in result.stderr


def test_dry_run_on_the_command_line_prints_the_four_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_experiment(tmp_path, "cli", [{"seeds": [0, 1], "set": TINY, "grid": RING_AND_FC}])

    status_code = cli.main(["sweep", "--experiment", str(path), "--out", str(tmp_path / "out"), "--dry-run"])

    assert status_code == 0
    assert "cli: tổng 4 / đã xong 0 / failed 0 / sẽ chạy 4" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def done_experiment(tmp_path: Path, provenance: manifest.Provenance) -> Path:
    """Nhóm một ô đã có attempt ok trong tmp_path/out, nên sweep không phải chạy tiến trình con nào."""
    path = write_experiment(tmp_path / "exp", "done", [{"seeds": [0], "set": TINY}])
    (cell,) = sweep.load_experiment(path).cells
    make_attempt(tmp_path / "out", sweep.cell_run_id(cell, provenance), 1, "ok")
    return path


def test_out_only_warns_about_a_mismatched_environment(
    tmp_path: Path, here: manifest.Provenance, capsys: pytest.CaptureFixture[str]
) -> None:
    mismatched = dataclasses.replace(here, env_match=False, mismatches=("numpy: khác",))
    path = done_experiment(tmp_path, mismatched)

    report = sweep.run_sweep(path, out=tmp_path / "out", provenance=mismatched, root=ROOT)

    assert (report.plan.done, report.outcomes) == (1, [])
    assert "env_match: false" in capsys.readouterr().err


def test_an_index_of_another_experiment_is_not_overwritten(tmp_path: Path, here: manifest.Provenance) -> None:
    path = done_experiment(tmp_path, here)
    (tmp_path / "out" / "experiments").mkdir(parents=True)
    (tmp_path / "out" / "experiments" / "done.json").write_text(
        json.dumps({"experiment": "khác", "versions": {}})
    )

    with pytest.raises(sweep.SweepRefused, match="khác"):
        sweep.run_sweep(path, out=tmp_path / "out", provenance=here, root=ROOT)


def test_workers_must_be_positive(tmp_path: Path, here: manifest.Provenance) -> None:
    with pytest.raises(ValueError, match="workers"):
        sweep.run_sweep(
            done_experiment(tmp_path, here), out=tmp_path / "out", workers=0, provenance=here, root=ROOT
        )


def test_child_exit_codes_map_to_outcomes(tmp_path: Path) -> None:
    make_attempt(tmp_path, "finished", 1, "diverged")
    make_attempt(tmp_path, "pending", 1, None)

    assert sweep.classify(0, tmp_path / "runs" / "finished") == "diverged"
    assert sweep.classify(0, tmp_path / "runs" / "pending") == "failed"  # thoát 0 mà không có summary
    assert sweep.classify(1, tmp_path / "runs" / "pending") == "failed"
    assert sweep.classify(3, tmp_path / "runs" / "finished") == "done"  # xong ở nơi khác trong lúc chạy
    assert sweep.classify(3, tmp_path / "runs" / "pending") == "refused"
    assert sweep.classify(4, tmp_path / "runs" / "pending") == "locked"
    assert sweep.classify(5, tmp_path / "runs" / "pending") == "code_changed"
    assert sweep.classify(-9, tmp_path / "runs" / "pending") == "crashed"  # bị SIGKILL


def test_results_inside_the_worktree_are_refused(worktree: tuple[Path, Path]) -> None:
    # REVIEW-C3-P4.1 I2: kết quả nằm trong worktree sẽ mất khi `git worktree remove`.
    _, wt = worktree
    path = commit_experiments(wt, {"inner": [{"seeds": [0], "set": {**TINY, "dataset": "mnist"}}]})["inner"]

    with pytest.raises(sweep.SweepRefused, match="nằm trong worktree"):
        sweep.run_sweep(path, results_root=wt / "results-inside", data_root=wt / "data", root=wt)

    assert not (wt / "results-inside").exists()
