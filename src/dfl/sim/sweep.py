"""Chạy một nhóm thí nghiệm: YAML thí nghiệm → lưới → run_id, mỗi lượt một tiến trình con mới (module sim;
SPEC §5.2, §5.3, ENG-9; ADR-008).

YAML thí nghiệm (configs/experiments/<tên>.yaml):

    experiment: e1_noniid              # tên chỉ mục results/experiments/<tên>.json
    profile: ../profiles/full.yaml     # cấu hình gốc, đường dẫn tính từ file này
    blocks:                            # ô của nhóm là hợp các ô của mọi khối
      - seeds: [0, 1, 2, 3, 4]
        set: {algorithm: {tau: 20}}    # ghi đè chung cho mọi ô của khối (tùy chọn)
        grid:                          # tích Descartes (tùy chọn); khóa là khóa cấu hình,
          partition: [{scheme: iid}, {scheme: dirichlet, alpha: 0.1}]  # mỗi giá trị trộn như extends
          topology: [{kind: ring}, {kind: fully_connected}]

Trong một khối, seed là vòng ngoài cùng: seed đầu của mọi ô chạy trước, nên xem sớm được toàn lưới.

- Trạng thái một run_id theo các attempt của nó: có attempt ok hay diverged là xong; nếu không thì attempt mới
  nhất quyết định: chưa có summary là bị ngắt (chạy lại), failed là failed (chỉ chạy lại khi có
  --retry-failed).
- Sweep luôn đòi cây sạch, kể cả khi ghi ra --out (CLAUDE.md). Ghi vào results/ thì còn đòi: môi trường khớp
  file khóa, không có ô dữ liệu tổng hợp, chạy từ một worktree liên kết (`git worktree add`) có gốc là `root`,
  và file thí nghiệm cùng profile nằm trong worktree đó (cây sạch nên chúng đã được commit).
- Mỗi lượt là `python -m dfl run` trong một tiến trình con mới (subprocess, không fork; SPEC §5.2), với
  PYTHONPATH=<root>/src và các biến số luồng. Tiến trình con kiểm nó nạp dfl từ `root` và có cùng
  code_version với sweep. Mã thoát của nó (cli.py): 0 xong, 1 lỗi, 3 bị từ chối, 4 run_id đang bị khóa (bỏ
  qua), 5 code khác của sweep. Gặp 5, hay 3 với run_id chưa xong, thì sweep dừng xếp lượt mới.
- Chỉ mục ghi lúc bắt đầu, sau các phép kiểm, mỗi code_version một mục (contracts.ExperimentIndex), qua file
  tạm rồi os.replace. Ghi vào results/ thì giữ khóa flock trên results/experiments/<tên>.lock khi ghi.
- Không nạp torch: tiến trình cha chỉ dựng lưới, tính run_id và đợi tiến trình con.

subprocess.run: https://docs.python.org/3.12/library/subprocess.html#subprocess.run
concurrent.futures.ThreadPoolExecutor: https://docs.python.org/3.12/library/concurrent.futures.html
itertools.product: https://docs.python.org/3.12/library/itertools.html#itertools.product
tempfile.TemporaryDirectory: https://docs.python.org/3.12/library/tempfile.html#tempfile.TemporaryDirectory
PyYAML (safe_load, safe_dump): https://pyyaml.org/wiki/PyYAMLDocumentation
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dfl.config import RunConfig, canonical_json, from_mapping, load_mapping, merge
from dfl.contracts import ExperimentIndex, ExperimentVersion
from dfl.io.manifest import Provenance, is_linked_worktree, provenance_of, repo_root, run_id
from dfl.io.run_logger import FINISHED, attempt_statuses, file_lock

RESULTS_ROOT = Path("results")
EXPERIMENT_KEYS = frozenset({"experiment", "profile", "blocks", "description"})
BLOCK_KEYS = frozenset({"seeds", "set", "grid"})
CONFIG_KEYS = frozenset(field.name for field in dataclasses.fields(RunConfig))
NAME = re.compile(r"[A-Za-z0-9_.-]+")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# Mã thoát của `python -m dfl run` (cli.py).
EXIT_FAILED, EXIT_REFUSED, EXIT_LOCKED, EXIT_CODE_MISMATCH = 1, 3, 4, 5
# Kết quả làm sweep dừng xếp lượt mới.
STOPPING = frozenset({"refused", "code_changed"})


class SweepRefused(Exception):
    """Sweep không được chạy (SPEC §5.3): cây bẩn, hay thiếu điều kiện để ghi vào results/."""


@dataclass(frozen=True, slots=True)
class Cell:
    block: int
    seed: int
    config: RunConfig


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    path: Path
    profile: Path
    cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    total: int
    done: int
    failed: int
    to_run: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    cell: Cell
    run_id: str
    # ok, diverged, failed, crashed (tiến trình con chết), locked (lượt đang chạy ở nơi khác), done (xong ở
    # nơi khác trong lúc sweep chạy), refused, code_changed.
    outcome: str
    returncode: int


@dataclass(frozen=True, slots=True)
class SweepReport:
    experiment: Experiment
    plan: Plan
    outcomes: list[Outcome]
    stopped: bool


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} phải là một mapping YAML, nhận {type(value).__name__}")
    return value


def _check_keys(value: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{where}: khóa {unknown} không có; dùng {sorted(allowed)}")


def _seeds(value: object, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}.seeds phải là danh sách seed không rỗng")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in value) or len(set(value)) != len(
        value
    ):
        raise ValueError(f"{where}.seeds phải là các số nguyên không âm, không trùng, nhận {value}")
    return value


def _overrides(value: Mapping[str, Any], where: str) -> None:
    unknown = sorted(set(value) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{where}: {unknown} không phải khóa cấu hình")
    if "seed" in value:
        raise ValueError(f"{where}: seed lấy từ seeds của khối, không ghi trong set hay grid")


def load_experiment(path: str | Path) -> Experiment:
    """Đọc YAML thí nghiệm và dựng mọi ô (cấu hình đã kiểm); sai định dạng thì raise ValueError."""
    path = Path(path).resolve()
    document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)
    _check_keys(document, EXPERIMENT_KEYS, path.name)
    name = document.get("experiment")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"{path.name}: experiment phải là tên file hợp lệ ([A-Za-z0-9_.-]+), nhận {name!r}")
    if not isinstance(document.get("profile"), str):
        raise ValueError(f"{path.name}: profile phải là đường dẫn tới file cấu hình")
    profile = (path.parent / document["profile"]).resolve()
    base = load_mapping(profile)
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{path.name}: blocks phải là danh sách khối không rỗng")
    cells: list[Cell] = []
    seen: dict[str, str] = {}
    for b, block in enumerate(blocks):
        where = f"{path.name}, khối {b}"
        block = _mapping(block, where)
        _check_keys(block, BLOCK_KEYS, where)
        settings = _mapping(block.get("set", {}), f"{where}.set")
        grid = _mapping(block.get("grid", {}), f"{where}.grid")
        _overrides(settings, f"{where}.set")
        _overrides(grid, f"{where}.grid")
        for axis, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{where}.grid.{axis} phải là danh sách giá trị không rỗng")
        for seed in _seeds(block.get("seeds"), where):
            for combo in itertools.product(*grid.values()):
                mapping = merge(base, settings)
                for axis, value in zip(grid, combo, strict=True):
                    mapping = merge(mapping, {axis: value})
                try:
                    config = from_mapping(merge(mapping, {"seed": seed}))
                except ValueError as err:
                    raise ValueError(f"{where}, seed {seed}: {err}") from err
                key = canonical_json(config)
                if key in seen:
                    raise ValueError(f"{where}, seed {seed}: ô trùng với {seen[key]}")
                seen[key] = f"khối {b}, seed {seed}"
                cells.append(Cell(block=b, seed=seed, config=config))
    return Experiment(name=name, path=path, profile=profile, cells=tuple(cells))


def select(cells: Sequence[Cell], only: Sequence[str]) -> list[Cell]:
    """Các ô có mọi `khóa.con=giá trị` của `only` (giá trị đọc bằng YAML, ví dụ partition.alpha=100.0)."""
    conditions = []
    for item in only:
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"--only cần dạng khóa=giá trị, nhận {item!r}")
        conditions.append((key.split("."), yaml.safe_load(text), key))
    kept = []
    for cell in cells:
        values = dataclasses.asdict(cell.config)
        matches = True
        for path, wanted, key in conditions:
            value: Any = values
            for part in path:
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f"--only: {key} không phải khóa cấu hình")
                value = value[part]
            matches = matches and value == wanted
        if matches:
            kept.append(cell)
    return kept


def cell_run_id(cell: Cell, provenance: Provenance) -> str:
    return run_id(cell.config, provenance.code_version, provenance.machine)


def cell_state(run_dir: Path) -> str:
    """done, failed, interrupted (attempt mới nhất chưa có summary) hay new (chưa có attempt)."""
    statuses = attempt_statuses(run_dir)
    if not statuses:
        return "new"
    if any(status in FINISHED for status in statuses.values()):
        return "done"
    return "interrupted" if statuses[max(statuses)] is None else "failed"


def plan(cells: Sequence[Cell], provenance: Provenance, target: Path, *, retry_failed: bool) -> Plan:
    """Kế hoạch theo trạng thái trong `target`/runs/: lượt nào xong, failed, và lượt nào sẽ chạy."""
    states = [cell_state(target / "runs" / cell_run_id(cell, provenance)) for cell in cells]
    runnable = {"new", "interrupted"} | ({"failed"} if retry_failed else set())
    return Plan(
        total=len(cells),
        done=states.count("done"),
        failed=states.count("failed"),
        to_run=tuple(cell for cell, state in zip(cells, states, strict=True) if state in runnable),
    )


def cell_yaml(config: RunConfig) -> str:
    """Cấu hình của một ô cho tiến trình con; safe_dump viết số thực có dấu chấm (1.0e-05), đọc lại đúng."""
    return yaml.safe_dump(json.loads(canonical_json(config)), sort_keys=True, allow_unicode=True)


def _check(
    experiment: Experiment, cells: Sequence[Cell], provenance: Provenance, root: Path, results: bool
) -> None:
    if provenance.dirty:
        raise SweepRefused(
            "cây làm việc bẩn (src/, configs/ hay requirements.lock có thay đổi chưa commit, hoặc có file "
            "*.py, *.yaml bị ignore); sweep luôn đòi cây sạch, kể cả khi ghi ra --out"
        )
    if not provenance.env_match:
        if results:
            raise SweepRefused("môi trường lệch requirements.lock: " + "; ".join(provenance.mismatches))
        print(
            "cảnh báo: môi trường lệch requirements.lock; manifest sẽ ghi env_match: false", file=sys.stderr
        )
    if not results:
        return
    if any(cell.config.dataset == "synthetic" for cell in cells):
        raise SweepRefused("dữ liệu tổng hợp (profile smoke) không bao giờ được ghi vào results/; dùng --out")
    if not is_linked_worktree(root):
        raise SweepRefused(
            "ghi vào results/ phải chạy từ một worktree liên kết check out commit của lưới (git worktree "
            f"add), với PYTHONPATH=<worktree>/src; {root} không phải worktree như vậy"
        )
    for path in (experiment.path, experiment.profile):
        if not path.is_relative_to(root):
            raise SweepRefused(f"{path} nằm ngoài worktree {root}; file thí nghiệm phải được commit ở đó")


def _write_index(experiment: Experiment, provenance: Provenance, target: Path, lock: bool) -> None:
    path = target / "experiments" / f"{experiment.name}.json"
    entry = ExperimentVersion(
        run_ids=sorted({cell_run_id(cell, provenance) for cell in experiment.cells}),
        git_commit=provenance.git_commit,
        machine=provenance.machine,
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    def update() -> None:
        index: ExperimentIndex
        if path.exists():
            index = json.loads(path.read_text(encoding="utf-8"))
            if index.get("experiment") != experiment.name:
                raise SweepRefused(f"{path} là chỉ mục của nhóm {index.get('experiment')!r}")
        else:
            index = {"experiment": experiment.name, "versions": {}}
        index["versions"][provenance.code_version] = entry
        temporary = path.with_name(path.name + ".tmp")
        text = json.dumps(index, allow_nan=False, sort_keys=True, ensure_ascii=False, indent=2)
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, path)

    path.parent.mkdir(parents=True, exist_ok=True)
    if lock:
        with file_lock(path.with_suffix(".lock"), wait=True):
            update()
    else:
        update()


def classify(returncode: int, run_dir: Path) -> str:
    """Kết quả của một tiến trình con theo mã thoát của `python -m dfl run` (cli.py) và attempt nó để lại."""
    if returncode == 0:
        statuses = attempt_statuses(run_dir)
        latest = statuses[max(statuses)] if statuses else None
        return latest if latest in FINISHED else "failed"
    if returncode == EXIT_FAILED:
        return "failed"
    if returncode == EXIT_LOCKED:
        return "locked"
    if returncode == EXIT_REFUSED:
        return "done" if cell_state(run_dir) == "done" else "refused"
    if returncode == EXIT_CODE_MISMATCH:
        return "code_changed"
    return "crashed"


def _execute(
    cells: Sequence[Cell],
    provenance: Provenance,
    root: Path,
    target: Path,
    out: Path | None,
    data_root: Path,
    workers: int,
) -> tuple[list[Outcome], bool]:
    stop = threading.Event()
    python_path = os.pathsep.join([str(root / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])
    with tempfile.TemporaryDirectory(prefix="dfl-sweep-") as scratch:

        def launch(index: int, cell: Cell) -> Outcome | None:
            if stop.is_set():
                return None
            config_path = Path(scratch) / f"cell-{index}.yaml"
            config_path.write_text(cell_yaml(cell.config), encoding="utf-8")
            command = [sys.executable, "-m", "dfl", "run", "--config", str(config_path)]
            command += ["--data-root", str(data_root), "--expect-root", str(root)]
            command += ["--expect-code-version", provenance.code_version]
            command += ["--out", str(out)] if out is not None else ["--results", str(target)]
            env = {**os.environ, "PYTHONPATH": python_path}
            env.update(dict.fromkeys(THREAD_VARIABLES, str(cell.config.threads)))
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
            rid = cell_run_id(cell, provenance)
            outcome = classify(result.returncode, target / "runs" / rid)
            if outcome in STOPPING:
                stop.set()
            if outcome not in FINISHED and outcome != "done":
                tail = "\n".join(result.stderr.strip().splitlines()[-5:])
                print(f"{rid} (khối {cell.block}, seed {cell.seed}): {outcome}\n{tail}", file=sys.stderr)
            return Outcome(cell=cell, run_id=rid, outcome=outcome, returncode=result.returncode)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(launch, index, cell) for index, cell in enumerate(cells)]
            outcomes = [future.result() for future in futures]
    return [outcome for outcome in outcomes if outcome is not None], stop.is_set()


def run_sweep(
    experiment_path: str | Path,
    *,
    out: Path | None = None,
    results_root: Path = RESULTS_ROOT,
    workers: int = 1,
    dry_run: bool = False,
    retry_failed: bool = False,
    only: Sequence[str] = (),
    data_root: Path = Path("data"),
    provenance: Provenance | None = None,
    root: Path | None = None,
) -> SweepReport:
    """Chạy các ô chưa xong của một nhóm thí nghiệm; `--dry-run` chỉ lập kế hoạch, không ghi gì.

    `root` là gốc của checkout mà tiến trình con nạp dfl (mặc định: checkout của tiến trình này); `provenance`
    mặc định đọc từ `root`. Chỉ test truyền hai tham số này.
    """
    if workers < 1:
        raise ValueError(f"workers phải ≥ 1, nhận {workers}")
    root = (root or repo_root()).resolve()
    provenance = provenance or provenance_of(root)
    experiment = load_experiment(experiment_path)
    cells = select(experiment.cells, only)
    target = (Path(out) if out is not None else Path(results_root)).resolve()
    the_plan = plan(cells, provenance, target, retry_failed=retry_failed)
    if dry_run:
        return SweepReport(experiment=experiment, plan=the_plan, outcomes=[], stopped=False)
    _check(experiment, cells, provenance, root, results=out is None)
    _write_index(experiment, provenance, target, lock=out is None)
    outcomes, stopped = _execute(
        the_plan.to_run,
        provenance,
        root,
        target,
        target if out is not None else None,
        Path(data_root).resolve(),
        workers,
    )
    return SweepReport(experiment=experiment, plan=the_plan, outcomes=outcomes, stopped=stopped)
