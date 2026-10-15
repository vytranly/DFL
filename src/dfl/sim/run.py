"""Lắp và chạy một lượt từ cấu hình (module sim; KH §3.2–§3.6; SPEC §5.2, §5.3; ADR-005, ADR-007, ADR-008).

Thứ tự lắp: dữ liệu → chia (luồng partition) → gán vị trí node (luồng placement) → topology (đồ thị ER từ
luồng graph, matching từ luồng gossip theo round) → mixing → node (w⁰ chung từ luồng init, minibatch từ luồng
batch của từng node) → engine → observer → logger. Mỗi bước chỉ lấy ngẫu nhiên từ luồng của nó, nên cùng seed
thì cùng cách chia, cùng vị trí và cùng w⁰ ở mọi topology và thuật toán (G9, T-INV-5).

- Mọi node bắt đầu từ cùng w⁰ (giả định A3): model của mỗi node được dựng từ một Generator mới của luồng init,
  nên các node có cùng giá trị nhưng bộ nhớ tham số riêng. Model đánh giá được dựng cùng cách.
- π (objective_weights, ADR-005): MH 1/N; uniform tỉ lệ d_j + 1 trên đồ thị tĩnh; size-weighted,
  size-corrected và đường server n_j/n. Với gossip, mỗi W^(t) là ma trận của một matching, đối xứng nên ngẫu
  nhiên kép, nên π = 1/N với cả MH lẫn uniform.
- size-corrected: W là MH, và c_i = n_i/n̄ nhân vào mất mát của node i, tính một lần trước khi huấn luyện.
- Baseline (T11c): B1 và B1-4k là `central`, một node giữ đủ phần train theo thứ tự gốc (không dùng luồng
  partition hay placement), rút minibatch từ luồng riêng B1_BATCH, W = [1], không có message; B2 là
  `local_only` (W = I); B3 là đường server.
- Spectral gap ghi vào manifest là của W tĩnh, kể cả ma trận hiệu dụng của đường server; None với gossip và
  với W = I (local_only), vì khi đó nó không mô tả tốc độ trộn.
- Learning curve đánh giá trên tập con phân tầng của `eval.split` theo `eval.every_steps`. Sau round cuối, mọi
  node và acc(w̄) được đánh giá trên đủ tập (REQ-03) để ghi các trường final_* của SummaryOk.
- timing.jsonl (G21): thời gian huấn luyện và đánh giá đo quanh trainer và evaluator; phần còn lại của khoảng
  giữa hai lần ghi (snapshot, Transport, tổng hợp, các phép kiểm) tính vào combine_s.

torch.set_num_threads phải được gọi trước mọi phép tính eager hay autograd; manifest ghi giá trị đọc lại:
https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html
https://docs.pytorch.org/docs/2.14/generated/torch.get_num_threads.html
functools.partial: https://docs.python.org/3.12/library/functools.html#functools.partial
time.perf_counter: https://docs.python.org/3.12/library/time.html#time.perf_counter
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Protocol

import networkx as nx
import numpy as np
import torch

from dfl import seeding
from dfl.config import SYNTHETIC_DATASETS, PartitionConfig, RunConfig
from dfl.contracts import (
    CommCounters,
    EvalResult,
    Evaluator,
    LocalTrainer,
    MixingPolicy,
    NodeDataset,
    Partitioner,
    PartitionResult,
    RunLogger,
    StateDict,
    SummaryDiverged,
    SummaryFailed,
    SummaryOk,
    TimingRecord,
    TopologySchedule,
)
from dfl.data import datasets as data
from dfl.data.heterogeneity import HeterogeneityStats, summarize
from dfl.data.partition import (
    DirichletPartitioner,
    IIDPartitioner,
    LabelKPartitioner,
    QuantityPartitioner,
    assign_positions,
)
from dfl.io.manifest import (
    Provenance,
    build_manifest,
    current_provenance,
    linked_worktree_of,
    repo_root,
    run_id,
    unassembled_manifest,
)
from dfl.io.run_logger import FINISHED, JsonRunLogger, LockBusy, attempt_statuses, file_lock, next_attempt
from dfl.learning.local_train import SgdTrainer
from dfl.learning.models import build_model
from dfl.metrics.classification import ClassificationEvaluator
from dfl.network import mixing, topology
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.engine import Engine, RunDiverged
from dfl.sim.node import Node
from dfl.sim.observer import Observer, run_rounds


def set_threads(threads: int) -> int:
    """Đặt số luồng của torch cho lượt chạy; trả về giá trị đọc lại để ghi vào manifest."""
    torch.set_num_threads(threads)
    return torch.get_num_threads()


def partitioner(config: PartitionConfig) -> Partitioner:
    """Partitioner của `partition.scheme`; config đã bảo đảm tham số của scheme có mặt."""
    if config.scheme == "iid":
        return IIDPartitioner(config.min_size)
    if config.scheme == "dirichlet":
        return DirichletPartitioner(config.alpha, config.min_size)
    if config.scheme == "label_k":
        return LabelKPartitioner(config.k, config.min_size)
    if config.scheme == "quantity":
        return QuantityPartitioner(config.beta, config.min_size)
    raise ValueError("P2' (shards) chỉ dùng trong describe, không chạy trong một lượt (SPEC QS-11)")


def schedule_for(config: RunConfig) -> TopologySchedule | None:
    """Lịch topology của lượt; None với đường server (B3, E1c), vốn không có đồ thị."""
    n, seed, kind = config.num_nodes, config.seed, config.topology.kind
    if kind == "server":
        return None
    if kind == "ring":
        return topology.ring(n)
    if kind == "fully_connected":
        return topology.fully_connected(n)
    if kind == "star":
        return topology.star(n)
    if kind in ("local_only", "central"):
        return topology.local_only(n)
    if kind == "gossip":
        return topology.gossip(n, partial(seeding.numpy_rng, seed, Stream.GOSSIP))
    # Còn erdos_renyi và er_gossip; config điền p mặc định cho cả hai.
    base = topology.erdos_renyi(n, config.topology.p, seeding.numpy_rng(seed, Stream.GRAPH))
    if kind == "erdos_renyi":
        return base
    return topology.er_gossip(base.graph, partial(seeding.numpy_rng, seed, Stream.GOSSIP_ER))


def _metropolis_hastings(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def _uniform(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.uniform(graph)


# config.mixing → MixingPolicy; size-corrected trộn bằng MH và chỉnh mất mát bằng c_i (ADR-005).
MIXING_POLICIES: dict[str, MixingPolicy] = {
    "metropolis_hastings": _metropolis_hastings,
    "uniform": _uniform,
    "size_weighted": mixing.size_weighted,
    "size_corrected": _metropolis_hastings,
}


def objective_weights(config: RunConfig, schedule: TopologySchedule | None, sizes: np.ndarray) -> np.ndarray:
    """π của mục tiêu mà lượt chạy thực sự tối ưu (ADR-005), N phần tử."""
    if schedule is None or config.mixing in ("size_weighted", "size_corrected"):
        return sizes / sizes.sum()
    if not schedule.is_static:
        return np.full(config.num_nodes, 1.0 / config.num_nodes)
    return mixing.objective_weights(config.mixing, schedule.graph_at(0), sizes)


@dataclass(slots=True)
class PhaseTimer:
    """Thời gian huấn luyện và đánh giá cộng dồn giữa hai lần ghi timing.jsonl (G21)."""

    start: float = field(default_factory=time.perf_counter)
    last: float = 0.0
    train: float = 0.0
    evaluate: float = 0.0

    def restart(self) -> None:
        self.start, self.last, self.train, self.evaluate = time.perf_counter(), 0.0, 0.0, 0.0

    def lap(self, round_number: int) -> TimingRecord:
        wall = time.perf_counter() - self.start
        record = TimingRecord(
            round=round_number,
            wall_time_s=wall,
            train_s=self.train,
            combine_s=max(0.0, wall - self.last - self.train - self.evaluate),
            eval_s=self.evaluate,
            comm_s=0.0,  # Transport trong bộ nhớ không chờ (ADR-010)
        )
        self.last, self.train, self.evaluate = wall, 0.0, 0.0
        return record


@dataclass(frozen=True, slots=True)
class TimedTrainer:
    inner: LocalTrainer
    timer: PhaseTimer

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        start = time.perf_counter()
        try:
            return self.inner(model, data, loss_scale=loss_scale, generator=generator)
        finally:
            self.timer.train += time.perf_counter() - start


@dataclass(frozen=True, slots=True)
class TimedEvaluator:
    inner: Evaluator
    timer: PhaseTimer

    def evaluate(self, states: Sequence[StateDict]) -> EvalResult:
        start = time.perf_counter()
        try:
            return self.inner.evaluate(states)
        finally:
            self.timer.evaluate += time.perf_counter() - start


def initial_model(config: RunConfig) -> torch.nn.Module:
    """Model với w⁰ của lượt: Generator mới của luồng init mỗi lần gọi, nên mọi lần gọi cho cùng w⁰."""
    return build_model(config.model, seeding.torch_generator(config.seed, Stream.INIT))


@dataclass(frozen=True, slots=True, eq=False)
class Assembly:
    """Phần lắp của một lượt mà hai runtime dùng chung, tính tất định từ cấu hình và dữ liệu (ADR-010).

    Ở process runtime, mỗi tiến trình node và tiến trình observer tự tính lại phần này, nên mọi tiến trình có
    cùng cách chia, lịch topology, W, π, c_i và w⁰ mà không phải trao đổi gì.
    """

    config: RunConfig
    datasets: data.Datasets
    # Sau khi gán vị trí: parts[i] là chỉ số (trong phần train của node) của dữ liệu node i.
    partition: PartitionResult
    sizes: np.ndarray
    # None với đường server, vốn không có đồ thị hay mixing.
    schedule: TopologySchedule | None
    policy: MixingPolicy | None
    objective_weights: np.ndarray
    heterogeneity: HeterogeneityStats
    loss_scales: np.ndarray
    spectral_gap: float | None

    def node(self, i: int) -> Node:
        """Node i: w⁰ chung, dữ liệu của chính nó, luồng batch riêng, hệ số c_i."""
        config = self.config
        index = torch.from_numpy(self.partition.parts[i])
        central = config.topology.kind == "central"
        return Node(
            node_id=i,
            model=initial_model(config),
            data=NodeDataset(node_id=i, x=self.datasets.train.x[index], y=self.datasets.train.y[index]),
            generator=(
                seeding.torch_generator(config.seed, Stream.B1_BATCH)
                if central
                else seeding.torch_generator(config.seed, Stream.BATCH, i)
            ),
            loss_scale=float(self.loss_scales[i]),
        )

    def trainer(self) -> SgdTrainer:
        algorithm = self.config.algorithm
        return SgdTrainer(
            tau=algorithm.tau, lr=algorithm.lr, mu=algorithm.mu, batch_size=algorithm.batch_size
        )

    def evaluator(self, images: data.ImageSet) -> ClassificationEvaluator:
        counts = np.array(self.heterogeneity.node_class_counts, dtype=float)
        return ClassificationEvaluator(
            initial_model(self.config),
            images.x,
            images.y,
            label_distributions=counts / counts.sum(axis=1, keepdims=True),
            objective_weights=self.objective_weights,
        )

    def curve_set(self) -> data.ImageSet:
        """Tập con phân tầng cố định cho learning curve (ADR-007)."""
        return data.eval_subset(self.datasets, self.config.eval.split, self.config.eval.subset_size)

    def final_set(self) -> data.ImageSet:
        """Đủ tập đánh giá cho đánh giá cuối (REQ-03)."""
        return self.datasets.split(self.config.eval.split)


def _spectral_gap(
    config: RunConfig, schedule: TopologySchedule | None, policy: MixingPolicy | None, sizes: np.ndarray
) -> float | None:
    n = config.num_nodes
    if n < 2 or config.topology.kind in ("local_only", "central"):
        return None
    if schedule is None:  # đường server: mọi hàng của W hiệu dụng là n_k/n
        return mixing.spectral_gap(np.tile(sizes / sizes.sum(), (n, 1)))
    if not schedule.is_static or policy is None:
        return None
    return mixing.spectral_gap(policy(schedule.graph_at(0), sizes))


def assemble(config: RunConfig, datasets: data.Datasets) -> Assembly:
    """Cách chia, vị trí node, lịch topology, mixing, π, thống kê Non-IID, c_i và spectral gap của lượt."""
    n, seed = config.num_nodes, config.seed
    labels = datasets.train.y.numpy()
    if config.topology.kind == "central":
        placed = PartitionResult(parts=(np.arange(len(labels), dtype=np.int64),), retries=0)
    else:
        raw = partitioner(config.partition)(labels, n, seeding.numpy_rng(seed, Stream.PARTITION))
        placed = assign_positions(raw, seeding.numpy_rng(seed, Stream.PLACEMENT))
    sizes = np.array([len(part) for part in placed.parts], dtype=float)
    schedule = schedule_for(config)
    policy = None if schedule is None else MIXING_POLICIES[config.mixing]
    weights = objective_weights(config, schedule, sizes)
    return Assembly(
        config=config,
        datasets=datasets,
        partition=placed,
        sizes=sizes,
        schedule=schedule,
        policy=policy,
        objective_weights=weights,
        heterogeneity=summarize(labels, placed.parts, objective_weights=weights),
        loss_scales=sizes / sizes.mean() if config.mixing == "size_corrected" else np.ones(n),
        spectral_gap=_spectral_gap(config, schedule, policy, sizes),
    )


@dataclass(frozen=True, slots=True, eq=False)
class RunSetup:
    """Một lượt runtime inprocess đã lắp xong, chưa chạy round nào."""

    config: RunConfig
    datasets: data.Datasets
    partition: PartitionResult
    heterogeneity: HeterogeneityStats
    objective_weights: np.ndarray
    spectral_gap: float | None
    nodes: tuple[Node, ...]
    engine: Engine
    observer: Observer
    final_evaluator: Evaluator
    final_size: int
    timer: PhaseTimer

    def execute(self, logger: RunLogger) -> SummaryOk:
        return execute(self, logger)

    def counters(self) -> CommCounters:
        return self.engine.counters()


def build(config: RunConfig, datasets: data.Datasets) -> RunSetup:
    """Lượt runtime inprocess: N node trong một Engine, cùng Observer và Evaluator đánh giá cuối."""
    if config.runtime != "inprocess":
        raise ValueError(f"runtime {config.runtime}: build chỉ lắp runtime inprocess; xem process_runtime")
    assembly = assemble(config, datasets)
    nodes = tuple(assembly.node(i) for i in range(config.num_nodes))
    timer = PhaseTimer()
    server = assembly.schedule is None
    engine = Engine(
        nodes=nodes,
        trainer=TimedTrainer(assembly.trainer(), timer),
        transport=InMemoryTransport(config.num_nodes, with_server=server),
        schedule=assembly.schedule,
        mixing=assembly.policy,
        server_path=server,
    )
    observer = Observer(
        TimedEvaluator(assembly.evaluator(assembly.curve_set()), timer),
        tau=config.algorithm.tau,
        total_rounds=config.rounds,
        eval_every_steps=config.eval.every_steps,
    )
    final = assembly.final_set()
    return RunSetup(
        config=config,
        datasets=datasets,
        partition=assembly.partition,
        heterogeneity=assembly.heterogeneity,
        objective_weights=assembly.objective_weights,
        spectral_gap=assembly.spectral_gap,
        nodes=nodes,
        engine=engine,
        observer=observer,
        final_evaluator=assembly.evaluator(final),
        final_size=len(final.y),
        timer=timer,
    )


def execute(setup: RunSetup, logger: RunLogger) -> SummaryOk:
    """Chạy mọi round, ghi từng bản ghi đến lịch, rồi đánh giá cuối trên đủ tập và ghi SummaryOk."""
    setup.timer.restart()
    for record in run_rounds(setup.engine, setup.config.rounds, setup.observer):
        logger.log_round(record)
        logger.log_timing(setup.timer.lap(record["round"]))
    final = setup.final_evaluator.evaluate(setup.engine.states())
    counters = setup.engine.counters()
    summary: SummaryOk = {
        "status": "ok",
        "messages_total": counters.messages_cum,
        "bytes_total": counters.bytes_cum,
        "final_eval_split": setup.config.eval.split,
        "final_eval_size": setup.final_size,
        "final_acc_per_node": list(final.acc),
        "final_acc_local_per_node": list(final.acc_local),
        "final_f1_macro_per_node": list(final.f1_macro),
        "final_acc_avg_model": final.acc_avg_model,
    }
    logger.finish(summary)
    return summary


RESULTS_ROOT = Path("results")


class RunRefused(Exception):
    """Lượt không được ghi vào results/ (SPEC §5.3): cây bẩn, môi trường lệch, dữ liệu smoke, thư mục kết quả
    nằm trong worktree liên kết, hay run_id đã xong; hoặc `--out` dẫn vào results/."""


class RunLocked(RunRefused):
    """Một tiến trình khác đang giữ khóa của run_id; lượt này thoát mà không tạo attempt."""


class CodeMismatch(Exception):
    """Code của tiến trình con không phải code của sweep: code_version đã đổi giữa chừng (ADR-008)."""


def run_id_of(config: RunConfig, provenance: Provenance) -> str:
    return run_id(config, provenance.code_version, provenance.machine)


@contextmanager
def results_lock(run_dir: Path) -> Iterator[None]:
    """Giữ khóa flock của run_id (`run_dir`/.lock) từ lúc chọn số attempt tới lúc ghi summary (ADR-008)."""
    try:
        with file_lock(run_dir / ".lock", wait=False):
            yield
    except LockBusy as err:
        raise RunLocked(f"run_id {run_dir.name} đang được một tiến trình khác chạy") from err


def _check_results_allowed(config: RunConfig, provenance: Provenance, results_root: Path) -> None:
    if config.dataset in SYNTHETIC_DATASETS:
        raise RunRefused("dữ liệu tổng hợp (profile smoke) không bao giờ được ghi vào results/; dùng --out")
    if provenance.dirty:
        raise RunRefused(
            "cây làm việc bẩn (src/, configs/ hay requirements.lock có thay đổi chưa commit, hoặc có file "
            "*.py, *.yaml bị ignore); commit trước, hoặc ghi ra --out"
        )
    if not provenance.env_match:
        raise RunRefused(
            "môi trường lệch requirements.lock: " + "; ".join(provenance.mismatches) + "; hoặc ghi ra --out"
        )
    worktree = linked_worktree_of(results_root)
    if worktree is not None:
        raise RunRefused(
            f"thư mục kết quả {results_root.resolve()} nằm trong worktree {worktree}, nên `git worktree "
            "remove` sẽ xóa nó; cho results/ của worktree là symlink tới results/ của repo chính"
        )


def _check_out(out: Path, run_dir: Path, results_root: Path) -> None:
    """`out` bỏ qua mọi quy tắc của results/, nên không được dẫn vào results/, kể cả qua symlink hay `..`
    (REVIEW-C3-P4.1 I1)."""
    results = results_root.resolve()
    if any(path.resolve().is_relative_to(results) for path in (out, run_dir)):
        raise RunRefused(
            f"--out {out} dẫn vào thư mục kết quả {results}, mà --out bỏ qua các quy tắc của results/ (cây "
            "sạch, môi trường, khóa, run_id đã xong); bỏ --out để ghi vào results/, hoặc chọn thư mục khác"
        )


class Driver(Protocol):
    """Một lượt đã lắp, chạy được bởi một runtime: RunSetup (inprocess) hay process_runtime.ProcessRun."""

    partition: PartitionResult
    heterogeneity: HeterogeneityStats
    objective_weights: np.ndarray
    spectral_gap: float | None

    def execute(self, logger: RunLogger) -> SummaryOk: ...

    def counters(self) -> CommCounters: ...


def _driver(
    config: RunConfig, datasets: data.Datasets, provenance: Provenance, data_root: Path, timeout: float
) -> Driver:
    if config.runtime == "inprocess":
        return build(config, datasets)
    from dfl.sim import process_runtime  # process_runtime import module này

    return process_runtime.ProcessRun(
        assemble(config, datasets),
        data_root=data_root,
        code_version=provenance.code_version,
        root=repo_root(),
        timeout=timeout,
    )


def _attempt(
    config: RunConfig, run_dir: Path, provenance: Provenance, data_root: Path, threads: int, timeout: float
) -> Path:
    """Một attempt mới trong `run_dir`: manifest, các round, rồi đúng một summary (ok, diverged, failed)."""
    attempt = next_attempt(run_dir)
    logger = JsonRunLogger(run_dir)
    child_seeds = seeding.child_seeds(config.seed)
    try:
        setup = _driver(
            config, data.load(config.dataset, root=data_root), provenance, Path(data_root), timeout
        )
    except Exception as err:
        logger.start(
            unassembled_manifest(
                config, provenance=provenance, attempt=attempt, child_seeds=child_seeds, cpu_threads=threads
            )
        )
        logger.finish(SummaryFailed(status="failed", error=_describe(err), messages_total=0, bytes_total=0))
        raise
    logger.start(
        build_manifest(
            config,
            provenance=provenance,
            attempt=attempt,
            child_seeds=child_seeds,
            partition_retries=setup.partition.retries,
            cpu_threads=threads,
            spectral_gap=setup.spectral_gap,
            objective_weights=setup.objective_weights.tolist(),
            objective_shift_tv=setup.heterogeneity.objective_shift_tv,
            node_class_counts=setup.heterogeneity.node_class_counts,
        )
    )
    try:
        setup.execute(logger)
    except RunDiverged as err:
        counters = setup.counters()
        logger.finish(
            SummaryDiverged(
                status="diverged",
                diverged_at_round=err.diverged_at_round,
                reason=err.reason,
                messages_total=counters.messages_cum,
                bytes_total=counters.bytes_cum,
            )
        )
    except Exception as err:
        counters = setup.counters()
        logger.finish(
            SummaryFailed(
                status="failed",
                error=_describe(err),
                messages_total=counters.messages_cum,
                bytes_total=counters.bytes_cum,
            )
        )
        raise
    return run_dir / f"attempt-{attempt}"


def _describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def _check_code(
    config: RunConfig, run_dir: Path, provenance: Provenance, expected: str | None, threads: int
) -> None:
    """Sweep truyền code_version của nó; khác thì ghi một attempt failed rồi raise CodeMismatch (ADR-008)."""
    if expected is None or provenance.code_version == expected:
        return
    message = (
        f"code_version đổi giữa chừng: sweep bắt đầu với {expected[:12]}…, tiến trình con có "
        f"{provenance.code_version[:12]}…"
    )
    logger = JsonRunLogger(run_dir)
    logger.start(
        unassembled_manifest(
            config,
            provenance=provenance,
            attempt=next_attempt(run_dir),
            child_seeds=seeding.child_seeds(config.seed),
            cpu_threads=threads,
        )
    )
    logger.finish(
        SummaryFailed(status="failed", error=f"CodeMismatch: {message}", messages_total=0, bytes_total=0)
    )
    raise CodeMismatch(message)


def run(
    config: RunConfig,
    *,
    out: Path | None = None,
    results_root: Path = RESULTS_ROOT,
    data_root: Path = Path("data"),
    provenance: Provenance | None = None,
    expected_code_version: str | None = None,
    process_timeout: float = 600.0,
) -> Path:
    """Chạy một lượt; trả về thư mục attempt.

    Có `out` thì ghi vào `out`/runs/<run_id>/attempt-<k>/ ở mọi trạng thái cây và môi trường (manifest ghi
    `dirty`, `env_match`), miễn là nó không dẫn vào `results_root`. Không có `out` thì ghi vào `results_root`
    (mặc định results/) chỉ khi cây sạch, môi trường khớp file khóa, dữ liệu không phải smoke, thư mục kết
    quả (sau khi resolve) không nằm trong worktree liên kết nào, và run_id chưa có attempt xong; trong lúc
    chạy giữ khóa của run_id. Lượt phân kỳ ghi SummaryDiverged và trả về bình thường; lỗi ghi SummaryFailed
    rồi raise lại. `expected_code_version` do sweep truyền (xem `_check_code`). `process_timeout` là thời
    gian chờ (giây) của mọi phép gloo ở runtime processes. `provenance` chỉ để test truyền vào; mặc định đọc
    từ cây và môi trường hiện tại.
    """
    threads = set_threads(config.threads)
    provenance = provenance or current_provenance()
    if out is not None:
        run_dir = Path(out) / "runs" / run_id_of(config, provenance)
        _check_out(Path(out), run_dir, Path(results_root))
        _check_code(config, run_dir, provenance, expected_code_version, threads)
        return _attempt(config, run_dir, provenance, data_root, threads, process_timeout)
    _check_results_allowed(config, provenance, Path(results_root))
    run_dir = Path(results_root) / "runs" / run_id_of(config, provenance)
    with results_lock(run_dir):
        _check_code(config, run_dir, provenance, expected_code_version, threads)
        finished = sorted(k for k, status in attempt_statuses(run_dir).items() if status in FINISHED)
        if finished:
            raise RunRefused(f"run_id {run_dir.name} đã xong ở attempt-{finished[0]}; không tạo attempt mới")
        return _attempt(config, run_dir, provenance, data_root, threads, process_timeout)
