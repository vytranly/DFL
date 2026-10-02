"""Process runtime: mỗi node một tiến trình hệ điều hành, nói chuyện qua torch.distributed (gloo); tiến trình
chạy lượt kiêm observer (module sim; ADR-010; SPEC §9.2 T-INV-13).

- Tiến trình chạy `python -m dfl run` với `runtime: processes` lắp lượt (sim.run.assemble), ghi manifest, khởi
  chạy N tiến trình con `python -m dfl node --rank i …` (subprocess, interpreter mới, không fork; đúng lệnh mà
  container Docker sẽ dùng, T22c), rồi tự vào process group với rank N làm observer. Nó là nơi duy nhất ghi
  kết quả, nên quy tắc results/, khóa run_id và SummaryDiverged/SummaryFailed giống hệt runtime inprocess.
- Mọi tiến trình tự lắp lại cùng một lượt từ cấu hình (cách chia, lịch, W, π, c_i, w⁰), nên ngoài snapshot
  không phải trao đổi gì. Tiến trình node i chỉ giữ model, dữ liệu, Generator và Transport của chính nó
  (INV-1).
- Một round của node i: Adapt; snapshot; gom trạng thái lần 1; G_t = graph_at(t) và W^(t) tự tính (G18), kiểm
  INV-4 như engine; trao đổi snapshot với láng giềng qua TorchDistTransport (isend/irecv, tag = round); kiểm
  inbox; combine theo id người gửi tăng dần; gom trạng thái lần 2. Hai lần gom (all_gather) ứng đúng hai chỗ
  engine kiểm hữu hạn (sau Adapt và sau tổng hợp), nên khi một node phân kỳ, mọi tiến trình dừng gọn ở cùng
  round với bản inprocess.
- Mỗi lần gom mang [mã, message, byte, tải] của từng node; observer cộng thành CommCounters cùng định nghĩa
  với InMemoryTransport (INV-2). Ở mốc đánh giá, node gửi cho observer state sau tổng hợp, snapshot trước tổng
  hợp, mất mát và thời gian các pha bằng send/recv. Kênh đo này không đi qua Transport nên không bị đếm
  (INV-2, INV-5).
- rounds.jsonl trùng từng byte với bản inprocess (T-INV-13): gloo chép đúng byte của tensor (mã hóa uint8 của
  torchdist), mỗi tiến trình một luồng, mọi phép cộng cùng thứ tự, và observer đánh giá bằng cùng Observer và
  Evaluator trên cùng state.
- Khởi động: mỗi node tạo file sẵn sàng trong thư mục tạm ngay trước khi vào process group; observer đợi đủ N
  file, và nếu một tiến trình con thoát trước đó thì dừng ngay (NodeFailed), không đợi hết thời gian chờ của
  gloo. Rendezvous qua file:// trong cùng thư mục tạm, nên không tranh cổng TCP. Node nào thoát giữa chừng thì
  kết nối gloo đóng, phép gom tiếp theo của observer báo lỗi, và lượt ghi SummaryFailed.

torch.distributed (init_process_group với timeout, all_gather, send, recv, destroy_process_group):
https://docs.pytorch.org/docs/2.14/distributed.html
subprocess.Popen: https://docs.python.org/3.12/library/subprocess.html#subprocess.Popen
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dfl.config import RunConfig, to_yaml
from dfl.contracts import (
    CommCounters,
    LocalTrainer,
    MixingPolicy,
    RunLogger,
    Snapshot,
    SummaryOk,
    TimingRecord,
    TopologySchedule,
)
from dfl.data import datasets as data
from dfl.sim.engine import (
    RoundResult,
    RunDiverged,
    _check_inbox,
    _check_round,
    _first_non_finite,
    check_schedule_union,
    preflight,
)
from dfl.sim.node import Node, combine, make_snapshot
from dfl.sim.observer import Observer, is_due
from dfl.sim.run import Assembly, PhaseTimer, TimedEvaluator, assemble, initial_model, set_threads
from dfl.sim.torchdist import StateLayout, TorchDistTransport, decode_state, encode_state

# Mã trạng thái trong mỗi lần gom; khác 0 là dừng (thứ tự ưu tiên của lý do như engine).
OK, LOSS, PARAMS, COMBINED = 0, 1, 2, 3
REASONS = {
    LOSS: "mất mát huấn luyện của node {} không hữu hạn",
    PARAMS: "tham số của node {} không hữu hạn sau huấn luyện cục bộ",
    COMBINED: "tham số của node {} không hữu hạn sau tổng hợp",
}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# Mất mát, thời gian huấn luyện, giao tiếp, tổng hợp từ lần gửi trước.
META_SIZE = 4
# Sau lần gom cuối, node chỉ còn hủy process group rồi thoát; quá thời gian này thì coi là treo và kill.
EXIT_GRACE_S = 60.0


class NodeFailed(RuntimeError):
    """Một tiến trình node không khởi động được hay thoát giữa chừng; lượt ghi SummaryFailed."""


def _gather(row: list[int], world_size: int) -> list[list[int]]:
    """all_gather một hàng [mã, message, byte, tải] của mỗi tiến trình; mọi tiến trình nhận cùng bảng."""
    mine = torch.tensor(row, dtype=torch.int64)
    rows = [torch.zeros(len(row), dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(rows, mine)
    return [r.tolist() for r in rows]


def _node_row(code: int, transport: TorchDistTransport) -> list[int]:
    counters = transport.counters()
    return [code, counters.messages_cum, counters.bytes_cum, counters.max_node_bytes_cum]


def _stopped(rows: list[list[int]], num_nodes: int) -> bool:
    return any(row[0] != OK for row in rows[:num_nodes])


# -------------------------------------------------------------------------------------------- tiến trình node


def run_node(
    config: RunConfig,
    *,
    rank: int,
    init_method: str,
    data_root: Path,
    ready_dir: Path | None,
    timeout: float,
) -> None:
    """Thân của `python -m dfl node --rank <rank>`: lắp lượt, giữ node `rank`, chạy mọi round."""
    n = config.num_nodes
    if not 0 <= rank < n:
        raise ValueError(f"rank của node phải nằm trong 0..{n - 1}; rank {n} là observer")
    set_threads(config.threads)
    assembly = assemble(config, data.load(config.dataset, root=data_root))
    node = assembly.node(rank)
    trainer = assembly.trainer()
    schedule, policy, sizes = assembly.schedule, assembly.policy, assembly.sizes
    if schedule is None or policy is None:
        raise ValueError("process runtime cần topology có đồ thị (config đã từ chối server và central)")
    check_schedule_union(
        schedule, n, config.rounds
    )  # INV-4; tiến trình chạy lượt cũng đã kiểm trước khi spawn
    del assembly  # chỉ giữ dữ liệu của node này (INV-1)
    layout = StateLayout.of(node.model.state_dict())
    if ready_dir is not None:
        (ready_dir / f"ready-{rank}").touch()
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=n + 1, timeout=timedelta(seconds=timeout)
    )
    try:
        _node_rounds(config, node, trainer, schedule, policy, sizes, layout)
    finally:
        dist.destroy_process_group()


def _node_rounds(
    config: RunConfig,
    node: Node,
    trainer: LocalTrainer,
    schedule: TopologySchedule,
    policy: MixingPolicy,
    sizes: np.ndarray,
    layout: StateLayout,
) -> None:
    rank, n, world = node.node_id, config.num_nodes, config.num_nodes + 1
    transport = TorchDistTransport(rank, n, layout)
    train_s = comm_s = combine_s = 0.0
    for t in range(config.rounds):
        start = time.perf_counter()
        rng_before = torch.get_rng_state()
        loss = float(trainer(node.model, node.data, loss_scale=node.loss_scale, generator=node.generator))
        if not torch.equal(torch.get_rng_state(), rng_before):
            raise ValueError(f"Adapt của node {rank} đã dùng RNG toàn cục của torch (INV-5)")
        snapshot = make_snapshot(rank, t, node.model.state_dict())
        train_s += time.perf_counter() - start
        code = LOSS if not math.isfinite(loss) else PARAMS if _first_non_finite(snapshot.state) else OK
        if _stopped(_gather(_node_row(code, transport), world), n):
            return
        start = time.perf_counter()
        graph = schedule.graph_at(t)
        weights = policy(graph, sizes)
        _check_round(graph, weights, n, static=schedule.is_static)
        neighbours = sorted(graph.neighbors(rank))
        exchange = time.perf_counter()
        transport.expect(t, neighbours)
        for dst in neighbours:
            transport.send(dst, snapshot)
        inbox = transport.inbox(rank, t)
        comm_s += time.perf_counter() - exchange
        _check_inbox(inbox, rank, t)
        combined = combine(snapshot, inbox, weights[rank])
        exchange = time.perf_counter()
        transport.end_round(t)
        comm_s += time.perf_counter() - exchange
        node.model.load_state_dict(combined)
        combine_s += time.perf_counter() - start
        code = COMBINED if _first_non_finite(combined) else OK
        if _stopped(_gather(_node_row(code, transport), world), n):
            return
        if is_due(
            t + 1,
            tau=config.algorithm.tau,
            total_rounds=config.rounds,
            eval_every_steps=config.eval.every_steps,
        ):
            observer = n
            dist.send(encode_state(combined, layout), dst=observer, tag=t)
            dist.send(encode_state(snapshot.state, layout), dst=observer, tag=t)
            dist.send(
                torch.tensor([loss, train_s, comm_s, combine_s], dtype=torch.float64), dst=observer, tag=t
            )
            train_s = comm_s = combine_s = 0.0


# ------------------------------------------------------------------------------------------------ observer


class ProcessRun:
    """Driver của sim.run cho runtime processes: khởi chạy N tiến trình node rồi làm observer (rank N)."""

    def __init__(
        self, assembly: Assembly, *, data_root: Path, code_version: str, root: Path, timeout: float
    ) -> None:
        self.assembly = assembly
        self.partition = assembly.partition
        self.heterogeneity = assembly.heterogeneity
        self.objective_weights = assembly.objective_weights
        self.spectral_gap = assembly.spectral_gap
        self._data_root = Path(data_root).resolve()
        self._code_version = code_version
        self._root = Path(root).resolve()
        self._timeout = timeout
        self._counters = CommCounters(messages_cum=0, bytes_cum=0, max_node_bytes_cum=0)

    def counters(self) -> CommCounters:
        """Số đếm của cả lượt ở lần gom gần nhất (cộng message, byte; lớn nhất tải qua node)."""
        return self._counters

    def execute(self, logger: RunLogger) -> SummaryOk:
        config = self.assembly.config
        # Cùng các phép kiểm trước round đầu như runtime inprocess (INV-4, G3, INV-5), trên các node mà tiến
        # trình này tự lắp: lượt bị engine từ chối thì cũng bị từ chối ở đây, trước khi có tiến trình con nào.
        preflight(
            tuple(self.assembly.node(i) for i in range(config.num_nodes)),
            self.assembly.schedule,
            config.rounds,
        )
        with tempfile.TemporaryDirectory(prefix="dfl-processes-") as scratch_name:
            scratch = Path(scratch_name)
            config_path = scratch / "config.yaml"
            config_path.write_text(to_yaml(config), encoding="utf-8")
            init_method = f"file://{scratch / 'rendezvous'}"
            children = [self._spawn(i, config_path, init_method, scratch) for i in range(config.num_nodes)]
            try:
                self._wait_ready(children, scratch)
                dist.init_process_group(
                    "gloo",
                    init_method=init_method,
                    rank=config.num_nodes,
                    world_size=config.num_nodes + 1,
                    timeout=timedelta(seconds=self._timeout),
                )
                try:
                    summary = self._observe(logger)
                except RunDiverged:
                    self._settle(children, scratch)
                    raise
                except RuntimeError as err:
                    if isinstance(err, NodeFailed):
                        raise
                    raise NodeFailed(
                        f"gloo báo lỗi giữa lượt: {err}; {self._exits(children, scratch)}"
                    ) from err
                finally:
                    dist.destroy_process_group()
                self._settle(children, scratch)
                logger.finish(summary)
                return summary
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait()

    def _spawn(
        self, rank: int, config_path: Path, init_method: str, scratch: Path
    ) -> subprocess.Popen[bytes]:
        config = self.assembly.config
        command = [sys.executable, "-m", "dfl", "node", "--rank", str(rank)]
        command += ["--world-size", str(config.num_nodes + 1), "--init-method", init_method]
        command += ["--config", str(config_path), "--data-root", str(self._data_root)]
        command += ["--ready-dir", str(scratch), "--expect-code-version", self._code_version]
        command += ["--timeout", str(self._timeout)]
        python_path = os.pathsep.join(
            [str(self._root / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
        env = {
            **os.environ,
            "PYTHONPATH": python_path,
            **dict.fromkeys(THREAD_VARIABLES, str(config.threads)),
        }
        with (scratch / f"node-{rank}.log").open("wb") as log:
            return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    def _exits(self, children: list[subprocess.Popen[bytes]], scratch: Path) -> str:
        """Mã thoát và vài dòng log cuối của các node đã thoát lỗi, để ghi vào SummaryFailed."""
        parts = []
        for rank, child in enumerate(children):
            code = child.poll()
            if code not in (None, 0):
                tail = (scratch / f"node-{rank}.log").read_text(errors="replace").strip().splitlines()[-3:]
                parts.append(f"node {rank} thoát mã {code}: " + " | ".join(tail))
        return "; ".join(parts) or "chưa node nào thoát lỗi"

    def _wait_ready(self, children: list[subprocess.Popen[bytes]], scratch: Path) -> None:
        deadline = time.monotonic() + self._timeout
        while not all((scratch / f"ready-{rank}").exists() for rank in range(len(children))):
            if any(child.poll() is not None for child in children):
                raise NodeFailed(f"node thoát trước khi vào process group; {self._exits(children, scratch)}")
            if time.monotonic() > deadline:
                raise NodeFailed(f"node chưa sẵn sàng sau {self._timeout} s")
            time.sleep(0.05)

    def _settle(self, children: list[subprocess.Popen[bytes]], scratch: Path) -> None:
        """Đợi node thoát sau lần gom cuối. Lúc này mọi round đã được quan sát, nên node chậm thoát hay thoát
        lỗi chỉ là cảnh báo, không đổi kết quả đã có (ok hay diverged); node còn sống bị kill ở `execute`."""
        try:
            self._wait_exit(children, scratch)
        except NodeFailed as err:
            print(f"cảnh báo: {err}", file=sys.stderr)

    def _wait_exit(self, children: list[subprocess.Popen[bytes]], scratch: Path) -> None:
        deadline = time.monotonic() + min(self._timeout, EXIT_GRACE_S)
        for child in children:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as err:
                raise NodeFailed(
                    f"node không thoát trong {EXIT_GRACE_S:.0f} s sau lần gom cuối (treo)"
                ) from err
        if any(child.returncode != 0 for child in children):
            raise NodeFailed(f"node thoát lỗi sau lượt chạy; {self._exits(children, scratch)}")

    def _counters_of(self, rows: list[list[int]]) -> CommCounters:
        nodes = rows[: self.assembly.config.num_nodes]
        return CommCounters(
            messages_cum=sum(row[1] for row in nodes),
            bytes_cum=sum(row[2] for row in nodes),
            max_node_bytes_cum=max(row[3] for row in nodes),
        )

    def _check(self, rows: list[list[int]], round_idx: int) -> None:
        self._counters = self._counters_of(rows)
        codes = [row[0] for row in rows[: self.assembly.config.num_nodes]]
        for code, reason in REASONS.items():
            if code in codes:
                raise RunDiverged(round_idx, reason.format(codes.index(code)))

    def _observe(self, logger: RunLogger) -> SummaryOk:
        """Quan sát mọi round và đánh giá cuối; trả SummaryOk để `execute` ghi sau khi node đã thoát."""
        assembly = self.assembly
        config = assembly.config
        n, world = config.num_nodes, config.num_nodes + 1
        layout = StateLayout.of(initial_model(config).state_dict())
        timer = PhaseTimer()
        observer = Observer(
            TimedEvaluator(assembly.evaluator(assembly.curve_set()), timer),
            tau=config.algorithm.tau,
            total_rounds=config.rounds,
            eval_every_steps=config.eval.every_steps,
        )
        final_set = assembly.final_set()
        final_evaluator = assembly.evaluator(final_set)
        idle = [OK, 0, 0, 0]
        states: list[dict[str, torch.Tensor]] = []
        timer.restart()
        for t in range(config.rounds):
            self._check(_gather(idle, world), t)
            self._check(_gather(idle, world), t)
            if not observer.is_due(t + 1):
                continue
            states, snapshots, metas = [], [], []
            for i in range(n):
                states.append(_receive_state(i, t, layout))
                snapshots.append(
                    Snapshot(sender=i, round_idx=t, state=_receive_state(i, t, layout), nbytes=layout.nbytes)
                )
                meta = torch.empty(META_SIZE, dtype=torch.float64)
                dist.recv(meta, src=i, tag=t)
                metas.append(meta.tolist())
            result = RoundResult(
                round_idx=t,
                weights=np.zeros((0, 0)),
                train_losses=tuple(meta[0] for meta in metas),
                snapshots=tuple(snapshots),
            )
            record = observer.record(result, states, self._counters)
            logger.log_round(record)
            lap = timer.lap(t + 1)
            logger.log_timing(
                TimingRecord(
                    round=t + 1,
                    wall_time_s=lap["wall_time_s"],
                    train_s=max(meta[1] for meta in metas),
                    combine_s=max(meta[3] for meta in metas),
                    eval_s=lap["eval_s"],
                    comm_s=max(meta[2] for meta in metas),
                )
            )
        final = final_evaluator.evaluate(states)
        summary: SummaryOk = {
            "status": "ok",
            "messages_total": self._counters.messages_cum,
            "bytes_total": self._counters.bytes_cum,
            "final_eval_split": config.eval.split,
            "final_eval_size": len(final_set.y),
            "final_acc_per_node": list(final.acc),
            "final_acc_local_per_node": list(final.acc_local),
            "final_f1_macro_per_node": list(final.f1_macro),
            "final_acc_avg_model": final.acc_avg_model,
        }
        return summary


def _receive_state(sender: int, round_idx: int, layout: StateLayout) -> dict[str, torch.Tensor]:
    buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
    dist.recv(buffer, src=sender, tag=round_idx)
    return decode_state(buffer, layout)
