"""Observer chỉ đọc và vòng chạy có quan sát (module sim; KH §3.4 bước 4, §3.5, §4.5; ADR-007; INV-5, G21).

Observer chỉ đọc những gì engine đưa ra: snapshot w^(t+½) của round (đo drift trước tổng hợp), bản sao model
sau tổng hợp (accuracy, macro-F1, acc^loc, acc^eff, consensus distance) và số đếm của Transport. Nó đánh giá
bằng Evaluator có model cấp phát sẵn và không dùng nguồn ngẫu nhiên nào, nên tắt Observer không đổi quỹ đạo
huấn luyện (T-INV-12). Không có đại lượng nào ở đây quay ngược vào huấn luyện (INV-1).

Lịch đánh giá (ADR-007): sau mỗi `eval_every_steps` bước cục bộ (lưới 100 bước chung cho mọi τ; mốc phải rơi
đúng cuối một round nên `eval_every_steps` phải chia hết cho τ), và luôn ở round cuối; `None` là mỗi round
(profile smoke). Thống kê qua node dùng ddof = 0 (contracts.RoundRecord). statistics.pstdev ra đúng 0 khi mọi
giá trị bằng nhau (numpy.std có thể ra cỡ 1e-17), điều mà SAN-5 đòi trên FC + MH. Đối chiếu docstring của
Python 3.12.3 ("Return the square root of the population variance"):
https://docs.python.org/3.12/library/statistics.html#statistics.pstdev
https://docs.python.org/3.12/library/statistics.html#statistics.fmean
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from statistics import fmean, pstdev

from dfl.contracts import CommCounters, Evaluator, RoundRecord, StateDict
from dfl.metrics.consensus import consensus_distance
from dfl.sim.engine import Engine, RoundResult


class Observer:
    def __init__(
        self, evaluator: Evaluator, *, tau: int, total_rounds: int, eval_every_steps: int | None
    ) -> None:
        if tau < 1 or total_rounds < 1:
            raise ValueError(f"τ và số round phải ≥ 1, nhận τ = {tau}, R = {total_rounds}")
        if eval_every_steps is not None and (eval_every_steps < 1 or eval_every_steps % tau != 0):
            raise ValueError(
                f"eval_every_steps phải là số dương chia hết cho τ = {tau}, để mọi mốc rơi đúng cuối "
                f"một round (ADR-007), nhận {eval_every_steps}"
            )
        self._evaluator = evaluator
        self._tau = tau
        self.total_rounds = total_rounds
        self._every = eval_every_steps

    def is_due(self, round_number: int) -> bool:
        """`round_number` là số round đã tổng hợp xong, như RoundRecord.round."""
        if self._every is None or round_number == self.total_rounds:
            return True
        return (round_number * self._tau) % self._every == 0

    def record(self, result: RoundResult, states: Sequence[StateDict], counters: CommCounters) -> RoundRecord:
        """Bản ghi của round vừa xong: `states` là model sau tổng hợp, `result.snapshots` là trước đó."""
        evaluation = self._evaluator.evaluate(states)
        round_number = result.round_idx + 1
        return RoundRecord(
            round=round_number,
            local_steps=round_number * self._tau,
            acc_mean=fmean(evaluation.acc),
            acc_std=pstdev(evaluation.acc),
            acc_min=min(evaluation.acc),
            acc_max=max(evaluation.acc),
            acc_local_mean=fmean(evaluation.acc_local),
            acc_local_std=pstdev(evaluation.acc_local),
            acc_local_min=min(evaluation.acc_local),
            f1_macro_mean=fmean(evaluation.f1_macro),
            f1_macro_std=pstdev(evaluation.f1_macro),
            f1_macro_min=min(evaluation.f1_macro),
            acc_avg_model=evaluation.acc_avg_model,
            acc_eff_mean=fmean(evaluation.acc_eff),
            drift_pre_combine=consensus_distance([snapshot.state for snapshot in result.snapshots]),
            consensus_distance=consensus_distance(states),
            messages_cum=counters.messages_cum,
            bytes_cum=counters.bytes_cum,
            max_node_bytes_cum=counters.max_node_bytes_cum,
            train_loss_mean=fmean(result.train_losses),
            acc_per_node=list(evaluation.acc),
            acc_local_per_node=list(evaluation.acc_local),
        )


def run_rounds(engine: Engine, total_rounds: int, observer: Observer | None) -> Iterator[RoundRecord]:
    """Chạy round 0 … total_rounds − 1 và sinh RoundRecord ở mỗi round đến lịch.

    Trước round đầu tiên, kiểm hợp các G_t của lượt liên thông (INV-4). Là generator, để người gọi ghi từng
    bản ghi ngay khi có. RunDiverged của engine đi thẳng ra ngoài, trước khi bản ghi của round phân kỳ được
    tạo. `observer=None` chỉ huấn luyện, không đánh giá (T-INV-12).
    """
    if observer is not None and observer.total_rounds != total_rounds:
        raise ValueError(
            f"Observer được dựng cho {observer.total_rounds} round, nhưng lượt chạy {total_rounds}"
        )
    engine.check_schedule(total_rounds)
    for round_idx in range(total_rounds):
        result = engine.run_round(round_idx)
        if observer is not None and observer.is_due(round_idx + 1):
            yield observer.record(result, engine.states(), engine.counters())
