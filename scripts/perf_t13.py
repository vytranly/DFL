"""Đo hiệu năng cho T13 (docs/PERF.md): lõi rảnh, thời gian một bước, thời gian một lần đánh giá.

Nằm ngoài src/ nên không đổi code_version. Mỗi lệnh in một bảng Markdown để dán vào docs/PERF.md, kèm tải máy
lúc đo. Chạy từ gốc repo:
    .venv/bin/python scripts/perf_t13.py cores --seconds 60
    .venv/bin/python scripts/perf_t13.py step
    .venv/bin/python scripts/perf_t13.py eval
    python scripts/perf_t13.py laptop          # QS-8, trên laptop demo (mọi hệ điều hành)

/proc/stat (cột idle, iowait tính bằng jiffies): https://docs.kernel.org/filesystems/proc.html
os.getloadavg: https://docs.python.org/3.12/library/os.html#os.getloadavg
time.perf_counter: https://docs.python.org/3.12/library/time.html#time.perf_counter
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import torch

from dfl import seeding
from dfl.config import load_config
from dfl.contracts import NodeDataset
from dfl.data import datasets
from dfl.learning.augment import CropFlip
from dfl.learning.local_train import SgdTrainer
from dfl.learning.models import build_model
from dfl.seeding import Stream
from dfl.sim.run import assemble, initial_model

STEPS = 50
REPEATS = 5


def _cpu_times() -> tuple[int, int]:
    fields = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = fields[3] + fields[4]  # idle + iowait
    return idle, sum(fields)


def cores(seconds: float) -> None:
    """Số lõi rảnh trung bình = tỉ lệ idle × số lõi, đo qua /proc/stat trong `seconds` giây."""
    idle0, total0 = _cpu_times()
    load0 = os.getloadavg()
    time.sleep(seconds)
    idle1, total1 = _cpu_times()
    free = (idle1 - idle0) / (total1 - total0) * os.cpu_count()
    stamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"| {stamp} | {seconds:.0f} s | {os.cpu_count()} | {free:.2f} | {load0[0]:.1f} |")


def _step_ms(dataset: str, model: str, threads: int, augment: bool) -> float:
    torch.set_num_threads(threads)
    data = datasets.load(dataset, with_test=False)
    node = NodeDataset(node_id=0, x=data.train.x[:2750], y=data.train.y[:2750])
    trainer = SgdTrainer(tau=STEPS, lr=0.01, augment=CropFlip(fill=data.black()) if augment else None)
    net = build_model(model, seeding.torch_generator(0, Stream.INIT))
    generator = seeding.torch_generator(0, Stream.BATCH, 0)
    trainer(net, node, loss_scale=1.0, generator=generator)  # làm nóng
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        trainer(net, node, loss_scale=1.0, generator=generator)
        times.append((time.perf_counter() - start) / STEPS * 1000)
    return statistics.median(times)


def step() -> None:
    """ms một bước (B = 32), trung vị của 5 × 50 bước; dữ liệu tổng hợp cùng shape (thời gian không phụ thuộc
    giá trị điểm ảnh)."""
    cases = [("synthetic", m, False) for m in ("logreg", "smallcnn", "mlp2nn", "cnn_mcmahan")]
    cases.append(("synthetic_cifar10", "cnn_niidbench", True))
    print("| Model | 1 luồng | 2 luồng | 4 luồng | 8 luồng | tải lúc đo |")
    print("|---|---|---|---|---|---|")
    for dataset, model, augment in cases:
        cells = [f"{_step_ms(dataset, model, k, augment):.2f}" for k in (1, 2, 4, 8)]
        print(f"| {model} | " + " | ".join(cells) + f" | {os.getloadavg()[0]:.1f} |")


def evaluation() -> None:
    """Giây cho một mốc đánh giá (20 node + model trung bình) trên tập con 2,000 ảnh test, và cho đánh giá
    cuối trên đủ 10,000 ảnh; MNIST thật, SmallCNN, 1 luồng."""
    torch.set_num_threads(1)
    config = load_config("configs/profiles/fast.yaml")
    assembly = assemble(config, datasets.load("mnist"))
    states = [initial_model(config).state_dict() for _ in range(config.num_nodes)]
    for name, images in (
        ("mốc (2,000 ảnh)", assembly.curve_set()),
        ("cuối (10,000 ảnh)", assembly.final_set()),
    ):
        evaluator = assembly.evaluator(images)
        evaluator.evaluate(states)  # làm nóng
        times = []
        for _ in range(3):
            start = time.perf_counter()
            evaluator.evaluate(states)
            times.append(time.perf_counter() - start)
        print(f"| {name} | {statistics.median(times):.2f} s | {os.getloadavg()[0]:.1f} |")


# QS-8: demo gồm 2 lượt (ring và FC, N = 20) cộng vẽ hình, tổng ≤ 5 phút. Phần cố định ước tính cho khởi động
# Python/torch, nạp MNIST và vẽ hình; T26 đo lại trên laptop.
DEMO_BUDGET_S = 300.0
DEMO_FIXED_S = 60.0


def laptop() -> None:
    """QS-8: in máy, phiên bản, ms một bước SmallCNN (1 luồng), giây một mốc đánh giá, và số bước demo lớn
    nhất (bội của 100) để 2 lượt vừa DEMO_BUDGET_S. Không dùng /proc hay getloadavg, nên chạy được trên
    Windows."""
    import dataclasses
    import platform

    step_ms = _step_ms("synthetic", "smallcnn", 1, False)
    config = dataclasses.replace(load_config("configs/profiles/fast.yaml"), dataset="synthetic")
    assembly = assemble(config, datasets.load("synthetic"))
    states = [initial_model(config).state_dict() for _ in range(config.num_nodes)]
    evaluator = assembly.evaluator(assembly.final_set())  # 500 ảnh test tổng hợp
    evaluator.evaluate(states)
    start = time.perf_counter()
    evaluator.evaluate(states)
    per_image = (time.perf_counter() - start) / len(assembly.final_set().y)
    mark_s, final_s = 2_000 * per_image, 10_000 * per_image  # đánh giá tỉ lệ với số ảnh

    def run_s(steps: int) -> float:
        return config.num_nodes * steps * step_ms / 1000 + steps // 100 * mark_s + final_s

    steps = max(
        (s for s in range(100, 4_001, 100) if 2 * run_s(s) + DEMO_FIXED_S <= DEMO_BUDGET_S), default=0
    )
    print(f"máy: {platform.platform()}; {platform.processor() or platform.machine()}; {os.cpu_count()} lõi")
    print(f"python {platform.python_version()}, torch {torch.__version__}")
    print(f"một bước SmallCNN (B = 32, 1 luồng): {step_ms:.2f} ms")
    print(
        f"một mốc đánh giá (21 model, 2,000 ảnh): {mark_s:.2f} s; đánh giá cuối (10,000 ảnh): {final_s:.2f} s"
    )
    limit = f"{DEMO_BUDGET_S:.0f} s, kể cả {DEMO_FIXED_S:.0f} s cố định"
    print(f"demo: số bước lớn nhất để 2 lượt N = 20 vừa {limit}:")
    print(f"  {steps}")
    if steps:
        print(f"  ước tính: mỗi lượt {run_s(steps):.0f} s, tổng {2 * run_s(steps) + DEMO_FIXED_S:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("cores").add_argument("--seconds", type=float, default=60.0)
    sub.add_parser("step")
    sub.add_parser("eval")
    sub.add_parser("laptop")
    args = parser.parse_args()
    {"cores": lambda: cores(args.seconds), "step": step, "eval": evaluation, "laptop": laptop}[args.what]()


if __name__ == "__main__":
    main()
