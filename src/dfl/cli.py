"""Dòng lệnh `python -m dfl <lệnh>` (SPEC §4, §5.2; QS-7): `download` (T03), `run` (T11b), `describe` (T05a);
`sweep` được thêm ở T12.

Module này không import torch: mỗi lệnh con import module của nó bên trong hàm xử lý. Nhờ vậy `run` đọc cấu
hình, đặt OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS theo `threads` của cấu hình, rồi mới nạp
sim.run (và torch), vì các thư viện này đọc biến môi trường lúc được nạp (SPEC §5.2).

Mã thoát của `run`: 0 khi lượt xong (ok hay diverged; phân kỳ là kết quả, không phải lỗi); 1 khi lượt lỗi
(SummaryFailed đã được ghi, traceback in ra stderr); 3 khi lượt bị từ chối ghi vào results/ (SPEC §5.3); 4 khi
một tiến trình khác đang giữ khóa của run_id. Sweep (T12) dựa vào các mã này.

argparse, lệnh con bằng add_subparsers và set_defaults:
https://docs.python.org/3.12/library/argparse.html#sub-commands
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Tính từ thư mục đang đứng, tức gốc repo; .gitignore có /data/.
DATA_ROOT = Path("data")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
EXIT_REFUSED = 3
EXIT_LOCKED = 4


def _download(args: argparse.Namespace) -> int:
    from dfl.data import datasets  # nạp torch và torchvision chỉ khi chạy lệnh này

    datasets.download_mnist(args.root)
    print(f"MNIST đã có trong {args.root}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from dfl.config import load_config  # không nạp torch

    config = load_config(args.config)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config.threads)
    from dfl.sim import run as run_module  # nạp torch sau khi đã đặt số luồng

    try:
        attempt = run_module.run(config, out=args.out, data_root=args.data_root)
    except run_module.RunLocked as err:
        print(f"bỏ qua: {err}", file=sys.stderr)
        return EXIT_LOCKED
    except run_module.RunRefused as err:
        print(f"từ chối: {err}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"đã ghi {attempt}")
    return 0


def _describe(args: argparse.Namespace) -> int:
    from dfl.config import load_config  # không nạp torch

    config = load_config(args.config)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config.threads)
    from dfl.data import datasets  # nạp torch sau khi đã đặt số luồng
    from dfl.sim import describe

    labels = datasets.load(config.dataset, root=args.data_root).train.y.numpy()
    report = describe.describe(config, labels, seeds=args.seeds)
    describe.write(report, args.out)
    print(f"đã ghi {args.out / 'describe.json'} và {args.out / 'describe.csv'}")
    if not args.check_req02:
        return 0
    problems = describe.check_req02(describe.js_means(report), describe.load_table(args.table))
    for problem in problems:
        print(f"✗ {problem}")
    print("REQ-02: đạt" if not problems else f"REQ-02: không đạt ({len(problems)} vi phạm)")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl", description="Mô phỏng học liên kết phi tập trung (DFL) trên dữ liệu Non-IID."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<lệnh>")
    download = commands.add_parser("download", help="tải dữ liệu một lần (cần mạng)")
    download.add_argument("dataset", choices=["mnist"], help="bộ dữ liệu cần tải")
    download.add_argument("--root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    download.set_defaults(handler=_download)
    run = commands.add_parser("run", help="chạy một lượt từ một file cấu hình")
    run.add_argument("--config", type=Path, required=True, help="file YAML của lượt chạy")
    run.add_argument(
        "--out", type=Path, default=None, help="thư mục kết quả ngoài results/, ví dụ build/smoke"
    )
    run.add_argument("--data-root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    run.set_defaults(handler=_run)
    describe = commands.add_parser("describe", help="mô tả các cách chia dữ liệu, không huấn luyện")
    describe.add_argument("--config", type=Path, required=True, help="file YAML (dùng dataset, N, min_size)")
    describe.add_argument("--seeds", type=int, default=20, help="số seed, 0 … seeds − 1 (mặc định: 20)")
    describe.add_argument("--out", type=Path, required=True, help="thư mục ghi describe.json, describe.csv")
    describe.add_argument(
        "--data-root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)"
    )
    describe.add_argument("--check-req02", action="store_true", help="so JS với bảng KH §3.2 (REQ-02)")
    describe.add_argument(
        "--table", type=Path, default=Path("configs/reference/kh_js_table.yaml"), help="bảng JS tham chiếu"
    )
    describe.set_defaults(handler=_describe)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
