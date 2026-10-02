"""Dòng lệnh `python -m dfl <lệnh>` (SPEC §4, §5.2; QS-7): `download` (T03), `run` (T11b); `describe` và
`sweep` được thêm ở T05a, T12.

Module này không import torch: mỗi lệnh con import module của nó bên trong hàm xử lý. Nhờ vậy `run` đọc cấu
hình, đặt OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS theo `threads` của cấu hình, rồi mới nạp
sim.run (và torch), vì các thư viện này đọc biến môi trường lúc được nạp (SPEC §5.2).

argparse, lệnh con bằng add_subparsers và set_defaults:
https://docs.python.org/3.12/library/argparse.html#sub-commands
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

# Tính từ thư mục đang đứng, tức gốc repo; .gitignore có /data/.
DATA_ROOT = Path("data")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


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

    attempt = run_module.run(config, out=args.out, data_root=args.data_root)
    print(f"đã ghi {attempt}")
    return 0


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
    run.add_argument("--out", type=Path, required=True, help="thư mục kết quả, ví dụ build/smoke")
    run.add_argument("--data-root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    run.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
