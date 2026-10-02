"""Dòng lệnh `python -m dfl <lệnh>` (SPEC §4, §5.2; QS-7). T03 có lệnh `download`; `describe`, `run` và
`sweep` được thêm ở T05a, T11b, T12.

Module này không import torch: mỗi lệnh con import module của nó bên trong hàm xử lý, để `run` đọc cấu hình
và đặt biến môi trường số luồng trước khi torch được nạp (SPEC §5.2).

argparse, lệnh con bằng add_subparsers và set_defaults:
https://docs.python.org/3.12/library/argparse.html#sub-commands
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# Tính từ thư mục đang đứng, tức gốc repo; .gitignore có /data/.
DATA_ROOT = Path("data")


def _download(args: argparse.Namespace) -> int:
    from dfl.data import datasets  # nạp torch và torchvision chỉ khi chạy lệnh này

    datasets.download_mnist(args.root)
    print(f"MNIST đã có trong {args.root}")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
