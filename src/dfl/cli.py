"""Dòng lệnh `python -m dfl <lệnh>` (SPEC §4, §5.2; QS-7): `download` (T03), `run` (T11b), `describe` (T05a),
`sweep` (T12), `node` (T22b: một tiến trình node của process runtime, cũng là lệnh của container Docker).

Module này không import torch: mỗi lệnh con import module của nó bên trong hàm xử lý. Nhờ vậy `run` đọc cấu
hình, đặt OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS theo `threads` của cấu hình, rồi mới nạp
sim.run (và torch), vì các thư viện này đọc biến môi trường lúc được nạp (SPEC §5.2).

Mã thoát của `run`: 0 khi lượt xong (ok hay diverged; phân kỳ là kết quả, không phải lỗi); 1 khi lượt lỗi
(SummaryFailed đã được ghi, traceback in ra stderr); 3 khi lượt bị từ chối ghi vào results/ (SPEC §5.3); 4 khi
một tiến trình khác đang giữ khóa của run_id; 5 khi code của tiến trình không phải code mà sweep chỉ định (nạp
dfl ngoài `--expect-root`, hay code_version khác `--expect-code-version`). Sweep dựa vào các mã này.

argparse, lệnh con bằng add_subparsers và set_defaults:
https://docs.python.org/3.12/library/argparse.html#sub-commands
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Tính từ thư mục đang đứng, tức gốc repo; .gitignore có /data/.
DATA_ROOT = Path("data")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
EXIT_REFUSED = 3
EXIT_LOCKED = 4
EXIT_CODE_MISMATCH = 5


def _download(args: argparse.Namespace) -> int:
    from dfl.data import datasets  # nạp torch và torchvision chỉ khi chạy lệnh này

    if args.dataset == "cifar10":
        datasets.download_cifar10(args.root)  # bản binary, ADR-012
        print(f"CIFAR-10 đã có trong {args.root / datasets.CIFAR10_DIR}")
        return 0
    datasets.download_mnist(args.root)
    print(f"MNIST đã có trong {args.root}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from dfl.config import load_config  # không nạp torch
    from dfl.io.manifest import repo_root

    if args.expect_root is not None and repo_root() != args.expect_root.resolve():
        print(
            f"code: dfl được nạp từ {repo_root()}, không từ worktree {args.expect_root} mà sweep chỉ định; "
            "đặt PYTHONPATH=<worktree>/src",
            file=sys.stderr,
        )
        return EXIT_CODE_MISMATCH
    config = load_config(args.config)
    if args.runtime is not None:  # ghi đè trước khi tính run_id, nên runtime vẫn nằm trong cấu hình chuẩn hóa
        config = dataclasses.replace(config, runtime=args.runtime)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config.threads)
    from dfl.sim import run as run_module  # nạp torch sau khi đã đặt số luồng

    try:
        attempt = run_module.run(
            config,
            out=args.out,
            results_root=args.results,
            data_root=args.data_root,
            expected_code_version=args.expect_code_version,
        )
    except run_module.CodeMismatch as err:
        print(f"code: {err}", file=sys.stderr)
        return EXIT_CODE_MISMATCH
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

    labels = datasets.load(config.dataset, root=args.data_root, with_test=False).train.y.numpy()
    report = describe.describe(config, labels, seeds=args.seeds)
    report["topologies"] = describe.describe_topologies(config, seeds=args.seeds)
    describe.write(report, args.out)
    print("\n".join(describe.topology_table(report["topologies"], config.num_nodes)))
    print(f"đã ghi {args.out / 'describe.json'} và {args.out / 'describe.csv'}")
    if not args.check_req02:
        return 0
    problems = describe.check_req02(describe.js_means(report), describe.load_table(args.table))
    for problem in problems:
        print(f"✗ {problem}")
    print("REQ-02: đạt" if not problems else f"REQ-02: không đạt ({len(problems)} vi phạm)")
    return 1 if problems else 0


def _sweep(args: argparse.Namespace) -> int:
    from dfl.sim import sweep  # không nạp torch: mỗi lượt chạy trong tiến trình con riêng

    try:
        report = sweep.run_sweep(
            args.experiment,
            out=args.out,
            results_root=args.results,
            workers=args.workers,
            dry_run=args.dry_run,
            retry_failed=args.retry_failed,
            only=args.only,
            data_root=args.data_root,
        )
    except sweep.SweepRefused as err:
        print(f"từ chối: {err}", file=sys.stderr)
        return EXIT_REFUSED
    plan = report.plan
    print(
        f"{report.experiment.name}: tổng {plan.total} / đã xong {plan.done} / failed {plan.failed} / "
        f"sẽ chạy {len(plan.to_run)}"
    )
    if args.dry_run:
        return 0
    counts: dict[str, int] = {}
    for outcome in report.outcomes:
        counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
    print(
        "kết quả: " + (", ".join(f"{name} {n}" for name, n in sorted(counts.items())) or "không có lượt nào")
    )
    if report.stopped:
        print("sweep dừng xếp lượt mới (code_version đổi, hay một lượt bị từ chối bất ngờ)", file=sys.stderr)
    problems = {"failed", "crashed", "refused", "code_changed"}
    return 1 if report.stopped or any(o.outcome in problems for o in report.outcomes) else 0


def _node(args: argparse.Namespace) -> int:
    from dfl.config import load_config  # không nạp torch
    from dfl.io.manifest import current_code_version

    config = load_config(args.config)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(config.threads)
    if args.expect_code_version is not None and current_code_version() != args.expect_code_version:
        print("code: code_version của node khác của tiến trình chạy lượt", file=sys.stderr)
        return EXIT_CODE_MISMATCH
    if args.world_size != config.num_nodes + 1:
        print(f"world-size phải là N + 1 = {config.num_nodes + 1} (N node và một observer)", file=sys.stderr)
        return 2
    from dfl.sim import process_runtime  # nạp torch sau khi đã đặt số luồng

    process_runtime.run_node(
        config,
        rank=args.rank,
        init_method=args.init_method,
        data_root=args.data_root,
        ready_dir=args.ready_dir,
        timeout=args.timeout,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl", description="Mô phỏng học liên kết phi tập trung (DFL) trên dữ liệu Non-IID."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<lệnh>")
    download = commands.add_parser("download", help="tải dữ liệu một lần (cần mạng)")
    download.add_argument("dataset", choices=["mnist", "cifar10"], help="bộ dữ liệu cần tải")
    download.add_argument("--root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    download.set_defaults(handler=_download)
    run = commands.add_parser("run", help="chạy một lượt từ một file cấu hình")
    run.add_argument("--config", type=Path, required=True, help="file YAML của lượt chạy")
    run.add_argument(
        "--out", type=Path, default=None, help="thư mục kết quả ngoài results/, ví dụ build/smoke"
    )
    run.add_argument("--data-root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    run.add_argument(
        "--results", type=Path, default=Path("results"), help="thư mục results; --out không được dẫn vào nó"
    )
    run.add_argument(
        "--runtime",
        choices=["inprocess", "processes"],
        default=None,
        help="ghi đè runtime của cấu hình (ADR-010)",
    )
    run.add_argument("--expect-code-version", default=None, help="sweep truyền: code_version phải có")
    run.add_argument(
        "--expect-root", type=Path, default=None, help="sweep truyền: dfl phải nạp từ worktree này"
    )
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
    sweep = commands.add_parser("sweep", help="chạy các ô chưa xong của một nhóm thí nghiệm")
    sweep.add_argument(
        "--experiment", type=Path, required=True, help="YAML thí nghiệm (configs/experiments/)"
    )
    sweep.add_argument("--out", type=Path, default=None, help="ghi ra thư mục này thay vì results/ (demo)")
    sweep.add_argument(
        "--results", type=Path, default=Path("results"), help="thư mục results (mặc định: results)"
    )
    sweep.add_argument("--workers", type=int, default=1, help="số lượt chạy song song (mặc định: 1)")
    sweep.add_argument("--dry-run", action="store_true", help="chỉ in tổng / đã xong / failed / sẽ chạy")
    sweep.add_argument("--retry-failed", action="store_true", help="chạy lại cả các run_id có attempt failed")
    sweep.add_argument("--only", action="append", default=[], help="chỉ các ô có khóa=giá trị, lặp được")
    sweep.add_argument("--data-root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    sweep.set_defaults(handler=_sweep)
    node = commands.add_parser("node", help="một tiến trình node của process runtime (ADR-010)")
    node.add_argument("--rank", type=int, required=True, help="id của node, 0 … N − 1")
    node.add_argument("--world-size", type=int, required=True, help="N + 1 (N node và observer)")
    node.add_argument(
        "--init-method", required=True, help="rendezvous của gloo: file://… hay tcp://host:port"
    )
    node.add_argument("--config", type=Path, required=True, help="file YAML của lượt chạy")
    node.add_argument("--data-root", type=Path, default=DATA_ROOT, help="thư mục dữ liệu (mặc định: data)")
    node.add_argument("--ready-dir", type=Path, default=None, help="tạo file ready-<rank> ở đây khi sẵn sàng")
    node.add_argument("--expect-code-version", default=None, help="code_version mà tiến trình chạy lượt có")
    node.add_argument("--timeout", type=float, default=600.0, help="thời gian chờ của gloo, giây")
    node.set_defaults(handler=_node)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
