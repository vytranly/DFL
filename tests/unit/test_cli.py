"""Test cho src/dfl/cli.py và src/dfl/__main__.py: khung lệnh `python -m dfl` và lệnh `download` (T03; SPEC
§4, §5.2; QS-7). Không dùng mạng: hàm tải được thay bằng bản ghi lại lời gọi.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dfl import cli
from dfl.config import RunConfig
from dfl.data import datasets
from dfl.sim import run as run_module


def test_download_mnist_fetches_into_the_given_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(datasets, "download_mnist", calls.append)

    code = cli.main(["download", "mnist", "--root", str(tmp_path)])

    assert (code, calls) == (0, [tmp_path])


def test_download_cifar10_fetches_into_the_given_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(datasets, "download_cifar10", calls.append)
    monkeypatch.setattr(datasets, "download_mnist", lambda root: pytest.fail("không được tải MNIST"))

    code = cli.main(["download", "cifar10", "--root", str(tmp_path)])

    assert (code, calls) == (0, [tmp_path])


def test_download_defaults_to_the_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(datasets, "download_mnist", calls.append)

    cli.main(["download", "mnist"])

    assert calls == [Path("data")]


@pytest.mark.parametrize(
    "argv", [["download", "cifar100"], [], ["frobnicate"]], ids=["dataset", "none", "cmd"]
)
def test_bad_command_lines_exit_with_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2


def test_python_m_dfl_runs_the_cli_without_importing_torch() -> None:
    # `run` (T11b) phải đặt biến môi trường số luồng trước khi torch được nạp (SPEC §5.2), nên dựng parser và
    # in trợ giúp không được nạp torch. Kiểm trong tiến trình sạch, vì tiến trình pytest đã nạp torch.
    code = (
        "import runpy, sys\n"
        "sys.argv = ['dfl', '--help']\n"
        "try:\n"
        "    runpy.run_module('dfl', run_name='__main__')\n"
        "except SystemExit as stop:\n"
        "    assert stop.code == 0, stop.code\n"
        "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "download" in result.stdout


SMOKE = Path(__file__).resolve().parents[2] / "configs" / "profiles" / "smoke.yaml"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def test_run_sets_thread_variables_before_torch_is_imported() -> None:
    # SPEC §5.2: CLI đọc cấu hình (không nạp torch), đặt biến môi trường số luồng, rồi mới nạp sim.run. Kiểm
    # trong tiến trình sạch, với một dfl.sim.run giả ghi lại trạng thái lúc được gọi.
    code = (
        "import json, os, sys, types\n"
        "import dfl.sim\n"
        "seen = {}\n"
        "def run(config, *, out, data_root, results_root, expected_code_version):\n"
        "    seen.update(torch='torch' in sys.modules, threads=config.threads, out=str(out),\n"
        f"                env=[os.environ.get(k) for k in {THREAD_VARIABLES!r}])\n"
        "    return out\n"
        "fake = types.ModuleType('dfl.sim.run')\n"
        "fake.run = run\n"
        "sys.modules['dfl.sim.run'] = dfl.sim.run = fake\n"
        "from dfl.cli import main\n"
        f"status = main(['run', '--config', {str(SMOKE)!r}, '--out', 'build/x'])\n"
        "print(json.dumps(seen))\n"
        "sys.exit(status)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARIABLES}

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env
    )

    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.splitlines()[-1])
    assert seen == {"torch": False, "threads": 1, "out": "build/x", "env": ["1", "1", "1"]}


def test_run_runtime_flag_overrides_the_config_before_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def record(config: RunConfig, **kwargs: object) -> Path:
        seen.append(config.runtime)
        return Path("x")

    monkeypatch.setattr(run_module, "run", record)
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")

    cli.main(["run", "--config", str(SMOKE), "--out", "build/x", "--runtime", "processes"])

    assert seen == ["processes"]


def test_run_without_out_writes_to_results() -> None:
    args = cli.build_parser().parse_args(["run", "--config", str(SMOKE)])

    assert args.out is None


@pytest.mark.parametrize(
    ("error", "code"),
    [(run_module.RunRefused("cây bẩn"), 3), (run_module.RunLocked("đang chạy"), 4)],
    ids=["refused", "locked"],
)
def test_run_maps_refusals_to_exit_codes_for_the_sweep(
    monkeypatch: pytest.MonkeyPatch, error: Exception, code: int
) -> None:
    def refuse(*args: object, **kwargs: object) -> Path:
        raise error

    monkeypatch.setattr(run_module, "run", refuse)
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")  # để monkeypatch trả lại môi trường cũ sau test

    assert cli.main(["run", "--config", str(SMOKE)]) == code
