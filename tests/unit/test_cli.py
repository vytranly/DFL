"""Test cho src/dfl/cli.py và src/dfl/__main__.py: khung lệnh `python -m dfl` và lệnh `download` (T03; SPEC
§4, §5.2; QS-7). Không dùng mạng: hàm tải được thay bằng bản ghi lại lời gọi.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from dfl import cli
from dfl.data import datasets


def test_download_mnist_fetches_into_the_given_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(datasets, "download_mnist", calls.append)

    code = cli.main(["download", "mnist", "--root", str(tmp_path)])

    assert (code, calls) == (0, [tmp_path])


def test_download_defaults_to_the_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(datasets, "download_mnist", calls.append)

    cli.main(["download", "mnist"])

    assert calls == [Path("data")]


@pytest.mark.parametrize(
    "argv", [["download", "cifar10"], [], ["frobnicate"]], ids=["dataset", "none", "cmd"]
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
