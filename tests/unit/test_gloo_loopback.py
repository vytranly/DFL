"""Gloo của process runtime cục bộ chỉ nghe trên loopback (security audit T25).

Mặc định gloo nghe trên địa chỉ phân giải từ hostname, có thể là card mạng thật, nên máy cùng mạng tới được
cổng của nó. `GLOO_SOCKET_IFNAME` chọn interface; người dùng tự đặt thì giữ nguyên giá trị đó.
"""

import os
import socket
import sys

import pytest

from dfl.sim import process_runtime


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tên interface loopback của Linux")
def test_the_loopback_interface_of_linux_is_lo() -> None:
    assert process_runtime.loopback_interface() == "lo"


def test_no_loopback_interface_leaves_gloo_on_its_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "eth0")])

    assert process_runtime.loopback_interface() is None
    assert "GLOO_SOCKET_IFNAME" not in process_runtime.gloo_env({})


def test_node_processes_get_the_loopback_interface_unless_the_user_chose_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_runtime, "loopback_interface", lambda: "lo")

    assert process_runtime.gloo_env({"PATH": "/bin"}) == {"PATH": "/bin", "GLOO_SOCKET_IFNAME": "lo"}
    assert process_runtime.gloo_env({"GLOO_SOCKET_IFNAME": "eth1"})["GLOO_SOCKET_IFNAME"] == "eth1"


def test_the_observer_sets_the_interface_only_while_gloo_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_runtime, "loopback_interface", lambda: "lo")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)

    with process_runtime.gloo_loopback():
        inside = os.environ.get("GLOO_SOCKET_IFNAME")

    assert inside == "lo" and "GLOO_SOCKET_IFNAME" not in os.environ
