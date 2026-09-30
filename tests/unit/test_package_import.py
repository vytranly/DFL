"""Smoke test cho khung dự án: pytest có ít nhất một test để thoát mã 0, và mọi subpackage import được."""

import importlib

import pytest

SUBPACKAGES = ["dfl", "dfl.data", "dfl.network", "dfl.learning", "dfl.sim", "dfl.metrics", "dfl.io"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name: str) -> None:
    assert importlib.import_module(name) is not None
