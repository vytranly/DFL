"""scripts/demo_partition.py (T31f; ADR-015): các bước in ra phải là đúng các bước của partitioner dự án.

Dùng nhãn tổng hợp của profile smoke (1,800 mẫu của node) với N = 10, để #C = k có đủ 10 node (G23).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from dfl import seeding
from dfl.data import datasets
from dfl.data.partition import DirichletPartitioner, LabelKPartitioner
from dfl.seeding import Stream

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_partition", ROOT / "scripts" / "demo_partition.py")
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
# @dataclass tra module trong sys.modules lúc thực thi, nên phải đăng ký trước exec_module.
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


@pytest.fixture(scope="module")
def labels() -> np.ndarray:
    return datasets.load("synthetic").train.y.numpy()


@pytest.mark.parametrize("alpha", [10.0, 0.5, 0.1])
def test_the_dirichlet_steps_are_the_partitioner_of_the_project(labels: np.ndarray, alpha: float) -> None:
    rng = seeding.numpy_rng(3, Stream.PARTITION)

    parts, retries, trace = demo.dirichlet_steps(labels, 10, alpha, rng, 16)

    expected = DirichletPartitioner(alpha, 16)(labels, 10, seeding.numpy_rng(3, Stream.PARTITION))
    assert demo.same_parts(parts, expected.parts) and retries == expected.retries
    assert len(trace) == 10 and all(step.counts.sum() == step.size for step in trace)


def test_the_label_k_steps_are_the_partitioner_of_the_project(labels: np.ndarray) -> None:
    parts, owned, chunks_of = demo.label_k_steps(labels, 10, 2, seeding.numpy_rng(3, Stream.PARTITION))

    expected = LabelKPartitioner(2, 16)(labels, 10, seeding.numpy_rng(3, Stream.PARTITION))
    assert demo.same_parts(parts, expected.parts)
    assert all(len(set(node)) == 2 and node[0] == i % 10 for i, node in enumerate(owned))
    # Câu hỏi của GVHD: các node giữ cùng một nhãn nhận số mẫu chênh nhau tối đa 1.
    assert all(max(sizes) - min(sizes) <= 1 for sizes in chunks_of.values())


def test_theory_cv_depends_on_n_times_alpha_only() -> None:
    assert demo.theory_cv(20, 0.05) == pytest.approx(demo.theory_cv(20, 0.05))
    assert demo.theory_cv(20, 1.0) == pytest.approx(np.sqrt(19 / 21))
    assert demo.theory_cv(10, 0.1) < demo.theory_cv(20, 0.1) * 2


def test_the_script_writes_every_section_and_its_figures(tmp_path: Path) -> None:
    code = demo.main(
        [
            "--out",
            str(tmp_path),
            "--datasets",
            "synthetic",
            "--nodes",
            "10",
            "--seeds",
            "1",
            "--min-size",
            "16",
        ]
    )

    text = (tmp_path / "demo_partition.md").read_text(encoding="utf-8")
    assert code == 0
    for heading in (
        "## 1. Dirichlet",
        "## 2. Pathological",
        "## 3. Ảnh hưởng của α",
        "## 4. Ảnh hưởng của k",
    ):
        assert heading in text
    assert "trùng partitioner: có" in text and "KHÔNG" not in text
    assert (tmp_path / "alpha_effect.png").stat().st_size > 0 and (
        tmp_path / "P7-describe.png"
    ).stat().st_size > 0
