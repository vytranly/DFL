"""Cấu hình và fixture dùng chung cho mọi test (SPEC §7).

- Mọi test chạy torch một luồng, như mỗi lượt chạy thật (SPEC §5.2, số luồng mặc định 1). Trên máy chủ dùng
  chung, 8 luồng làm test tích hợp chậm gấp khoảng 5 lần vì tranh lõi. Tài liệu ghi set_num_threads "must be
  called before running eager, JIT or autograd code", nên nó được gọi trong pytest_configure, trước mọi test.
- Fixture autouse kiểm RNG toàn cục của torch không đổi qua mỗi test trong tests/integration/ và tests/e2e/,
  để mọi đường code chạy qua engine đều bị kiểm T-INV-5 (SPEC §6): nguồn ngẫu nhiên của huấn luyện chỉ đi qua
  `torch.Generator` riêng. Test unit không bị kiểm, vì nhiều test dùng torch không qua engine.

torch.set_num_threads: https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html
torch.get_rng_state: https://docs.pytorch.org/docs/2.14/generated/torch.get_rng_state.html
pytest_configure: https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_configure
Autouse fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html#autouse-fixtures-fixtures-you-don-t-have-to-request
FixtureRequest.path ("Path where the test function was collected", đối chiếu docstring của pytest 9.1.1):
https://docs.pytest.org/en/stable/reference/reference.html#pytest.FixtureRequest.path
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

TESTS_DIR = Path(__file__).parent
GUARDED_DIRS = frozenset({"integration", "e2e"})


def pytest_configure(config: pytest.Config) -> None:
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def global_torch_rng_is_untouched(request: pytest.FixtureRequest) -> Iterator[None]:
    if Path(request.path).relative_to(TESTS_DIR).parts[0] not in GUARDED_DIRS:
        yield
        return
    before = torch.get_rng_state()
    yield
    assert torch.equal(torch.get_rng_state(), before), (
        "test đã tiêu thụ RNG toàn cục của torch; mọi nguồn ngẫu nhiên phải đi qua torch.Generator riêng "
        "(T-INV-5)"
    )
