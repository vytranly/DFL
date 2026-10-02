"""Test cho hàm so JS với bảng KH §3.2 của `dfl describe --check-req02` (T05a; SPEC REQ-02), trên số giả."""

from pathlib import Path

import pytest

from dfl.sim import describe

TABLE = Path(__file__).resolve().parents[2] / "configs" / "reference" / "kh_js_table.yaml"
FAKE_TABLE = {
    "iid": 0.001,
    "dirichlet_100": 0.002,
    "dirichlet_1": 0.185,
    "dirichlet_0.5": 0.283,
    "dirichlet_0.1": 0.553,
    "dirichlet_0.05": 0.631,
    "label_2": 0.613,
    "label_1": 0.758,
    "quantity_0.5": 0.003,
}


def test_values_matching_the_table_in_the_required_order_pass() -> None:
    measured = {**FAKE_TABLE, "dirichlet_1": 0.2, "iid": 0.0012}  # lệch 8% và 0.0002: trong sai số

    assert describe.check_req02(measured, FAKE_TABLE) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dirichlet_0.1": 0.62}, "dirichlet_0.1"),  # lệch 12% > 10%
        ({"iid": 0.0035}, "iid"),  # lệch 0.0025 > 0.002
        ({"dirichlet_0.5": 0.18, "dirichlet_1": 0.185}, "tăng nghiêm ngặt"),
        ({"label_1": 0.613}, "tăng nghiêm ngặt"),
    ],
    ids=["relative-tolerance", "absolute-floor", "dirichlet-order", "label-order"],
)
def test_a_value_off_the_table_or_out_of_order_fails_naming_the_scheme(
    change: dict[str, float], message: str
) -> None:
    problems = describe.check_req02({**FAKE_TABLE, **change}, FAKE_TABLE)

    assert any(message in problem for problem in problems)


def test_a_missing_scheme_fails() -> None:
    measured = {k: v for k, v in FAKE_TABLE.items() if k != "label_2"}

    assert any("label_2" in problem for problem in describe.check_req02(measured, FAKE_TABLE))


def test_the_reference_table_holds_the_js_column_of_kh_3_2() -> None:
    assert describe.load_table(TABLE) == FAKE_TABLE


def test_the_table_lists_exactly_the_described_schemes_except_shards() -> None:
    assert set(FAKE_TABLE) == {scheme.key for scheme in describe.schemes(min_size=64)} - {"shards"}


@pytest.mark.parametrize(
    "text", ["iid: 1e-3\n", "- 0.001\n", "iid: .nan\n"], ids=["yaml-string", "list", "nan"]
)
def test_a_malformed_table_is_rejected(tmp_path: Path, text: str) -> None:
    # PyYAML đọc 1e-3 thành chuỗi (SPEC §5.3), nên bảng phải viết 0.001.
    path = tmp_path / "table.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match="mapping"):
        describe.load_table(path)
