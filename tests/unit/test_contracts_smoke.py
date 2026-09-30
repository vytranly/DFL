"""Smoke test cho src/dfl/contracts.py (P1.7): import được; mỗi Protocol đúng tên và đúng tập thành viên;
schema bản ghi khớp SPEC §5.3; value object không sửa được; import không nạp torch (SPEC §3.2)."""

import dataclasses
import subprocess
import sys
import typing

import pytest

from dfl import contracts

PROTOCOL_MEMBERS = {
    "Partitioner": {"__call__"},
    "TopologySchedule": {"num_nodes", "is_static", "graph_at"},
    "MixingPolicy": {"__call__"},
    "Transport": {"send", "inbox", "end_round", "counters"},
    "LocalTrainer": {"__call__"},
    "Evaluator": {"evaluate"},
    "RunLogger": {"start", "log_round", "log_timing", "finish"},
}

RECORD_KEYS = {
    "RoundRecord": {
        "round",
        "local_steps",
        "acc_mean",
        "acc_std",
        "acc_min",
        "acc_max",
        "acc_local_mean",
        "acc_local_std",
        "acc_local_min",
        "f1_macro_mean",
        "f1_macro_std",
        "f1_macro_min",
        "acc_avg_model",
        "acc_eff_mean",
        "drift_pre_combine",
        "consensus_distance",
        "messages_cum",
        "bytes_cum",
        "max_node_bytes_cum",
        "train_loss_mean",
        "acc_per_node",
        "acc_local_per_node",
    },
    "TimingRecord": {"round", "wall_time_s", "train_s", "combine_s", "eval_s"},
    "SummaryOk": {
        "status",
        "messages_total",
        "bytes_total",
        "final_acc_per_node_full_test",
        "final_acc_local_per_node_full_test",
        "final_f1_macro_per_node_full_test",
        "final_acc_avg_model_full_test",
    },
    "SummaryFailed": {"status", "error", "messages_total", "bytes_total"},
    "ManifestRecord": {
        "run_id",
        "code_version",
        "dirty",
        "experiment",
        "attempt",
        "config",
        "seed",
        "child_seeds",
        "versions",
        "machine",
        "cpu_threads",
        "created_at",
        "spectral_gap",
        "consensus_weights",
        "objective_shift_tv",
        "node_class_counts",
    },
}

VALUE_OBJECTS = ["Snapshot", "CommCounters", "NodeDataset", "EvalResult"]


def _fake_with(members: set[str]) -> object:
    """Đối tượng chỉ có đúng các thành viên cho trước (giá trị khác None, để Protocol chấp nhận)."""
    return type("Fake", (), {name: (lambda self, *args, **kwargs: None) for name in members})()


@pytest.mark.parametrize("name", sorted(PROTOCOL_MEMBERS))
def test_contract_is_a_runtime_checkable_protocol_without_extra_members(name: str) -> None:
    protocol = getattr(contracts, name)

    assert typing.Protocol in protocol.__bases__
    assert isinstance(_fake_with(PROTOCOL_MEMBERS[name]), protocol)


@pytest.mark.parametrize(("name", "members"), sorted(PROTOCOL_MEMBERS.items()))
def test_every_expected_protocol_member_is_required(name: str, members: set[str]) -> None:
    protocol = getattr(contracts, name)

    for missing in sorted(members):
        assert not isinstance(_fake_with(members - {missing}), protocol), missing


@pytest.mark.parametrize(("name", "keys"), sorted(RECORD_KEYS.items()))
def test_record_schema_has_exactly_the_spec_keys(name: str, keys: set[str]) -> None:
    record = getattr(contracts, name)

    assert typing.is_typeddict(record)
    assert record.__required_keys__ == keys


def test_summary_record_is_the_union_of_ok_and_failed_variants() -> None:
    variants = typing.get_args(contracts.SummaryRecord.__value__)

    assert variants == (contracts.SummaryOk, contracts.SummaryFailed)


@pytest.mark.parametrize("name", VALUE_OBJECTS)
def test_value_objects_cannot_be_modified_after_creation(name: str) -> None:
    cls = getattr(contracts, name)
    fields = dataclasses.fields(cls)
    instance = cls(**{field.name: None for field in fields})

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, fields[0].name, 0)


def test_importing_contracts_does_not_load_torch() -> None:
    code = "import sys, dfl.contracts; sys.exit('torch' in sys.modules)"

    result = subprocess.run([sys.executable, "-c", code], check=False)

    assert result.returncode == 0
