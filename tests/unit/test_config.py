"""Test cho src/dfl/config.py: đọc YAML, điền mặc định, kiểm kiểu/giá trị, JSON chuẩn hóa cho run_id
(T02b; SPEC §5.2, §5.3; ADR-007, ADR-008, ADR-010, ADR-011; G23).
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dfl import config as cfg

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_base_config_holds_the_defaults_of_kh_3_5() -> None:
    config = cfg.load_config(CONFIGS / "base.yaml")

    assert (config.dataset, config.num_nodes, config.local_steps, config.model) == (
        "mnist",
        20,
        4000,
        "smallcnn",
    )
    assert (config.algorithm.tau, config.algorithm.batch_size, config.rounds) == (20, 32, 200)
    assert (config.mixing, config.partition.min_size, config.threads) == ("metropolis_hastings", 64, 1)
    assert (config.eval.split, config.eval.subset_size, config.eval.every_steps) == ("test", 2000, 100)
    assert (config.runtime, config.deploy) == ("inprocess", "local")


def test_base_yaml_spells_out_exactly_the_dataclass_defaults() -> None:
    # Mặc định nằm ở hai nơi (dataclass và base.yaml) để base.yaml đọc được; test giữ hai nơi trùng nhau.
    assert cfg.canonical_json(cfg.load_config(CONFIGS / "base.yaml")) == cfg.canonical_json(cfg.RunConfig())


def test_smoke_profile_follows_spec_5_2() -> None:
    config = cfg.load_config(CONFIGS / "profiles" / "smoke.yaml")

    assert (config.dataset, config.num_nodes, config.algorithm.tau, config.rounds) == ("synthetic", 4, 2, 6)
    assert config.eval.every_steps is None  # đánh giá mỗi round
    assert config.eval.subset_size == 500
    assert config.partition.scheme in {"iid", "dirichlet"}  # #C = k cần N ≥ 10 (G23)


def test_profile_extends_base_and_merges_nested_sections(tmp_path: Path) -> None:
    profile = write(tmp_path, "p.yaml", f"extends: {CONFIGS / 'base.yaml'}\nalgorithm:\n  tau: 10\n")

    config = cfg.load_config(profile)

    assert config.algorithm.tau == 10
    assert config.algorithm.lr == cfg.RunConfig().algorithm.lr  # giữ giá trị của base
    assert config.rounds == 400


def test_equivalent_yaml_files_give_the_same_canonical_json(tmp_path: Path) -> None:
    # Khác thứ tự khóa, 0.05 so với 5.0e-2, 0 so với 0.0, và trường bỏ trống dùng mặc định.
    first = write(tmp_path, "a.yaml", "num_nodes: 20\nalgorithm:\n  lr: 0.05\n  mu: 0\n  tau: 20\n")
    second = write(tmp_path, "b.yaml", "algorithm:\n  tau: 20\n  mu: 0.0\n  lr: 5.0e-2\nnum_nodes: 20\n")

    assert cfg.canonical_json(cfg.load_config(first)) == cfg.canonical_json(cfg.load_config(second))


def test_canonical_json_sorts_keys_writes_floats_by_repr_and_omits_run_context(tmp_path: Path) -> None:
    profile = write(tmp_path, "p.yaml", f"extends: {CONFIGS / 'base.yaml'}\nalgorithm:\n  lr: 1.0e-5\n")

    text = cfg.canonical_json(cfg.load_config(profile))

    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    assert '"lr":1e-05' in text
    for context in ("extends", "name", "out", "workers"):
        assert f'"{context}"' not in text


def test_a_number_that_pyyaml_reads_as_a_string_is_rejected_with_a_hint(tmp_path: Path) -> None:
    # PyYAML đọc 5e-2 thành chuỗi (thiếu dấu chấm thập phân), nên phải viết 0.05 hay 5.0e-2 (SPEC §5.3).
    path = write(tmp_path, "c.yaml", "algorithm:\n  lr: 5e-2\n")

    with pytest.raises(ValueError, match="5.0e-2"):
        cfg.load_config(path)


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("partition:\n  scheme: label_k\n  k: 2\nnum_nodes: 8\n", "N ≥ 10"),
        ("algorithm:\n  tau: 30\nlocal_steps: 3000\n", "chia hết"),
        ("algorithm:\n  tau: 30\nlocal_steps: 4000\neval:\n  every_steps: null\n", "local_steps"),
        ("num_node: 20\n", "không có"),
        ("algorithm:\n  momentum: 0.9\n", "không có"),
        ("topology:\n  kind: hypercube\n", "kind"),
        ("num_nodes: true\n", "số nguyên"),
        ("num_nodes: 20.0\n", "số nguyên"),
        ("algorithm:\n  lr: -0.1\n", "lr"),
        ("runtime: yes\n", "runtime"),
        ("deploy: docker\n", "processes"),
        ("mixing: size_weighted\n", "size_weighted"),
        ("topology:\n  kind: server\n", "size_weighted"),
        ("partition:\n  scheme: dirichlet\n", "alpha"),
        ("partition:\n  alpha: 0.5\n", "alpha"),
        ("topology:\n  p: 0.3\n", "p"),
        ("topology:\n  kind: erdos_renyi\n  p: 1.5\n", "p"),
        ("eval:\n  split: train\n", "split"),
        ("model: resnet\n", "model"),
        ("model: cnn_niidbench\n", "1×28×28"),
        ("dataset: cifar10\n", "cnn_niidbench"),
        ("augmentation: crop_flip\n", "3×32×32"),
        ("dataset: cifar10\nmodel: cnn_niidbench\naugmentation: flip\n", "augmentation"),
        ("[1, 2]\n", "mapping"),
        ("algorithm: 3\n", "mapping"),
        ("num_nodes: 0\n", "num_nodes"),
        ("num_nodes: null\n", "trống"),
        ("algorithm:\n  lr: [0.1]\n", "phải là số"),
        ("partition:\n  scheme: label_k\n  k: 11\n", "1..10"),
        ("partition:\n  scheme: dirichlet\n  alpha: 0.0\n", "alpha"),
        ("partition:\n  scheme: quantity\n  beta: -0.5\n", "beta"),
        ("extends: 3\n", "extends"),
        ("num_nodes: twenty\n", "số nguyên"),
        ("topology:\n  kind: central\n", "num_nodes: 1"),
        ("num_nodes: 1\n", "central"),
        ("num_nodes: 1\ntopology:\n  kind: central\npartition:\n  scheme: dirichlet\n  alpha: 0.5\n", "iid"),
        ("num_nodes: 1\ntopology:\n  kind: central\nmixing: uniform\n", "metropolis_hastings"),
        ("mixing: size_corrected\nalgorithm:\n  mu: 0.1\n", "mu = 0"),
        ("runtime: processes\ntopology:\n  kind: server\nmixing: size_weighted\n", "inprocess"),
        ("runtime: processes\nnum_nodes: 1\ntopology:\n  kind: central\n", "inprocess"),
    ],
    ids=[
        "label-k-small-n",
        "grid-not-divisible",
        "steps-not-divisible",
        "unknown-top-key",
        "unknown-nested-key",
        "bad-topology",
        "bool-as-int",
        "float-as-int",
        "negative-lr",
        "yaml-bool-as-enum",
        "docker-needs-processes",
        "size-weighted-off-fc",
        "server-needs-size-weighted",
        "dirichlet-needs-alpha",
        "alpha-without-dirichlet",
        "p-without-er",
        "p-out-of-range",
        "bad-split",
        "bad-model",
        "model-dataset-shape-mismatch",
        "cifar10-needs-its-model",
        "crop-flip-needs-cifar10-images",
        "bad-augmentation",
        "not-a-mapping",
        "section-not-a-mapping",
        "zero-nodes",
        "null-required",
        "list-as-float",
        "k-too-large",
        "alpha-zero",
        "beta-negative",
        "extends-not-a-path",
        "text-as-int",
        "central-needs-one-node",
        "one-node-needs-central",
        "central-needs-iid",
        "central-needs-mh",
        "size-corrected-needs-mu-0",
        "processes-without-server",
        "processes-without-central",
    ],
)
def test_invalid_configs_raise_value_error_naming_the_constraint(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    path = write(tmp_path, "bad.yaml", yaml_text)

    with pytest.raises(ValueError, match=message):
        cfg.load_config(path)


def test_erdos_renyi_gets_the_default_edge_probability(tmp_path: Path) -> None:
    config = cfg.load_config(write(tmp_path, "er.yaml", "topology:\n  kind: erdos_renyi\n"))

    assert config.topology.p == 0.3


def test_config_is_frozen() -> None:
    config = cfg.RunConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.num_nodes = 3


def test_an_empty_file_gives_the_defaults(tmp_path: Path) -> None:
    assert cfg.load_config(write(tmp_path, "empty.yaml", "")) == cfg.RunConfig()


def test_extends_cycles_are_rejected(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", "extends: b.yaml\nnum_nodes: 5\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\n")

    with pytest.raises(ValueError, match="vòng"):
        cfg.load_config(tmp_path / "a.yaml")


def test_quantity_scheme_takes_beta(tmp_path: Path) -> None:
    config = cfg.load_config(write(tmp_path, "q.yaml", "partition:\n  scheme: quantity\n  beta: 0.5\n"))

    assert (config.partition.scheme, config.partition.beta) == ("quantity", 0.5)


def test_reading_a_config_does_not_import_torch() -> None:
    # CLI đọc cấu hình rồi mới đặt biến môi trường số luồng và nạp torch (SPEC §5.2). Kiểm trong một
    # tiến trình sạch, vì tiến trình pytest đã nạp torch.
    code = (
        "import sys; from dfl.config import load_config; "
        f"load_config({str(CONFIGS / 'profiles' / 'smoke.yaml')!r}); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_fast_profile_follows_spec_5_2() -> None:
    config = cfg.load_config(CONFIGS / "profiles" / "fast.yaml")

    assert (config.dataset, config.num_nodes, config.local_steps, config.algorithm.tau) == (
        "mnist",
        20,
        1000,
        10,
    )
    assert (config.rounds, config.eval.every_steps, config.eval.split) == (100, 100, "test")


def test_central_baseline_is_one_node_without_topology(tmp_path: Path) -> None:
    config = cfg.load_config(write(tmp_path, "b1.yaml", "num_nodes: 1\ntopology:\n  kind: central\n"))

    assert (config.num_nodes, config.topology.kind, config.partition.scheme) == (1, "central", "iid")


def test_every_dataset_and_model_declares_an_input_shape() -> None:
    assert set(cfg.DATASET_INPUTS) == set(cfg.DATASETS)
    assert set(cfg.MODEL_INPUTS) == set(cfg.MODELS)
    assert all(shape in set(cfg.MODEL_INPUTS.values()) for shape in cfg.DATASET_INPUTS.values())


@pytest.mark.parametrize(
    ("profile", "base", "dataset"),
    [("smoke_cifar10", "smoke", "synthetic_cifar10"), ("cifar10_fast", "fast", "cifar10")],
)
def test_cifar10_profiles_change_only_the_dataset_and_model_of_their_mnist_twin(
    profile: str, base: str, dataset: str
) -> None:
    # ADR-012: cùng quy mô, ngân sách và lịch đánh giá với profile MNIST tương ứng.
    loaded = cfg.load_config(CONFIGS / "profiles" / f"{profile}.yaml")
    twin = cfg.load_config(CONFIGS / "profiles" / f"{base}.yaml")

    assert loaded == dataclasses.replace(
        twin, dataset=dataset, model="cnn_niidbench", augmentation="crop_flip"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "none"),
        ("dataset: cifar10\nmodel: cnn_niidbench\n", "crop_flip"),
        ("dataset: synthetic_cifar10\nmodel: cnn_niidbench\naugmentation: null\n", "crop_flip"),
        ("dataset: cifar10\nmodel: cnn_niidbench\naugmentation: none\n", "none"),
    ],
    ids=["mnist-default", "cifar10-default", "synthetic-cifar10-null", "cifar10-none"],
)
def test_augmentation_null_follows_the_dataset_and_survives_the_child_yaml(
    tmp_path: Path, text: str, expected: str
) -> None:
    # ADR-012: null là crop_flip với ảnh 3×32×32, none với MNIST; "none" không bị PyYAML đọc thành null.
    config = cfg.load_config(write(tmp_path, "a.yaml", text))

    assert config.augmentation == expected
    assert cfg.load_config(write(tmp_path, "child.yaml", cfg.to_yaml(config))) == config
