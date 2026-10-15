"""Cấu hình một lượt chạy: đọc YAML, điền mặc định, kiểm kiểu và giá trị, JSON chuẩn hóa cho run_id
(nền tảng dùng chung; SPEC §5.2, §5.3, §6; ADR-007, ADR-008, ADR-010, ADR-011, ADR-012).

- Chỉ dùng `yaml.safe_load` (F6). PyYAML theo YAML 1.1: `5e-2` (thiếu dấu chấm thập phân) là chuỗi, còn
  `yes`/`no` là bool, nên mọi trường đều được kiểm kiểu; số thực phải viết `0.05` hay `5.0e-2`.
- Cấu hình là dataclass frozen, vì run_id được tính từ nó (SPEC §6). Mọi đối tượng `RunConfig` đều đã được
  kiểm, kể cả khi dựng thẳng trong code.
- Profile kế thừa bằng khóa `extends` (đường dẫn tính từ thư mục của file đang đọc). Mapping được trộn đệ quy;
  giá trị khác thì bị thay.
- JSON chuẩn hóa: khóa sắp xếp, số thực theo repr; không chứa tên nhóm thí nghiệm, `--out`, `--workers` hay
  `extends`. Tham số mà scheme hay topology không dùng phải để null, và trường số thực luôn là float, để hai
  cấu hình tương đương cho cùng JSON.
- Không import torch: CLI đọc cấu hình rồi mới đặt biến môi trường số luồng và nạp torch (SPEC §5.2).

PyYAML (safe_load): https://pyyaml.org/wiki/PyYAMLDocumentation
json (sort_keys, separators, allow_nan): https://docs.python.org/3.12/library/json.html
dataclasses (frozen, __post_init__, object.__setattr__ trong dataclass frozen):
https://docs.python.org/3.12/library/dataclasses.html
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMES = ("iid", "dirichlet", "label_k", "quantity", "shards")
# central: baseline B1/B1-4k, một node giữ đủ dữ liệu train, không có topology hay mixing (T11c).
TOPOLOGIES = (
    "ring",
    "fully_connected",
    "erdos_renyi",
    "gossip",
    "er_gossip",
    "star",
    "local_only",
    "server",
    "central",
)
ER_KINDS = ("erdos_renyi", "er_gossip")
DEFAULT_ER_P = 0.3
MIXINGS = ("metropolis_hastings", "uniform", "size_weighted", "size_corrected")
# Shape (C, H, W) của ảnh mỗi dataset, và của ảnh mỗi model nhận; model phải khớp dataset (ADR-012). Bảng nằm
# ở đây vì module này không import torch; tests/unit/test_models.py giữ nó đúng với model thật.
MNIST_INPUT = (1, 28, 28)
CIFAR10_INPUT = (3, 32, 32)
DATASET_INPUTS = {
    "mnist": MNIST_INPUT,
    "cifar10": CIFAR10_INPUT,
    "synthetic": MNIST_INPUT,
    "synthetic_cifar10": CIFAR10_INPUT,
}
MODEL_INPUTS = {
    "smallcnn": MNIST_INPUT,
    "logreg": MNIST_INPUT,
    "mlp2nn": MNIST_INPUT,
    "cnn_mcmahan": MNIST_INPUT,
    "cnn_niidbench": CIFAR10_INPUT,
}
MODELS = tuple(MODEL_INPUTS)
DATASETS = tuple(DATASET_INPUTS)
# none: không augmentation; crop_flip: đệm 4, cắt dời, lật ngang, chỉ cho ảnh 3×32×32 (ADR-012).
AUGMENTATIONS = ("none", "crop_flip")
# Dữ liệu tổng hợp của các profile smoke: không bao giờ được ghi vào results/ (SPEC §5.3).
SYNTHETIC_DATASETS = ("synthetic", "synthetic_cifar10")
SPLITS = ("test", "validation")
RUNTIMES = ("inprocess", "processes")
DEPLOYS = ("local", "docker")
# Tham số của từng scheme chia dữ liệu (KH §3.2); tham số khác phải để null.
SCHEME_PARAMS = {"iid": (), "dirichlet": ("alpha",), "label_k": ("k",), "quantity": ("beta",), "shards": ()}


def _one_of(value: str, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise ValueError(f"{where} phải là một trong {allowed}, nhận {value!r}")


def _at_least(value: float, low: float, where: str) -> None:
    if not (math.isfinite(value) and value >= low):
        raise ValueError(f"{where} phải hữu hạn và ≥ {low}, nhận {value}")


def _shape_text(shape: tuple[int, ...]) -> str:
    return "×".join(str(size) for size in shape)


def _positive(value: float, where: str) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{where} phải hữu hạn và > 0, nhận {value}")


@dataclass(frozen=True)
class PartitionConfig:
    scheme: str = "iid"
    alpha: float | None = None
    k: int | None = None
    beta: float | None = None
    min_size: int = 64

    def __post_init__(self) -> None:
        _one_of(self.scheme, SCHEMES, "partition.scheme")
        for name in ("alpha", "k", "beta"):
            used = name in SCHEME_PARAMS[self.scheme]
            value = getattr(self, name)
            if used and value is None:
                raise ValueError(f"partition.{name} phải có với scheme {self.scheme}")
            if not used and value is not None:
                raise ValueError(f"partition.{name} chỉ dùng với scheme {self._owner(name)}; để null")
        if self.alpha is not None:
            _positive(self.alpha, "partition.alpha")
        if self.beta is not None:
            _positive(self.beta, "partition.beta")
        if self.k is not None and not 1 <= self.k <= 10:
            raise ValueError(f"partition.k phải nằm trong 1..10, nhận {self.k}")
        _at_least(self.min_size, 1, "partition.min_size")

    @staticmethod
    def _owner(name: str) -> str:
        return next(scheme for scheme, params in SCHEME_PARAMS.items() if name in params)


@dataclass(frozen=True)
class TopologyConfig:
    kind: str = "ring"
    p: float | None = None

    def __post_init__(self) -> None:
        _one_of(self.kind, TOPOLOGIES, "topology.kind")
        if self.kind in ER_KINDS:
            if self.p is None:
                object.__setattr__(
                    self, "p", DEFAULT_ER_P
                )  # dataclass frozen: tài liệu dataclasses chỉ cách này
            if not (math.isfinite(self.p) and 0 < self.p <= 1):
                raise ValueError(f"topology.p phải nằm trong (0, 1], nhận {self.p}")
        elif self.p is not None:
            raise ValueError(f"topology.p chỉ dùng với {ER_KINDS}; để null với {self.kind}")


@dataclass(frozen=True)
class AlgorithmConfig:
    tau: int = 20
    lr: float = 0.05
    mu: float = 0.0
    batch_size: int = 32

    def __post_init__(self) -> None:
        _at_least(self.tau, 1, "algorithm.tau")
        _positive(self.lr, "algorithm.lr")
        _at_least(self.mu, 0, "algorithm.mu")
        _at_least(self.batch_size, 1, "algorithm.batch_size")


@dataclass(frozen=True)
class EvalConfig:
    split: str = "test"
    subset_size: int = 2000
    # Mốc đánh giá theo bước cục bộ (ADR-007); None là đánh giá mỗi round (profile smoke).
    every_steps: int | None = 100

    def __post_init__(self) -> None:
        _one_of(self.split, SPLITS, "eval.split")
        _at_least(self.subset_size, 1, "eval.subset_size")
        if self.every_steps is not None:
            _at_least(self.every_steps, 1, "eval.every_steps")


@dataclass(frozen=True)
class RunConfig:
    """Mọi tham số ảnh hưởng tới kết quả của một lượt; mặc định theo KH §3.5 (configs/base.yaml ghi lại)."""

    dataset: str = "mnist"
    num_nodes: int = 20
    seed: int = 0
    # τ·R, giống nhau cho mọi điều kiện (ADR-006).
    local_steps: int = 4000
    model: str = "smallcnn"
    # null: crop_flip với ảnh 3×32×32 (CIFAR-10), none với dataset khác. Cấu hình chuẩn hóa ghi giá trị đã
    # điền (ADR-012), nên dataclasses.replace đổi dataset thì phải đặt lại trường này.
    augmentation: str | None = None
    mixing: str = "metropolis_hastings"
    # Số luồng mỗi lượt, hằng số của cả lưới (SPEC §5.2).
    threads: int = 1
    runtime: str = "inprocess"
    deploy: str = "local"
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        _one_of(self.dataset, DATASETS, "dataset")
        _one_of(self.model, MODELS, "model")
        _one_of(self.mixing, MIXINGS, "mixing")
        _one_of(self.runtime, RUNTIMES, "runtime")
        _one_of(self.deploy, DEPLOYS, "deploy")
        cifar_images = DATASET_INPUTS[self.dataset] == CIFAR10_INPUT
        if self.augmentation is None:
            # dataclass frozen: tài liệu dataclasses chỉ cách này
            object.__setattr__(self, "augmentation", "crop_flip" if cifar_images else "none")
        _one_of(self.augmentation, AUGMENTATIONS, "augmentation")
        if self.augmentation == "crop_flip" and not cifar_images:
            raise ValueError(
                f"augmentation crop_flip chỉ dùng với ảnh {_shape_text(CIFAR10_INPUT)} (cifar10, "
                "synthetic_cifar10): lật ngang làm sai nghĩa chữ số MNIST (ADR-012)"
            )
        if MODEL_INPUTS[self.model] != DATASET_INPUTS[self.dataset]:
            shape = DATASET_INPUTS[self.dataset]
            fitting = [name for name, accepted in MODEL_INPUTS.items() if accepted == shape]
            raise ValueError(
                f"model {self.model} nhận ảnh {_shape_text(MODEL_INPUTS[self.model])}, còn dataset "
                f"{self.dataset} có ảnh {_shape_text(shape)}; dùng một trong {fitting} (ADR-012)"
            )
        _at_least(self.num_nodes, 1, "num_nodes")
        _at_least(self.seed, 0, "seed")
        _at_least(self.local_steps, 1, "local_steps")
        _at_least(self.threads, 1, "threads")
        tau = self.algorithm.tau
        if self.local_steps % tau != 0:
            raise ValueError(
                f"local_steps = {self.local_steps} phải chia hết cho τ = {tau}, để R là số nguyên"
            )
        if self.eval.every_steps is not None and self.eval.every_steps % tau != 0:
            raise ValueError(
                f"eval.every_steps = {self.eval.every_steps} phải chia hết cho τ = {tau}, "
                "để mọi mốc rơi đúng cuối một round (ADR-007)"
            )
        if self.partition.scheme == "label_k" and self.num_nodes < 10:
            raise ValueError(f"#C = k cần N ≥ 10 để mọi nhãn có node sở hữu, nhận N = {self.num_nodes} (G23)")
        if self.deploy == "docker" and self.runtime != "processes":
            raise ValueError("deploy: docker chỉ dùng với runtime: processes (ADR-010)")
        if self.mixing == "size_weighted" and self.topology.kind not in ("fully_connected", "server"):
            raise ValueError(
                "mixing size_weighted chỉ dùng với topology fully_connected hay server (ADR-005)"
            )
        if self.topology.kind == "server" and self.mixing != "size_weighted":
            raise ValueError(
                "topology server (B3, E1c) tổng hợp theo n_k/n, nên mixing phải là size_weighted"
            )
        if (self.topology.kind == "central") != (self.num_nodes == 1):
            raise ValueError(
                "topology central (B1, B1-4k) là đúng một node: cần num_nodes: 1, và num_nodes: 1 chỉ dùng "
                f"với central, nhận {self.topology.kind} với N = {self.num_nodes}"
            )
        if self.topology.kind == "central" and self.partition.scheme != "iid":
            raise ValueError("topology central giữ đủ dữ liệu train, không chia; để partition.scheme: iid")
        if self.topology.kind == "central" and self.mixing != "metropolis_hastings":
            raise ValueError("topology central không trộn; để mixing: metropolis_hastings (mặc định)")
        if self.runtime == "processes" and self.topology.kind in ("server", "central"):
            raise ValueError(
                f"topology {self.topology.kind} chỉ có ở runtime inprocess: process runtime chạy các node "
                "DFL, không có server ảo hay baseline một node (ADR-010)"
            )
        if self.mixing == "size_corrected" and self.algorithm.mu != 0:
            raise ValueError(
                "mixing size_corrected chỉ dùng với mu = 0: c_i làm yếu tương đối số hạng prox ở node lớn "
                f"(ADR-005), nhận mu = {self.algorithm.mu}"
            )

    @property
    def rounds(self) -> int:
        return self.local_steps // self.algorithm.tau


SECTIONS = {
    "partition": PartitionConfig,
    "topology": TopologyConfig,
    "algorithm": AlgorithmConfig,
    "eval": EvalConfig,
}


def _is_number_text(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _coerce(value: object, annotation: str, where: str) -> object:
    """Kiểm kiểu một giá trị đọc từ YAML; số nguyên ghi cho trường số thực được đổi thành float."""
    optional = "None" in annotation
    base = annotation.split("|")[0].strip()
    if value is None:
        if optional:
            return None
        raise ValueError(f"{where} không được để trống")
    if isinstance(value, str) and base != "str" and _is_number_text(value):
        raise ValueError(
            f"{where} = {value!r} bị PyYAML đọc thành chuỗi; viết số có dấu chấm thập phân, ví dụ 0.05 "
            "hay 5.0e-2"
        )
    if base == "int":
        if type(value) is not int:
            raise ValueError(f"{where} phải là số nguyên, nhận {value!r}")
        return value
    if base == "float":
        if type(value) not in (int, float):
            raise ValueError(f"{where} phải là số, nhận {value!r}")
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"{where} phải là chuỗi, nhận {value!r} (PyYAML đọc yes/no/on/off thành bool)")
    return value


def _build[S](cls: type[S], data: object, where: str) -> S:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where or 'cấu hình'} phải là một mapping YAML, nhận {type(data).__name__}")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        prefix = f"{where}." if where else ""
        raise ValueError(f"khóa {[prefix + str(key) for key in unknown]} không có trong cấu hình")
    kwargs = {}
    for name, value in data.items():
        path = f"{where}.{name}" if where else name
        if name in SECTIONS and not where:
            kwargs[name] = _build(SECTIONS[name], value, name)
        else:
            kwargs[name] = _coerce(value, str(known[name].type), path)
    return cls(**kwargs)


def from_mapping(data: Mapping[str, object]) -> RunConfig:
    """Dựng và kiểm cấu hình từ một mapping (khóa thiếu dùng mặc định)."""
    return _build(RunConfig, data, "")


def _merge(base: dict[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _merge(dict(existing), value)
        else:
            merged[key] = value
    return merged


def _read(path: Path, chain: tuple[Path, ...]) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: cấu hình phải là một mapping YAML, nhận {type(data).__name__}")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ValueError(f"{path}: extends phải là đường dẫn tới file YAML")
    parent_path = (path.parent / parent).resolve()
    if parent_path in chain:
        raise ValueError(f"{path}: extends tạo thành vòng ({parent_path})")
    return _merge(_read(parent_path, (*chain, parent_path)), data)


def load_mapping(path: str | Path) -> dict[str, object]:
    """Mapping của một file cấu hình sau khi trộn chuỗi `extends`, chưa điền mặc định hay kiểm."""
    resolved = Path(path).resolve()
    return _read(resolved, (resolved,))


def merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    """Trộn như `extends`: mapping được trộn đệ quy theo từng khóa, giá trị khác thì bị thay."""
    return _merge(dict(base), override)


def load_config(path: str | Path) -> RunConfig:
    """Đọc một file cấu hình (có thể kế thừa qua `extends`), điền mặc định và kiểm."""
    return from_mapping(load_mapping(path))


def to_yaml(config: RunConfig) -> str:
    """Cấu hình đầy đủ dạng YAML cho tiến trình con (sweep, process runtime); load_config đọc lại đúng.

    yaml.safe_dump viết số thực có dấu chấm (1.0e-05), nên PyYAML không đọc nhầm thành chuỗi.
    """
    return yaml.safe_dump(json.loads(canonical_json(config)), sort_keys=True, allow_unicode=True)


def canonical_json(config: RunConfig) -> str:
    """JSON chuẩn hóa của cấu hình, đầu vào của run_id (ADR-008)."""
    return json.dumps(dataclasses.asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
