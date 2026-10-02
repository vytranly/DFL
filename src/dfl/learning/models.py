"""Model của node (module learning; ADR-004, ADR-011; SPEC §6; T-INV-5).

SmallCNN là model của mọi thí nghiệm lõi: Conv(1→16, 5×5) → ReLU → MaxPool 2 → Conv(16→32, 5×5) → ReLU →
MaxPool 2 → FC(512→64) → ReLU → FC(64→10), 46,730 tham số, không BatchNorm, không dropout (ADR-004).
Thang model của E8 (ADR-011), chọn bằng trường `model` của cấu hình; không model nào có BatchNorm hay dropout:
- M0 `logreg`: Linear 784 → 10, 7,850 tham số;
- M1 `smallcnn`: SmallCNN;
- M2 `mlp2nn`: 784 → 200 → 200 → 10 với ReLU, 199,210 tham số (2NN của McMahan et al. 2017);
- M3 `cnn_mcmahan`: Conv 5×5 (1→32, padding 2) → ReLU → MaxPool 2 → Conv 5×5 (32→64, padding 2) → ReLU →
  MaxPool 2 → FC 3136 → 512 → ReLU → FC 512 → 10, 1,663,370 tham số (CNN của McMahan et al. 2017).

w⁰ lấy mẫu từ luồng init (seeding.Stream.INIT), không từ RNG toàn cục của torch. Model được dựng bằng
`skip_init`, tức tham số chưa khởi tạo, rồi mỗi tham số được điền bằng `uniform_(…, generator=…)` theo đúng
phân bố mặc định của PyTorch, U(−√k, √k) với k = 1/fan_in. Nhờ vậy cùng seed thì cùng w⁰ ở mọi topology và
thuật toán (G9), và dựng model không làm lệch luồng ngẫu nhiên nào khác (T-INV-5). Thứ tự lấy mẫu là thứ tự
lớp trong model, weight trước bias; đổi thứ tự là đổi w⁰ của mọi seed.

torch.nn.utils.skip_init (module phải nhận `device` và chuyển nó cho mọi tham số):
https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.skip_init.html
Khởi tạo mặc định của Conv2d (k = groups/(C_in·∏kernel_size)) và Linear (k = 1/in_features):
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html
torch.Tensor.uniform_(from, to, *, generator): https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.uniform_.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.max_pool2d.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.relu.html
"""

from __future__ import annotations

import math

import torch

# Lớp có quy tắc khởi tạo; với cả hai, weight[0].numel() là fan_in (C_in·kh·kw hay in_features), groups = 1.
INITIALIZED_LAYERS = (torch.nn.Conv2d, torch.nn.Linear)


class SmallCNN(torch.nn.Module):
    """SmallCNN của ADR-004. Dựng bằng `make_smallcnn`; gọi thẳng lớp này thì PyTorch khởi tạo bằng RNG
    toàn cục."""

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, device=device, dtype=dtype)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, device=device, dtype=dtype)
        self.fc1 = torch.nn.Linear(32 * 4 * 4, 64, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(64, 10, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        return self.fc2(torch.nn.functional.relu(self.fc1(x.flatten(1))))


class LogisticRegression(torch.nn.Module):
    """M0 của ADR-011: hồi quy logistic trên 784 điểm ảnh."""

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, 10, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TwoLayerMLP(torch.nn.Module):
    """M2 của ADR-011: 2NN của McMahan et al. 2017."""

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 200, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(200, 200, device=device, dtype=dtype)
        self.fc3 = torch.nn.Linear(200, 10, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x.flatten(1)))
        return self.fc3(torch.nn.functional.relu(self.fc2(x)))


class McMahanCNN(torch.nn.Module):
    """M3 của ADR-011: CNN của McMahan et al. 2017; padding 2 giữ 28 × 28 qua mỗi lớp conv."""

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2, device=device, dtype=dtype)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2, device=device, dtype=dtype)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 512, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(512, 10, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        return self.fc2(torch.nn.functional.relu(self.fc1(x.flatten(1))))


# Tên trong cấu hình (config.MODELS) → lớp model.
MODEL_CLASSES: dict[str, type[torch.nn.Module]] = {
    "logreg": LogisticRegression,
    "smallcnn": SmallCNN,
    "mlp2nn": TwoLayerMLP,
    "cnn_mcmahan": McMahanCNN,
}


def init_from_stream(model: torch.nn.Module, generator: torch.Generator) -> None:
    """Điền mọi tham số bằng U(−√k, √k), k = 1/fan_in, lấy từ `generator` theo thứ tự lớp.

    Sau skip_init, tham số và buffer chứa bộ nhớ chưa khởi tạo, nên tham số hay buffer nào không có quy tắc
    ở đây sẽ làm w⁰ không tất định; khi đó raise ValueError thay vì để lọt.
    """
    initialized: set[int] = set()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, INITIALIZED_LAYERS):
                bound = 1 / math.sqrt(layer.weight[0].numel())
                for param in (layer.weight, layer.bias):
                    if param is not None:
                        param.uniform_(-bound, bound, generator=generator)
                        initialized.add(id(param))
    missing = [name for name, param in model.named_parameters() if id(param) not in initialized]
    missing += [name for name, _ in model.named_buffers()]
    if missing:
        raise ValueError(f"không có quy tắc khởi tạo cho {missing}: sau skip_init chúng chứa bộ nhớ rác")


def _build[M: torch.nn.Module](cls: type[M], generator: torch.Generator) -> M:
    model = torch.nn.utils.skip_init(cls)
    init_from_stream(model, generator)
    return model


def build_model(name: str, generator: torch.Generator) -> torch.nn.Module:
    """Model `name` (config.MODELS), float32 trên CPU, w⁰ lấy từ `generator` (luồng init của seeding.py)."""
    if name not in MODEL_CLASSES:
        raise ValueError(f"model phải là một trong {sorted(MODEL_CLASSES)}, nhận {name!r}")
    return _build(MODEL_CLASSES[name], generator)


def make_smallcnn(generator: torch.Generator) -> SmallCNN:
    """SmallCNN, model của thí nghiệm lõi; cho cùng w⁰ như build_model("smallcnn", generator)."""
    return _build(SmallCNN, generator)
