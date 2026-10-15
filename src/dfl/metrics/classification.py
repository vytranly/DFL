"""Độ đo phân loại từ ma trận nhầm lẫn và Evaluator chỉ đọc (module metrics-io; KH §3.6; ADR-007; INV-5).

Mọi độ đo của một model lấy từ ma trận nhầm lẫn C trên tập đánh giá (C[y, ŷ] = số mẫu nhãn y bị đoán là ŷ):
- Acc = trace(C) / n.
- Acc^loc_i = Σ_c P_i(c) · recall_c và Acc^eff = Σ_c P_eff(c) · recall_c, với P_eff = Σ_j π_j P_j. Hai đại
  lượng bằng accuracy có trọng số từng mẫu P_i(y)/P_eval(y), tương ứng P_eff(y)/P_eval(y) (T-INV-10).
- Macro-F1 = trung bình trên đủ 10 lớp của 2TP_c / (2TP_c + FP_c + FN_c); lớp có mẫu số 0 cho 0, như
  `f1_score(average="macro", labels=range(10), zero_division=0)` của scikit-learn 1.9:
  https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.f1_score.html

Evaluator nạp từng state vào một model cấp phát sẵn bằng load_state_dict (không tạo module mới, không import
`learning`, ENG-7), chạy dưới torch.no_grad và không dùng nguồn ngẫu nhiên nào (INV-5). argmax trả về chỉ số
đầu tiên khi có nhiều giá trị lớn nhất bằng nhau, nên dự đoán tất định.
torch.nn.Module.load_state_dict:
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
torch.nn.Module.eval: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval
torch.no_grad: https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
torch.argmax: https://docs.pytorch.org/docs/2.14/generated/torch.argmax.html
numpy.bincount: https://numpy.org/doc/2.5/reference/generated/numpy.bincount.html
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from dfl.contracts import EvalResult, StateDict

NUM_CLASSES = 10
# Sai số cho phép khi kiểm một phân bố có tổng bằng 1.
DISTRIBUTION_ATOL = 1e-9


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """C[y, ŷ], shape (10, 10), kiểu int64."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    # Độ dài lệch thì broadcast im lặng, nhãn ngoài 0..9 thì rơi sang ô của lớp khác (review C3 P3.7, P2-12).
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true và y_pred phải là mảng 1-D cùng độ dài, nhận {y_true.shape} và {y_pred.shape}"
        )
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if len(values) and (values.min() < 0 or values.max() >= NUM_CLASSES):
            raise ValueError(f"{name} phải nằm trong 0..{NUM_CLASSES - 1}")
    counts = np.bincount(y_true * NUM_CLASSES + y_pred, minlength=NUM_CLASSES * NUM_CLASSES)
    return counts.reshape(NUM_CLASSES, NUM_CLASSES)


def accuracy(confusion: np.ndarray) -> float:
    return float(np.trace(confusion) / confusion.sum())


def class_weighted_recall(confusion: np.ndarray, class_weights: np.ndarray) -> float:
    """Σ_c q_c · TP_c / support_c: Acc^loc với q = P_i, Acc^eff với q = P_eff."""
    support = confusion.sum(axis=1)
    weights = np.asarray(class_weights, dtype=float)
    if ((weights > 0) & (support == 0)).any():
        raise ValueError("có lớp mang trọng số dương nhưng không có mẫu nào trong tập đánh giá")
    recall = np.divide(np.diag(confusion), support, out=np.zeros(NUM_CLASSES), where=support > 0)
    return float(weights @ recall)


def macro_f1(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(float)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)  # = 2TP + FP + FN
    f1 = np.divide(2 * tp, denominator, out=np.zeros(NUM_CLASSES), where=denominator > 0)
    return float(f1.mean())


def _check_distribution(values: np.ndarray, what: str) -> None:
    if (values < 0).any() or not np.allclose(values.sum(axis=-1), 1.0, rtol=0.0, atol=DISTRIBUTION_ATOL):
        raise ValueError(f"{what} phải không âm và có tổng bằng 1")


class ClassificationEvaluator:
    """Evaluator của contracts.py trên một tập đánh giá cố định (KH §3.6; ADR-007).

    `label_distributions` là P_i (N × 10, phân bố nhãn huấn luyện của từng node); `objective_weights` là π
    (N phần tử) của lượt chạy, từ mixing.objective_weights (ADR-005).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        label_distributions: np.ndarray,
        objective_weights: np.ndarray,
        batch_size: int = 1000,
    ) -> None:
        p_local = np.array(label_distributions, dtype=float)
        pi = np.array(objective_weights, dtype=float)
        if p_local.ndim != 2 or p_local.shape[1] != NUM_CLASSES or pi.shape != (p_local.shape[0],):
            raise ValueError("label_distributions phải có shape (N, 10) và objective_weights shape (N,)")
        _check_distribution(p_local, "mỗi hàng của label_distributions")
        _check_distribution(pi, "objective_weights")
        labels = y.detach().cpu().numpy().astype(np.int64)
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            raise ValueError("nhãn của tập đánh giá phải nằm trong 0..9")
        if np.bincount(labels, minlength=NUM_CLASSES).min() == 0:
            raise ValueError("tập đánh giá phải có đủ 10 lớp, để recall của mọi lớp xác định")
        self._model = model.eval()
        self._x = x
        self._labels = labels
        self._p_local = p_local
        self._p_eff = pi @ p_local
        self._batch_size = batch_size
        self._dtype = next(model.parameters()).dtype

    def evaluate(self, states: Sequence[StateDict]) -> EvalResult:
        if len(states) != len(self._p_local):
            raise ValueError(f"cần đúng N = {len(self._p_local)} state, nhận {len(states)}")
        confusions = [self._confusion(state) for state in states]
        average = {
            name: torch.stack([state[name] for state in states]).mean(dim=0)
            if tensor.is_floating_point()
            else tensor
            for name, tensor in states[0].items()
        }
        return EvalResult(
            acc=tuple(accuracy(c) for c in confusions),
            acc_local=tuple(
                class_weighted_recall(c, p) for c, p in zip(confusions, self._p_local, strict=True)
            ),
            acc_eff=tuple(class_weighted_recall(c, self._p_eff) for c in confusions),
            f1_macro=tuple(macro_f1(c) for c in confusions),
            acc_avg_model=accuracy(self._confusion(average)),
        )

    def _confusion(self, state: StateDict) -> np.ndarray:
        self._model.load_state_dict(state)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(self._labels), self._batch_size):
                batch = self._x[start : start + self._batch_size].to(self._dtype)
                predictions.append(self._model(batch).argmax(dim=1))
        return confusion_matrix(self._labels, torch.cat(predictions).cpu().numpy())
