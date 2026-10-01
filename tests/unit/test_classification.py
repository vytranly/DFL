"""Test cho src/dfl/metrics/classification.py: độ đo từ ma trận nhầm lẫn và Evaluator.

T09b; KH §3.6; T-INV-10; INV-5.

Oracle là scikit-learn 1.9 (chỉ có trong nhóm dev, SPEC §7):
https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.f1_score.html
https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.accuracy_score.html
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score

from dfl.contracts import EvalResult, Evaluator, StateDict
from dfl.metrics import classification as clf

LABELS = list(range(10))
N = 3
FEATURES = 6


def labels(seed: int, n: int = 1000) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 10, n)


def random_predictions() -> tuple[np.ndarray, np.ndarray]:
    return labels(0), labels(1)


def single_class_predictions() -> tuple[np.ndarray, np.ndarray]:
    return labels(2), np.full(1000, 3)


def predictions_never_class_7() -> tuple[np.ndarray, np.ndarray]:
    y_pred = labels(4)
    y_pred[y_pred == 7] = 0
    return labels(3), y_pred


def class_9_absent_everywhere() -> tuple[np.ndarray, np.ndarray]:
    # TP = FP = FN = 0 cho lớp 9: sklearn trả zero_division = 0 và vẫn tính lớp đó vào trung bình.
    return labels(5) % 9, labels(6) % 9


@pytest.mark.parametrize(
    "case",
    [random_predictions, single_class_predictions, predictions_never_class_7, class_9_absent_everywhere],
    ids=["random", "single-class", "never-7", "absent-9"],
)
def test_t_inv_10_macro_f1_matches_sklearn(case: Callable[[], tuple[np.ndarray, np.ndarray]]) -> None:
    # Arrange
    y_true, y_pred = case()

    # Act
    ours = clf.macro_f1(clf.confusion_matrix(y_true, y_pred))

    # Assert
    expected = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    assert np.isfinite(ours)
    assert ours == pytest.approx(expected, abs=1e-12)


def test_t_inv_10_acc_local_and_acc_eff_match_per_sample_weighted_accuracy() -> None:
    # Arrange: dự đoán đúng khoảng 70%; P_i và P_eff = Σ_j π_j P_j lệch hẳn khỏi phân bố của tập đánh giá.
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 10, 2000)
    y_pred = np.where(rng.random(2000) < 0.7, y_true, rng.integers(0, 10, 2000))
    p_local = rng.dirichlet(np.full(10, 0.3))
    p_eff = rng.dirichlet(np.ones(5)) @ rng.dirichlet(np.full(10, 0.5), size=5)
    p_eval = np.bincount(y_true, minlength=10) / len(y_true)
    confusion = clf.confusion_matrix(y_true, y_pred)

    for weights in (p_local, p_eff):
        # Act
        ours = clf.class_weighted_recall(confusion, weights)

        # Assert: Σ_c q_c recall_c = accuracy với trọng số từng mẫu q(y)/P_eval(y) (KH §3.6).
        expected = accuracy_score(y_true, y_pred, sample_weight=weights[y_true] / p_eval[y_true])
        assert ours == pytest.approx(expected, abs=1e-12)


def test_class_weighted_recall_rejects_weight_on_a_class_missing_from_the_eval_set() -> None:
    confusion = clf.confusion_matrix(labels(8) % 9, labels(9) % 9)

    with pytest.raises(ValueError, match="lớp"):
        clf.class_weighted_recall(confusion, np.full(10, 0.1))


def tiny_model(seed: int) -> torch.nn.Module:
    model = torch.nn.utils.skip_init(torch.nn.Linear, FEATURES, 10)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator))
    return model


def eval_set(n: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((n, FEATURES), generator=torch.Generator().manual_seed(100))
    return x, torch.arange(n) % 10


def node_states() -> list[StateDict]:
    return [tiny_model(seed).state_dict() for seed in range(N)]


def label_distributions() -> np.ndarray:
    return np.random.default_rng(11).dirichlet(np.full(10, 0.5), size=N)


def make_evaluator(model: torch.nn.Module | None = None, **kwargs: object) -> clf.ClassificationEvaluator:
    x, y = eval_set()
    options = {"label_distributions": label_distributions(), "objective_weights": np.full(N, 1 / N), **kwargs}
    return clf.ClassificationEvaluator(tiny_model(99) if model is None else model, x, y, **options)


def predictions(state: StateDict, x: torch.Tensor) -> np.ndarray:
    oracle = tiny_model(0)
    oracle.load_state_dict(state)
    with torch.no_grad():
        return oracle(x).argmax(dim=1).numpy()


def test_evaluator_matches_sklearn_on_each_node_model() -> None:
    # Arrange
    x, y = eval_set()
    y_np = y.numpy()
    p_local = label_distributions()
    pi = np.array([0.2, 0.3, 0.5])
    p_eval = np.bincount(y_np, minlength=10) / len(y_np)
    states = node_states()
    evaluator = make_evaluator(objective_weights=pi)

    # Act
    result = evaluator.evaluate(states)

    # Assert
    assert isinstance(evaluator, Evaluator)
    assert isinstance(result, EvalResult)
    p_eff = pi @ p_local
    for i, state in enumerate(states):
        pred = predictions(state, x)
        assert result.acc[i] == pytest.approx(accuracy_score(y_np, pred), abs=1e-12)
        f1 = f1_score(y_np, pred, average="macro", labels=LABELS, zero_division=0)
        assert result.f1_macro[i] == pytest.approx(f1, abs=1e-12)
        local = accuracy_score(y_np, pred, sample_weight=p_local[i][y_np] / p_eval[y_np])
        assert result.acc_local[i] == pytest.approx(local, abs=1e-12)
        eff = accuracy_score(y_np, pred, sample_weight=p_eff[y_np] / p_eval[y_np])
        assert result.acc_eff[i] == pytest.approx(eff, abs=1e-12)
    assert len(set(result.acc)) > 1  # không suông: ba model dự đoán khác nhau


def test_average_model_accuracy_is_the_accuracy_of_the_elementwise_mean_state() -> None:
    x, y = eval_set()
    states = node_states()
    mean_state = {name: torch.stack([state[name] for state in states]).mean(dim=0) for name in states[0]}

    result = make_evaluator().evaluate(states)

    assert result.acc_avg_model == pytest.approx(
        accuracy_score(y.numpy(), predictions(mean_state, x)), abs=1e-12
    )


def test_batching_does_not_change_the_result() -> None:
    states = node_states()

    assert make_evaluator(batch_size=7).evaluate(states) == make_evaluator(batch_size=1000).evaluate(states)


def test_evaluator_reads_states_without_changing_them_or_the_global_rng() -> None:
    # Arrange
    states = node_states()
    before = [{name: t.clone() for name, t in state.items()} for state in states]
    rng_before = torch.get_rng_state()

    # Act
    make_evaluator().evaluate(states)

    # Assert: state và RNG toàn cục không đổi (INV-5)
    for state, copy in zip(states, before, strict=True):
        for name in state:
            assert torch.equal(state[name], copy[name])
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_evaluator_loads_states_into_the_preallocated_model() -> None:
    # Arrange
    model = tiny_model(99)
    params = list(model.parameters())
    pointers = [param.data_ptr() for param in params]
    initial = model.weight.detach().clone()

    # Act
    make_evaluator(model).evaluate(node_states())

    # Assert: chính model được giao đã nhận state (không phải một bản sao), vào đúng bộ nhớ cũ.
    assert not torch.equal(model.weight, initial)
    assert all(a is b for a, b in zip(model.parameters(), params, strict=True))
    assert [param.data_ptr() for param in model.parameters()] == pointers


def test_evaluate_rejects_a_wrong_number_of_states() -> None:
    with pytest.raises(ValueError, match="N"):
        make_evaluator().evaluate(node_states()[:2])


@pytest.mark.parametrize(
    ("kwargs", "eval_labels"),
    [
        ({"label_distributions": np.full((N, 10), 0.2)}, None),  # hàng không cộng thành 1
        ({"objective_weights": np.array([0.5, 0.5, 0.5])}, None),
        ({"objective_weights": np.array([1.5, -0.25, -0.25])}, None),
        ({"objective_weights": np.full(N + 1, 1 / (N + 1))}, None),
        ({}, torch.arange(300) % 9),  # tập đánh giá thiếu lớp 9
        ({}, torch.arange(300) % 11),  # nhãn ngoài 0..9
    ],
    ids=["p-rows", "pi-sum", "pi-negative", "pi-shape", "missing-class", "label-range"],
)
def test_evaluator_rejects_invalid_inputs(
    kwargs: dict[str, object], eval_labels: torch.Tensor | None
) -> None:
    x, y = eval_set()
    options = {"label_distributions": label_distributions(), "objective_weights": np.full(N, 1 / N), **kwargs}

    with pytest.raises(ValueError):
        clf.ClassificationEvaluator(tiny_model(99), x, y if eval_labels is None else eval_labels, **options)
