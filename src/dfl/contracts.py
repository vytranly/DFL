"""Hợp đồng giao diện giữa các module của dfl (P1.7; SPEC §3; KH §4.2, §4.4, §4.5, §4.7).

File này chỉ gồm Protocol, TypedDict và dataclass, không có logic. Module cài đặt từng hợp đồng
(capability map, SPEC §3.1):

    Partitioner, PartitionResult         data        data/partition.py
    NodeDataset                          data        data/datasets.py
    TopologySchedule, MixingPolicy       network     network/topology.py, network/mixing.py
    Transport, Snapshot, CommCounters    network     network/transport.py
    LocalTrainer                         learning    learning/local_train.py
    Evaluator, EvalResult                metrics-io  metrics/classification.py
    RunLogger và các bản ghi *Record     metrics-io  io/run_logger.py, io/manifest.py (analysis chỉ đọc)

Quy ước chung:
- Node được đánh số 0..N-1; mọi dãy "theo node" có đúng N phần tử, xếp theo id.
- File này không import torch, numpy hay networkx lúc chạy (chỉ dưới TYPE_CHECKING), để các module thuần
  numpy import được nó mà không nạp torch (SPEC §3.2, ENG-6).
- Lỗi: đầu vào hay cấu hình sai thì raise ValueError, nói rõ ràng buộc bị vi phạm; gọi sai thứ tự thì raise
  RuntimeError; ghi đè kết quả đã có thì raise FileExistsError. Không trả None hay giá trị đặc biệt thay
  cho lỗi.
- Schema bản ghi nằm trong src/, nên đổi schema là đổi code_version và mọi kết quả cũ thành stale
  (ADR-008). Sau khi đã có kết quả Tier 1, đổi schema phải hỏi trước (SPEC §8).
- Bản ghi không bao giờ chứa NaN hay vô cực: lượt phân kỳ dừng lại và ghi SummaryDiverged (SPEC §5.3).
- API framework được đối chiếu ngày 2026-09-30 với tài liệu chính thức đúng phiên bản trong pyproject.toml
  (torch 2.14.0, torchvision 0.29.0, networkx 3.7, numpy 2.5.3) và với gói đã cài trong .venv; URL ghi ngay
  tại chỗ dùng. Chỗ không kiểm được ghi "UNVERIFIED".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    import torch

# Tham số và buffer của model, dạng tên → tensor, như torch.nn.Module.state_dict() trả về:
# https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict
type StateDict = dict[str, torch.Tensor]


# ------------------------------------------------------------------------------------------------ data


@dataclass(frozen=True, slots=True)
class PartitionResult:
    """Kết quả chia dữ liệu cho N node (KH §3.2); `retries` được ghi vào manifest."""

    # N mảng chỉ số int64 trỏ vào `labels`: rời nhau, hợp lại phủ đủ, mỗi mảng có ít nhất min_size phần tử.
    parts: tuple[np.ndarray, ...]
    # Số lần phải lấy mẫu lại (có node nhỏ hơn min_size, hoặc mọi xác suất còn lại bằng 0).
    retries: int


@runtime_checkable
class Partitioner(Protocol):
    """Chia chỉ số mẫu train cho N node theo một trong các cách chia P0–P3 (module data; KH §3.2).

    Tham số của cách chia (α, k, β, min_size) nằm trong đối tượng cài đặt, không nằm trong lời gọi.
    Hoán vị vị trí node (placement) là một bước riêng, dùng luồng ngẫu nhiên khác.
    numpy.ndarray: https://numpy.org/doc/2.5/reference/generated/numpy.ndarray.html
    numpy.random.Generator: https://numpy.org/doc/2.5/reference/random/generator.html
    """

    def __call__(self, labels: np.ndarray, num_nodes: int, rng: np.random.Generator) -> PartitionResult:
        """Chia `labels` cho đúng `num_nodes` node (T-INV-8).

        `labels` là mảng 1-D các nhãn 0..9 của tập train (55k mẫu, sau khi đã tách validation).
        Chỉ dùng `rng` (luồng "partition" của seeding.py): cùng trạng thái `rng` thì cùng kết quả, và thứ tự
        tiêu thụ số ngẫu nhiên giống hàm tham chiếu trong tests/reference/ (test golden).
        Lấy mẫu lại tối đa 1,000 lần, quá số đó thì raise ValueError. Cũng raise ValueError khi cấu hình không
        thực hiện được, ví dụ #C = k với N < 10 (G23).
        """
        ...


@dataclass(frozen=True, slots=True)
class NodeDataset:
    """Dữ liệu cục bộ D_i của một node, đã nằm sẵn trong RAM (module data tạo, module learning dùng).

    Ảnh lấy từ torchvision.datasets.MNIST; Fashion-MNIST có cùng tham số khởi tạo:
    https://docs.pytorch.org/vision/0.29/generated/torchvision.datasets.MNIST.html
    https://docs.pytorch.org/vision/0.29/generated/torchvision.datasets.FashionMNIST.html
    UNVERIFIED: hai thuộc tính .data và .targets không có trong trang tài liệu 0.29, chỉ có trong mã nguồn
    đã cài; task nào dùng chúng phải ghi rõ điều này.
    """

    node_id: int
    # float32, shape (n_i, 1, 28, 28), đã chuẩn hóa bằng mean/std của tập train.
    x: torch.Tensor
    # int64, shape (n_i,), nhãn 0..9.
    y: torch.Tensor


# ------------------------------------------------------------------------------------------------ network


@runtime_checkable
class TopologySchedule(Protocol):
    """Đồ thị giao tiếp theo từng round, cho T0–T5 và T3b (module network; KH §3.3; ADR-009).

    networkx.Graph: https://networkx.org/documentation/networkx-3.7/reference/classes/graph.html
    Generator dựng đồ thị tĩnh và hàm kiểm tra liên thông (NetworkX 3.7):
    https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.classic.cycle_graph.html
    https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.classic.complete_graph.html
    https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html
    https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.random_graphs.random_regular_graph.html
    https://networkx.org/documentation/networkx-3.7/reference/algorithms/generated/networkx.algorithms.components.is_connected.html
    """

    # Số node N; node được đánh số 0..N-1.
    num_nodes: int
    # True nếu mọi round dùng cùng một đồ thị (ring, ER, FC, k-regular, đồ thị mũ); False với gossip.
    is_static: bool

    def graph_at(self, round_idx: int) -> nx.Graph:
        """Đồ thị vô hướng, không đa cạnh, của round `round_idx`, với node 0..N-1 và không có self-loop.

        Là hàm thuần của (seed, round_idx): gọi thêm bao nhiêu lần, theo thứ tự nào, cũng không làm đổi
        lịch (T-INV-12, G18). Topology tĩnh luôn trả về cùng một đồ thị. Gossip lấy ngẫu nhiên từ luồng
        gossip của seeding.py, tức numpy.random.SeedSequence với spawn_key (ID_GOSSIP, round_idx) cho T3 và
        (ID_GOSSIP_ER, round_idx) cho T3b, không dùng chung entropy với luồng nào khác (ADR-009):
        https://numpy.org/doc/2.5/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
        Đồ thị tĩnh phải liên thông; gossip liên thông theo hợp các round (INV-4). Ngoại lệ duy nhất là
        baseline B2 (W = I). Người gọi không được sửa đồ thị trả về.
        """
        ...


@runtime_checkable
class MixingPolicy(Protocol):
    """Ma trận trộn W^(t) cho đồ thị của một round (module network; KH §3.3; ADR-005)."""

    def __call__(self, graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
        """Trả về ma trận (N, N) float64 thỏa INV-4.

        Không âm, tổng mỗi hàng bằng 1, W[i, j] = 0 khi i != j và (i, j) không là cạnh; MH còn đối xứng và
        ngẫu nhiên kép. `node_sizes` là n_i của từng node, chỉ size-weighted dùng tới. Baseline B2 dùng
        W = I (ngoại lệ duy nhất của INV-4).
        Raise ValueError nếu đồ thị có hướng, có đa cạnh, có self-loop, hoặc node không được đánh số 0..N-1.
        """
        ...


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Bản sao sở hữu riêng của w_i^(t+1/2), là payload duy nhất đi qua Transport (KH §3.4, bước 2).

    Mỗi node tạo đúng một Snapshot mỗi round bằng cách clone + detach từng tensor. Không dùng thẳng kết quả
    của state_dict(): tài liệu PyTorch 2.14 ghi nó "contains references to the module's parameters and
    buffers" (G22):
    https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict
    Cùng một đối tượng được gửi tới mọi láng giềng. Người nhận chỉ đọc `state` và ghi kết quả tổng hợp ra
    bộ đệm mới, không sửa snapshot tại chỗ (T-INV-4).
    """

    sender: int
    round_idx: int
    state: StateDict
    # Số byte tính phí, tính từ tensor thật lúc tạo snapshot: tổng numel × element_size (SmallCNN float32:
    # 4 × 46,730). Transport chỉ đọc trường này, không đọc `state`.
    nbytes: int


@dataclass(frozen=True, slots=True)
class CommCounters:
    """Số đếm tích lũy của CommLedger từ đầu lượt chạy (KH §3.6)."""

    messages_cum: int
    bytes_cum: int
    # Lớn nhất theo node của (byte gửi + byte nhận); với đường server của B3/E1c thì tính cả server ảo.
    max_node_bytes_cum: int


@runtime_checkable
class Transport(Protocol):
    """Kênh message-passing giữa các node (module network; ADR-002; INV-1, INV-2).

    Bản mặc định chạy in-memory và không copy payload. B3/E1c đi qua đường server của engine: server ảo có
    chỉ số N (không có dữ liệu) gửi và nhận qua chính Transport này, nên CommLedger đếm đúng N message lên
    và N message xuống mỗi round, không cần biến thể riêng (ADR-005).
    Tùy chọn T22: TorchDistTransport dùng torch.distributed với backend gloo, cùng giao diện nhưng đặt ở
    dfl.sim vì phải import torch (ENG-6); bảng backend của tài liệu 2.14 ghi gloo hỗ trợ send/recv trên CPU:
    https://docs.pytorch.org/docs/2.14/distributed.html
    """

    def send(self, dst: int, message: Snapshot) -> None:
        """Đưa `message` vào inbox của `dst` cho round `message.round_idx`.

        Đây là nơi duy nhất cộng số đếm: 1 message và `message.nbytes` byte (INV-2), tính vào tải của cả người
        gửi và người nhận. Không copy payload. `dst` nằm trong 0..N-1, hoặc là N (server ảo) ở đường server.
        Raise ValueError khi `dst == message.sender` (không có message node tự gửi cho mình, G7).
        """
        ...

    def inbox(self, dst: int, round_idx: int) -> Mapping[int, Snapshot]:
        """Trả về các snapshot gửi tới `dst` trong round `round_idx`, khóa là id người gửi.

        Node chỉ được đọc inbox của chính nó (INV-1). Đọc inbox không làm đổi số đếm.
        """
        ...

    def end_round(self, round_idx: int) -> None:
        """Bỏ mọi message của round `round_idx` để giải phóng tham chiếu tới snapshot; số đếm giữ nguyên."""
        ...

    def counters(self) -> CommCounters:
        """Trả về số đếm tích lũy tới thời điểm gọi."""
        ...


# ------------------------------------------------------------------------------------------------ learning


@runtime_checkable
class LocalTrainer(Protocol):
    """Bước Adapt của vòng ATC (module learning; KH §3.4; ADR-004, ADR-006).

    τ (số bước), η, μ và B nằm trong đối tượng cài đặt.
    torch.nn.Module: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html
    torch.Tensor: https://docs.pytorch.org/docs/2.14/tensors.html
    torch.Generator: https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html
    """

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        """Chạy τ bước SGD tại chỗ trên `model`, từ w_i^t tới w_i^(t+1/2); trả về mất mát trung bình.

        Mỗi bước lấy B mẫu có hoàn lại từ `data` bằng `generator` (luồng batch riêng của node), ví dụ qua
        torch.randint(..., generator=generator):
        https://docs.pytorch.org/docs/2.14/generated/torch.randint.html
        Mất mát là `loss_scale` × cross-entropy + μ/2 · ‖w − anchor‖², với anchor là bản clone + detach
        của tham số lúc bắt đầu gọi và không có gradient (G6). `loss_scale` là c_i: bằng 1 theo mặc định,
        bằng n_i/n̄ với biến thể size-corrected (ADR-005). Chỉ dùng `model`, `data` và `generator`; không
        truy cập node khác, Transport hay topology, và không gọi RNG toàn cục của torch (T-INV-5). Khi mất mát
        của một bước không hữu hạn thì dừng ngay và trả về đúng giá trị đó, để engine dừng lượt và ghi
        SummaryDiverged trước khi ghi bản ghi nào của round này.
        """
        ...


# ------------------------------------------------------------------------------------------------ metrics-io


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Kết quả đánh giá mọi node trên một tập đánh giá cố định, sau tổng hợp (KH §3.6; ADR-007).

    Mỗi dãy có N phần tử theo id node. Drift (trước tổng hợp) và consensus distance (sau tổng hợp) là
    hàm thuần trong metrics/consensus.py, không thuộc Evaluator.
    """

    # Acc_i trên tập đánh giá chung.
    acc: tuple[float, ...]
    # Acc^loc_i: recall theo lớp của model node i, lấy trọng số theo phân bố nhãn huấn luyện của node i.
    acc_local: tuple[float, ...]
    # Acc^eff của model node i: recall theo lớp lấy trọng số theo P_eff = Σ_j π_j P_j (π: objective_weights).
    acc_eff: tuple[float, ...]
    # Macro-F1 trên đủ 10 lớp của model node i: trung bình của 2TP/(2TP + FP + FN) (Q3).
    f1_macro: tuple[float, ...]
    # Accuracy của model trung bình w̄; chỉ để chẩn đoán, không bao giờ đưa vào huấn luyện (INV-1).
    acc_avg_model: float


@runtime_checkable
class Evaluator(Protocol):
    """Đánh giá model của các node trên một tập đánh giá cố định (module metrics-io; KH §3.6; INV-5).

    Tập đánh giá theo cấu hình: tập test (mọi nhóm thí nghiệm) hoặc tập validation (E0a, chọn lr).
    Learning curve dùng tập con phân tầng cố định 2,000 ảnh; đánh giá cuối dùng đủ tập (10,000 ảnh test
    hoặc 5,000 ảnh validation). Đó là hai đối tượng riêng. Tập con lấy từ luồng dữ liệu cố định của
    seeding.py, không từ seed của lượt, nên mọi lượt dùng cùng ảnh (SPEC §6). Phân bố nhãn P_i và trọng số π
    được truyền vào lúc tạo đối tượng.
    """

    def evaluate(self, states: Sequence[StateDict]) -> EvalResult:
        """Đánh giá mọi node; `states[i]` là tham số của node i.

        Chỉ đọc (INV-5): không sửa `states` và không tiêu thụ RNG của huấn luyện. Nạp lần lượt từng state
        vào một model đánh giá cấp phát sẵn bằng torch.nn.Module.load_state_dict, không khởi tạo model mới:
        https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
        """
        ...


class RoundRecord(TypedDict):
    """Một dòng của rounds.jsonl: một lần đánh giá theo lịch (KH §3.5), sau tổng hợp của round `round`.

    Đánh giá trên tập đánh giá của cấu hình (test, hoặc validation với E0a); mỗi round có tối đa một dòng.
    Tất định (INV-6), không chứa thời gian. std tính qua N node với ddof = 0; min là node kém nhất.
    Độ đo suy ra (R@x, AUC, độ dao động, mức sụt) không nằm ở đây mà do analysis/ tính (ADR-008).
    """

    # Số round đã tổng hợp xong, tức model w^(round); local_steps = round × τ.
    round: int
    local_steps: int
    acc_mean: float
    acc_std: float
    acc_min: float
    acc_max: float
    acc_local_mean: float
    acc_local_std: float
    acc_local_min: float
    f1_macro_mean: float
    f1_macro_std: float
    f1_macro_min: float
    acc_avg_model: float
    acc_eff_mean: float
    # Ξ_(t+1/2): khoảng cách RMS tới trung bình, trên các w^(t+1/2) của round này (trước tổng hợp).
    drift_pre_combine: float
    # Ξ_(t+1): khoảng cách RMS tới trung bình, trên các w^(t+1) (sau tổng hợp).
    consensus_distance: float
    messages_cum: int
    bytes_cum: int
    max_node_bytes_cum: int
    # Trung bình qua node của mất mát huấn luyện trung bình τ bước của round này.
    train_loss_mean: float
    # N phần tử theo id node; bắt buộc có, để tính độ ổn định theo từng node (G24).
    acc_per_node: list[float]
    acc_local_per_node: list[float]


class TimingRecord(TypedDict):
    """Một dòng của timing.jsonl, ghi cùng lúc với mỗi dòng rounds.jsonl (cùng `round`).

    Thời gian các pha tính từ lần ghi trước; `wall_time_s` tính từ đầu lượt chạy. Không tất định, nên
    tách khỏi rounds.jsonl (INV-6, G21). Ở process runtime, train_s, combine_s và comm_s là lớn nhất qua node.
    """

    round: int
    wall_time_s: float
    train_s: float
    combine_s: float
    eval_s: float
    # Thời gian chờ gửi và nhận snapshot trong các round từ lần ghi trước, lớn nhất qua node; chỉ có ở process
    # runtime, bằng 0 ở runtime inprocess vì Transport trong bộ nhớ không chờ (ADR-010; Q13).
    comm_s: float


class SummaryOk(TypedDict):
    """summary.json của một attempt chạy hết ngân sách (ADR-008); các trường final_* là đánh giá cuối."""

    status: Literal["ok"]
    messages_total: int
    bytes_total: int
    # Tập của đánh giá cuối: "test" (10,000 ảnh) cho mọi nhóm thí nghiệm, "validation" (5,000 ảnh) cho E0a.
    final_eval_split: Literal["validation", "test"]
    final_eval_size: int
    final_acc_per_node: list[float]
    final_acc_local_per_node: list[float]
    final_f1_macro_per_node: list[float]
    final_acc_avg_model: float


class SummaryDiverged(TypedDict):
    """summary.json của một attempt phân kỳ: mất mát hay tham số không còn hữu hạn (SPEC §5.3).

    Phân kỳ là kết quả tất định, không phải lỗi: attempt được tính là xong, không chạy lại, và phải được báo
    cáo ("k/n seed phân kỳ"), không được lọc bỏ khỏi thống kê.
    """

    status: Literal["diverged"]
    # Round đầu tiên có giá trị không hữu hạn; rounds.jsonl chỉ có các dòng trước round này.
    diverged_at_round: int
    # Đại lượng nào không hữu hạn, ở node nào.
    reason: str
    messages_total: int
    bytes_total: int


class SummaryFailed(TypedDict):
    """summary.json của một attempt lỗi; attempt lỗi được giữ lại, không xóa (ADR-008)."""

    status: Literal["failed"]
    # Loại lỗi và thông báo lỗi.
    error: str
    messages_total: int
    bytes_total: int


# summary.json là một trong ba biến thể, phân biệt bằng trường `status`. Attempt không có summary.json
# (tiến trình bị ngắt) được coi là chưa xong.
type SummaryRecord = SummaryOk | SummaryDiverged | SummaryFailed


class ManifestRecord(TypedDict):
    """manifest.json của một attempt (KH §4.7; SPEC §5.3; ADR-008); có thời điểm tạo nên không tất định."""

    # Hash của (cấu hình chuẩn hóa + code_version + machine), nên lượt trên máy khác là run_id khác (ADR-008).
    run_id: str
    # sha256 nội dung mọi file *.py trong src/ cùng requirements.lock, tính từ file trên đĩa, không cần git
    # (ADR-008). Trạng thái cây ghi riêng ở `dirty`.
    code_version: str
    # Commit HEAD lúc chạy, để check out lại khi tái lập (REP-1); "none" khi không có git.
    git_commit: str
    # True khi chạy từ cây có thay đổi chưa commit; khi đó chỉ được ghi ra ngoài results/.
    dirty: bool
    # True khi mọi gói trong requirements.lock đã cài đúng phiên bản; False chỉ được ghi ra ngoài results/.
    env_match: bool
    # k trong attempt-<k>.
    attempt: int
    # Cấu hình chuẩn hóa đầy đủ (SPEC §5.3), gồm tập đánh giá và số luồng mỗi lượt.
    config: dict[str, object]
    seed: int
    # Một số nguyên đại diện cho mỗi luồng ngẫu nhiên theo seed của lượt (partition, placement, graph, init,
    # batch, gossip), suy ra tất định từ `seed` bằng numpy.random.SeedSequence với spawn_key cố định:
    # https://numpy.org/doc/2.5/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
    child_seeds: dict[str, int]
    # Số lần Partitioner phải lấy mẫu lại (PartitionResult.retries).
    partition_retries: int
    # Phiên bản python và phiên bản đã cài của mọi gói trong requirements.lock, đọc bằng
    # importlib.metadata.version: https://docs.python.org/3.12/library/importlib.metadata.html
    versions: dict[str, str]
    # Tên CPU kèm hash của danh sách cờ CPU, để phát hiện khi máy ảo bị chuyển sang phần cứng khác.
    # Nằm trong run_id.
    machine: str
    # Số luồng đọc lại bằng torch.get_num_threads() sau khi đặt:
    # https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html
    cpu_threads: int
    # Thời điểm tạo, ISO 8601.
    created_at: str
    # Spectral gap của topology tĩnh; None với gossip và với W = I.
    spectral_gap: float | None
    # Trọng số π của mục tiêu mà lượt chạy thực sự tối ưu, N phần tử: MH 1/N; uniform tỉ lệ d_j + 1;
    # size-weighted và size-corrected n_j/n (ADR-005).
    objective_weights: list[float]
    # TV(Σ_j π_j P_j, P): độ lệch giữa phân bố nhãn mà lượt chạy thực sự tối ưu và phân bố test.
    objective_shift_tv: float
    # N × 10: số mẫu mỗi lớp của từng node.
    node_class_counts: list[list[int]]


class ExperimentVersion(TypedDict):
    """Một mục của chỉ mục nhóm thí nghiệm: các run_id của lưới dưới một code_version (ADR-008)."""

    # run_id của mọi ô của lưới, sắp tăng dần; có thể có run_id chưa có attempt xong (analysis kiểm).
    run_ids: list[str]
    # Commit HEAD của cây chạy sweep, và máy (cũng nằm trong run_id).
    git_commit: str
    machine: str
    # Lần sweep gần nhất ghi mục này, ISO 8601.
    updated_at: str


class ExperimentIndex(TypedDict):
    """results/experiments/<exp>.json, do sweep ghi (SPEC §5.3; ADR-008); analysis chọn lượt qua file này.

    Mỗi code_version một mục. Sweep chỉ thay mục của code_version của chính nó; mục của code_version khác
    không bao giờ bị xóa hay ghi đè, để lượt cũ thành "stale" mà vẫn truy được.
    """

    experiment: str
    versions: dict[str, ExperimentVersion]


@runtime_checkable
class RunLogger(Protocol):
    """Ghi kết quả của một attempt vào results/runs/<run_id>/attempt-<k>/ (module metrics-io; ADR-008).

    Thứ tự gọi: start, rồi log_round và log_timing bao nhiêu lần tùy ý, rồi finish đúng một lần. Gọi sai
    thứ tự thì raise RuntimeError.
    """

    def start(self, manifest: ManifestRecord) -> None:
        """Tạo thư mục attempt mới (tạo nguyên tử) và ghi manifest.json.

        Raise FileExistsError nếu thư mục đã tồn tại: không bao giờ ghi đè kết quả cũ.
        """
        ...

    def log_round(self, record: RoundRecord) -> None:
        """Ghi thêm một dòng vào rounds.jsonl."""
        ...

    def log_timing(self, record: TimingRecord) -> None:
        """Ghi thêm một dòng vào timing.jsonl."""
        ...

    def finish(self, summary: SummaryRecord) -> None:
        """Ghi summary.json và đóng các file; gọi cho cả lượt phân kỳ và lượt lỗi."""
        ...
