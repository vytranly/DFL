# Checklist triển khai — Đề 2 (Non-IID × DFL)

> Kế hoạch, quyết định, đồ thị phụ thuộc, rủi ro: `tasks/plan.md`. Trạng thái: nhóm đã duyệt (Gate G2, 2026-09-30).
> **Sửa đổi 2026-10-01 theo trả lời Q1 của GVHD** (cần tiến trình chạy song song, mỗi node độc lập; Docker; tăng độ phức tạp model để xem chi phí giao tiếp): T22a, T22b thành bắt buộc trước E0a; T22c (Docker) là tùy chọn (Q12); thêm T09c và E8 (T27a–c). Theo ADR-010, ADR-011. **Nhóm duyệt ngày 2026-10-01** (tasks/plan.md K14).
> **Cách làm một task** (H P3.0, lệnh `/build`):
> 1. Chọn task chưa làm đầu tiên mà mọi phụ thuộc đã xong.
> 2. Viết test FAIL trước, rồi code tối thiểu cho test PASS.
> 3. `make check-task` phải xanh.
> 4. Đánh dấu `[x]` và ghi `tasks/SESSION_LOG.md`.
> 5. Commit đúng một commit, chỉ `git add` file của task. Ngoại lệ: task thí nghiệm có hai commit, xem quy trình thí nghiệm ở dưới.
>
> Test bất biến: chỉ task chủ của T-INV-k (plan §8) đặt tên test dạng `test_t_inv_<k>_…` (SPEC §7). Task khác có thể có test thành phần liên quan nhưng không dùng tên đó. `make check-inv` (có từ P1.8) kiểm đủ 12 ID; thành 13 ID khi SPEC thêm T-INV-13 (ADR-010, Ask first).
> Task nào cần thêm hay đổi schema trong `src/dfl/contracts.py` thì thêm file đó vào commit của mình và sửa `tests/unit/test_contracts_smoke.py` trước.
>
> "File" liệt kê tối đa 5 file, không tính `tasks/todo.md` và `tasks/SESSION_LOG.md`. Mọi task còn phải đạt Definition of Done (H§4.2).
> Phụ trách theo SPEC §3.1 và §10 dòng 1 (QS-9, nhóm chốt ở P2.1).
> ⚠ = rủi ro cao, làm sớm. ∥ = làm song song được với các task ghi sau nó.
>
> **Kiểm chứng một task thí nghiệm** (gọi tắt là "quy trình thí nghiệm"):
> 0. Một lần, ở E0a: tạo nhánh `grid` tại commit của lưới và worktree của nó (`git worktree add <dir> grid`). Nhánh `grid` không bao giờ nhận thay đổi trong `src/` hay `requirements.lock`, nên `code_version` không đổi suốt Tier 1 + 2; code mới (ví dụ T22) chỉ vào nhánh chính. Trong worktree, `data/` và `results/` là symlink tới thư mục của repo chính, để mọi lượt ghi vào cùng một `results/`.
> 1. Commit thứ nhất, trên nhánh `grid`: thẻ tiền đăng ký và cấu hình, trước khi chạy (EXP-1; `make check-task` xanh). Nhánh chính merge nhánh `grid` sau mỗi nhóm.
> 2. Lưới hơn 10 lượt thì trình kết quả `python -m dfl sweep --experiment <file> --dry-run` cho nhóm (Ask first).
> 3. Chạy `python -m dfl sweep --experiment <file> --workers <k>` từ worktree của lưới (SPEC §5.3). Sweep đặt `PYTHONPATH=<worktree>/src` cho tiến trình con, vì bản cài editable của `.venv` trỏ về `src/` của repo chính (đã kiểm: không đặt thì nạp nhầm code của nhánh chính).
> 4. Chạy lại `--dry-run`: phải báo 0 lượt sẽ chạy và 0 attempt `failed` chưa xử lý.
> 5. Chạy lệnh kiểm SAN-5 của `analysis.sanity` trên nhóm vừa chạy, rồi các lệnh `analysis` ghi ở task.
> 6. Commit thứ hai, trên nhánh `grid`: kết quả và kết luận ghi vào thẻ, cùng `tasks/`. Lịch sử git cho thấy thẻ có trước kết quả.

## Điều kiện trước pha 3

- [x] P1.9 đã commit (`07b183c`)
- [ ] Gate G1: nhóm và GVHD duyệt SPEC, ADR-001…011 (Proposed → Accepted). Q1 đã trả lời (2026-10-01); còn chờ Q10–Q13 (OPEN_QUESTIONS.md)
- [x] Gate G2: nhóm duyệt `tasks/plan.md` và file này (2026-09-30)
- [x] T01 — Dựng repo, pyproject, ruff, pytest, CLAUDE.md, README (commit `2789c5c`)

## Tuần 3 — Nền tảng (ba người song song)

- [x] **T02a · Seeding** — TV3 · S · ∥ T02b, T02c, T07, T04a
  - Mô tả: `src/dfl/seeding.py` là nơi duy nhất tạo `SeedSequence`/`Generator` (SPEC §6).
    - Hằng số ID luồng: PARTITION, PLACEMENT, GRAPH, INIT, BATCH, B1_BATCH, GOSSIP, GOSSIP_ER; cùng `DATA_SEED` cho luồng dữ liệu cố định.
    - Hàm trả về `numpy.random.Generator` và `torch.Generator` cho từng luồng.
    - `child_seeds` cho manifest.
  - Chấp nhận:
    - Cùng (seed, luồng, khóa) cho cùng số. Các luồng hay khóa khác nhau không trùng entropy, ví dụ (ID_BATCH, 0) với (ID_BATCH, 1), hay (ID_GOSSIP, r) với (ID_GOSSIP_ER, r).
    - Luồng dữ liệu cố định không phụ thuộc seed của lượt. Không dùng `hash()` của Python, không dùng RNG toàn cục (`make check-floor`, luật F7, xanh).
    - Cách dựng `torch.Generator` từ `SeedSequence` theo tài liệu torch 2.14, có URL trong docstring.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_seeding.py`
  - Phụ thuộc: T01
  - File: `src/dfl/seeding.py`, `tests/unit/test_seeding.py`

- [x] **T02b · Cấu hình + YAML base/smoke** — TV3 · M · ∥ T02a, T02c, T07, T04a
  - Mô tả: dataclass cấu hình frozen, đọc bằng `yaml.safe_load`, điền mặc định, kiểm kiểu và giá trị; JSON chuẩn hóa cho `run_id` (khóa sắp xếp, số thực theo `repr`). Tạo `configs/base.yaml` (mặc định KH §3.5, lưới 40 mốc, tập con 2,000, 1 luồng) và `configs/profiles/smoke.yaml` (SPEC §5.2).
  - Chấp nhận:
    - Cấu hình sai thì raise `ValueError` nêu ràng buộc bị vi phạm: kiểu sai, ví dụ `5e-2` (PyYAML đọc thành chuỗi, nên phải viết `0.05` hay `5.0e-2`), #C = k với N < 10, 100 không chia hết cho τ ở profile có lưới 100 bước.
    - Hai YAML tương đương (khác thứ tự khóa, `0.05` so với `5.0e-2`) cho cùng JSON chuẩn hóa. JSON này không gồm tên nhóm, `--out`, `--workers`.
    - `base.yaml` và `smoke.yaml` đọc được và hợp lệ.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_config.py`
  - Phụ thuộc: T01
  - File: `src/dfl/config.py`, `tests/unit/test_config.py`, `configs/base.yaml`, `configs/profiles/smoke.yaml`

- [x] **T02c · `requirements.lock` + `.gitattributes`** — TV3 · M · ∥ T02a, T02b · cần QS-1
  - Mô tả: sinh `requirements.lock` trong venv chỉ có gói chạy (SPEC §4); thêm `.gitattributes` (`* text=auto eol=lf`); sửa lệnh cài đặt trong README và CLAUDE.md sang `-r`/`-c requirements.lock`.
  - Chấp nhận:
    - Venv mới cài đúng §4 thì `pip check` sạch và `make check-full` xanh. File khóa không có pytest, ruff, import-linter hay scikit-learn.
    - `git ls-files --eol '*.py'` cho `i/lf w/lf`.
    - Nhóm duyệt diff của CLAUDE.md, vì đây là file quy tắc.
  - Kiểm chứng: `make check-full`; cài thử trong venv tạm theo §4; `grep -Ei '^(pytest|ruff|import-linter|scikit-learn)' requirements.lock` không ra dòng nào
  - Phụ thuộc: T01; QS-1
  - File: `requirements.lock`, `.gitattributes`, `README.md`, `CLAUDE.md`

- [x] **T07 · Mixing** ⚠ — TV2 · S · làm đầu tiên; ∥ T02a–c, T04a
  - Mô tả: `network/mixing.py` gồm MH, uniform, size-weighted (chỉ cho FC), spectral gap, và trọng số mục tiêu π (ADR-005). Đầu vào sai (đồ thị có hướng, đa cạnh, self-loop, node đánh số sai) thì raise `ValueError`. Không import torch. Test chỉ cần đồ thị networkx, không cần T06.
  - Chấp nhận:
    - T-INV-9: spectral gap của ring(20) = ⅔(1 − cos(2π/20)) ≈ 0.0326; của FC = 1.
    - MH đối xứng, ngẫu nhiên kép, không âm, bằng 0 ngoài cạnh. Uniform chỉ ngẫu nhiên theo hàng (gap tính bằng `eigvals`). Size-weighted khớp $n_j/n$ và từ chối đồ thị không đầy đủ. Matching cho W đối xứng và W² = W.
    - π: MH là 1/N; uniform tỉ lệ với $d_j+1$; size-weighted và size-corrected là $n_j/n$.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_mixing.py`
  - Phụ thuộc: T01
  - File: `src/dfl/network/mixing.py`, `tests/unit/test_mixing.py`

- [x] **T04a · Heterogeneity** — TV1 · M · ∥ T02a–c, T07
  - Mô tả: `data/heterogeneity.py`, chỉ dùng numpy. Tính ma trận node × lớp, JS (bit), TV, số lớp chiếm ≥ 5%, CV(n), và lệch mục tiêu TV(Σ π_j P_j, P). Thêm module vào contract `pure-no-torch`.
  - Chấp nhận:
    - JS và TV khớp giá trị tính tay: hai phân bố trùng nhau cho 0; hai lớp rời nhau cho 1 bit. Lệch mục tiêu bằng 0 khi π đều và mọi node IID.
    - CV(n) và số lớp ≥ 5% đúng trên ví dụ nhỏ.
    - `lint-imports` báo lỗi nếu module này import torch.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_heterogeneity.py`; `lint-imports`
  - Phụ thuộc: T01
  - File: `src/dfl/data/heterogeneity.py`, `tests/unit/test_heterogeneity.py`, `pyproject.toml`

- [x] **T06 · Topology** — TV2 · S · ∥ T03, T04b
  - Mô tả: `network/topology.py` dựng T0 star, T1 ring, T2 ER (p = 0.3, lấy mẫu lại tới khi liên thông), T3 perfect matching trên $K_N$ mỗi round, T3b maximal matching trên cạnh ER, T4 FC, và lịch W = I cho B2. Đồ thị tĩnh được kiểm liên thông lúc dựng. `graph_at` là hàm thuần của (seed, round): lịch nhận một hàm `rng_for(round) -> numpy.random.Generator` do `sim` dựng từ `seeding.numpy_rng(seed, Stream.GOSSIP, round)` (ADR-009). `topology.py` không import `seeding`, vì `seeding` import torch (contract `pure-no-torch`). Không dùng `default_rng([seed, r])` như H P3.2 gợi ý (plan K10).
  - Chấp nhận:
    - Ring bậc 2; FC bậc N − 1; ER liên thông và tất định theo seed. T3: mỗi node tối đa một đối tác, N lẻ thì đúng một node đứng ngoài. T3b chỉ dùng cạnh ER và là maximal matching.
    - Gọi `graph_at(r)` nhiều lần, gọi xen `graph_at(r')`, hay dùng hai lịch tạo độc lập với cùng (seed, N) đều cho cùng đồ thị (SPEC REQ-01).
    - E[W] của T3 qua 10⁴ matching khớp ½I + (J − I)/(2(N − 1)) khi N chẵn, trong ±4 sai số chuẩn; hợp các đồ thị của T3 và của T3b qua 200 round liên thông (INV-4).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_topology.py`
  - Phụ thuộc: T02a, T07
  - File: `src/dfl/network/topology.py`, `tests/unit/test_topology.py`

- [x] **T03 · Datasets + lệnh `download`** — TV1 · M · ∥ T06
  - Mô tả: `data/datasets.py` nạp MNIST qua torchvision thành tensor trong RAM và chuẩn hóa bằng mean/std của tập train. Validation 5k và tập con đánh giá 2,000 ảnh được chọn phân tầng bằng luồng dữ liệu cố định. Có bộ dữ liệu tổng hợp cùng shape cho test và smoke. Tạo khung CLI (`cli.py`, `__main__.py`) với `python -m dfl download mnist`. `cli.py` chỉ import module dùng torch bên trong từng lệnh con, để `run` còn đặt được biến môi trường số luồng trước khi torch được nạp (T11b).
  - Chấp nhận:
    - Trên dataset giả: tách 55k/5k phân tầng, đếm lớp đúng. Ảnh là float32 shape (n, 1, 28, 28), nhãn là int64. Tập con đánh giá 2,000 ảnh của test và của validation phân tầng. Cùng `DATA_SEED` thì cùng cách tách và cùng tập con ở mọi seed của lượt.
    - Test không dùng mạng (ENG-8). Nếu dùng `.data`/`.targets` thì ghi UNVERIFIED như trong `contracts.py`.
    - `python -m dfl download mnist` tải được vào `data/` trên máy chủ (kiểm tay, cần mạng).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_datasets.py`; chạy tay `python -m dfl download mnist`
  - Phụ thuộc: T02a, T02b
  - File: `src/dfl/data/datasets.py`, `src/dfl/cli.py`, `src/dfl/__main__.py`, `tests/unit/test_datasets.py`

- [x] **T04b · Partitioner P0–P3** — TV1 · M
  - Mô tả: `data/partition.py`, chỉ dùng numpy, cài P0–P3 đúng NIID-Bench với min_size = 64. Lấy mẫu lại tối đa 1,000 lần, quá thì raise `ValueError`; #C = k với N < 10 cũng raise `ValueError`. Nhận `numpy.random.Generator` qua tham số, không import `seeding`. Test golden so với hàm tham chiếu sao từ spike (H P3.1); phép so này chỉ kiểm bản cài khớp spike (SPEC REQ-02).
  - Chấp nhận:
    - T-INV-8: rời nhau, phủ đủ, mọi node ≥ min_size; #C = k đúng k nhãn và mọi nhãn có chủ; golden trùng tuyệt đối; N < 10 báo lỗi.
    - Trên mảng nhãn tổng hợp theo số đếm lớp của MNIST (55k, không tải dữ liệu), α = 1000 cho JS < 0.01, và JS (trung bình 3 seed) tăng đơn điệu khi α giảm.
    - `PartitionResult.retries` đúng; vượt 1,000 lần thì `ValueError`; contract `pure-no-torch` có thêm `dfl.data.partition`.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_partition.py -k t_inv_8`; `lint-imports`
  - Phụ thuộc: T02a, T04a
  - File: `src/dfl/data/partition.py`, `tests/reference/niid_bench_reference.py`, `tests/unit/test_partition.py`, `pyproject.toml`

- [x] **T04c · Hoán vị vị trí node + P2' (shards)** — TV1 · S · ∥ T06
  - Mô tả: hoán vị partition vào vị trí node bằng Generator của luồng placement (KH §3.2), và P2' (shards của McMahan) theo QS-11, chỉ dùng trong `describe`.
  - Chấp nhận:
    - Cùng Generator thì cùng hoán vị; hoán vị là một hoán vị hợp lệ của 0..N−1 và không đổi kích thước hay nhãn của các phần.
    - P2': 2N mảnh sắp theo nhãn, mỗi node 2 mảnh, rời nhau và phủ đủ; test golden theo thứ tự tiêu thụ số ngẫu nhiên ghi trong docstring.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_partition_extras.py`
  - Phụ thuộc: T04b
  - File: `src/dfl/data/partition.py`, `tests/unit/test_partition_extras.py`

## Tuần 4 — C1, C2 (M2), engine bắt đầu

- [x] **T05a · `dfl describe`: dữ liệu** — TV1 · M · ∥ T08
  - Mô tả: `sim/describe.py` và lệnh `python -m dfl describe --config … --seeds 20 --out build/describe`. Với mỗi cách chia × seed, ghi `describe.json` (ma trận node × lớp, JS, TV, CV, lệch mục tiêu) và bảng CSV. Không huấn luyện.
  - Chấp nhận:
    - Test tích hợp trên dữ liệu tổng hợp: có đủ mọi cách chia × seed, JSON đọc lại được, không có NaN.
    - Cờ `--check-req02` so JS với bảng KH §3.2 (lưu ở `configs/reference/kh_js_table.yaml`) và trả mã lỗi khác 0 khi sai: JS trung bình phải tăng nghiêm ngặt IID < α100 < α1 < α0.5 < α0.1 < α0.05 và #C2 < #C1, mỗi giá trị lệch bảng không quá max(10%, 0.002). Hàm so sánh có unit test với số giả.
    - Chạy `--check-req02` trên MNIST thật (N = 20, 20 seed) ở máy chủ ra mã 0 (REQ-02); kết quả ghi vào SESSION_LOG.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_describe.py`; `python -m dfl describe --config configs/base.yaml --seeds 20 --out build/describe --check-req02` trên máy chủ
  - Phụ thuộc: T03, T04b, T04c
  - File: `src/dfl/sim/describe.py`, `src/dfl/cli.py`, `tests/integration/test_describe.py`, `configs/reference/kh_js_table.yaml`

- [ ] **T05b · Hình P7** — TV3 · S · ∥ T08
  - Mô tả: hàm vẽ P7 (heatmap node × lớp, bảng màu an toàn cho người mù màu) trong `analysis/plots.py`. Hàm đọc `describe.json`, và sau này cả `node_class_counts` trong manifest. `analysis` chỉ import `dfl.contracts`. Ở task này hình chỉ được ghi vào `build/`, không vào `figures/`: hình cho báo cáo chỉ sinh bằng lệnh chuẩn, có PROVENANCE, từ T16a trở đi (INT-1).
  - Chấp nhận:
    - Sinh được P7 cho mọi cách chia từ `describe.json` mẫu của test; trục có nhãn.
    - Contract `analysis-reads-only` xanh.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_plots_p7.py`
  - Phụ thuộc: T05a (định dạng JSON; có thể bắt đầu với file giả)
  - File: `analysis/plots.py`, `tests/unit/test_plots_p7.py`

- [ ] **Checkpoint C1** — sau T05a, T05b (plan §7): `make check-full`; chạy `describe --check-req02` và vẽ P7 trên MNIST thật vào `build/describe` (không vào `figures/`); review diff bằng `code-reviewer`; tăng `MIN_TESTS`; nhóm duyệt

- [x] **T08 · Transport + CommLedger + snapshot** — TV2 · M · ∥ T05a, T05b
  - Mô tả:
    - `network/transport.py`: in-memory, không copy payload. CommLedger đếm message và byte, tổng và theo từng node, gồm cả gửi lẫn nhận. Nhận server ảo có chỉ số N; cấm node tự gửi cho mình.
    - `make_snapshot` trong `sim/node.py`: clone + detach, `nbytes` tính từ tensor thật.
  - Chấp nhận:
    - CommLedger: byte = Σ `nbytes`; một state có shape của SmallCNN cho 4 × 46,730 byte; tải = byte gửi + byte nhận; message tự gửi bị từ chối. Đây là test thành phần; bất biến về số message qua engine thuộc T10b.
    - Node chỉ đọc được inbox của chính nó. `end_round` giải phóng tham chiếu mà giữ nguyên số đếm.
    - `transport.py` không import torch (contract `pure-no-torch` xanh).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_transport.py tests/unit/test_snapshot.py`
  - Phụ thuộc: T06
  - File: `src/dfl/network/transport.py`, `src/dfl/sim/node.py`, `tests/unit/test_transport.py`, `tests/unit/test_snapshot.py`

- [ ] **T08b · `dfl describe`: topology** — TV3 · S · ∥ T10a
  - Mô tả: mở rộng `describe` với bảng topology: spectral gap của T1, T4, và T2 trên 20 seed; số message/round dự kiến = 2·|E|, là thuộc tính của đồ thị, không phải phép đếm chi phí; T0 star; tốc độ trộn kỳ vọng của T3; ước lượng cho T3b từ 10⁴ matching. Chi phí thật chỉ do CommLedger đếm trong lượt chạy (INV-2); test của T08b gửi một round message giả qua Transport trên chính các đồ thị đó và so số đếm với 2·|E|.
  - Chấp nhận:
    - Ring 0.0326 (±1e-4), FC 1, ER trung bình trong [0.15, 0.25] trên 20 seed. Con số 0.312 của T3 được ghi là tốc độ trộn, không phải spectral gap.
    - Có test tích hợp ở N nhỏ. Lệnh chạy ở N = 20 in bảng so với KH §3.3 (demo của C2).
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_describe.py`
  - Phụ thuộc: T05a, T06, T07, T08
  - File: `src/dfl/sim/describe.py`, `tests/integration/test_describe.py`

- [ ] **Checkpoint C2 (M2)** — sau T08, T08b: `make check-full`; bảng spectral gap sinh từ code; review diff; tăng `MIN_TESTS`; nhóm duyệt

- [x] **T10a · Engine ATC lõi** ⚠ — TV2 · M · ∥ T09a
  - Mô tả:
    - `sim/node.py` và `sim/engine.py` chạy một round ATC cho topology tĩnh:
      1. Adapt: τ bước qua `LocalTrainer`.
      2. Mỗi node tạo đúng một snapshot.
      3. Combine: cộng theo id người gửi tăng dần vào bộ đệm mới.
      4. Chỉ ghi vào model sau khi mọi node đã tổng hợp xong.
    - W lấy từ `MixingPolicy`; B2 dùng W = I.
    - Test dùng MLP tí hon, dữ liệu tổng hợp 4 node, và một trainer theo Protocol `LocalTrainer` đã có trong `contracts.py` (P1.7), nên không cần T09 (plan K2). Engine truyền cho trainer model của node ở đầu round, nên anchor của số hạng prox là $w_i^t$ của round đó.
    - `tests/conftest.py` có fixture `autouse` kiểm `torch.get_rng_state()` không đổi, chỉ áp cho test nằm trong `tests/integration/` và `tests/e2e/` (lọc theo đường dẫn của test); đây là hạ tầng của T-INV-5, test T-INV-5 thuộc T11b.
  - Chấp nhận:
    - Viết trước và thấy FAIL trước khi có code: T-INV-2 (W = I trùng khớp huấn luyện chỉ cục bộ) và T-INV-4 (đảo thứ tự duyệt node thì kết quả trùng từng bit, checksum mọi snapshot không đổi).
    - Chỉ trộn tensor float; không có phép tại chỗ trên snapshot; node chỉ đọc model của mình và inbox (INV-1, INV-3), kiểm bằng một test dùng Transport ghi lại mọi lần đọc inbox.
    - Với μ > 0 và trainer giả ghi lại anchor, anchor ở round t bằng model của node đầu round t, và đổi mỗi round.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_engine_core.py`
  - Phụ thuộc: T06, T07, T08
  - File: `src/dfl/sim/node.py`, `src/dfl/sim/engine.py`, `tests/conftest.py`, `tests/integration/conftest.py`, `tests/integration/test_engine_core.py`

- [x] **T09a · SmallCNN + huấn luyện cục bộ** — TV1 (QS-9) · M · ∥ T10a
  - Mô tả:
    - `learning/models.py`: SmallCNN 46,730 tham số, không BatchNorm, không dropout. Tạo module mà không tiêu thụ RNG toàn cục, bằng `skip_init` hoặc `fork_rng`. $w^0$ lấy mẫu từ luồng init.
    - `learning/local_train.py`: τ bước SGD; minibatch có hoàn lại bằng `torch.randint(generator=…)`; số hạng prox μ với anchor clone + detach; `loss_scale`; dừng ngay khi mất mát không hữu hạn.
  - Chấp nhận:
    - T-INV-7: μ = 0 tương đương SGD thường; gradient của số hạng prox bằng μ(w − anchor); anchor không có gradient.
    - Đúng 46,730 tham số. Khác seed thì khác $w^0$ ở mọi tensor. `torch.get_rng_state()` không đổi sau khi tạo model và chạy huấn luyện cục bộ.
    - Mất mát NaN hay inf thì trả về ngay giá trị đó, không chạy các bước còn lại.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_models.py tests/unit/test_local_train.py`
  - Phụ thuộc: T02a
  - File: `src/dfl/learning/models.py`, `src/dfl/learning/local_train.py`, `tests/unit/test_models.py`, `tests/unit/test_local_train.py`

## Tuần 5–6 — Engine hoàn chỉnh, lát dọc đầu tiên (C3, M3)

- [x] **T09b · Độ đo phân loại + consensus + Evaluator** — TV1 (QS-9) · M · ∥ T10b, T11a
  - Mô tả:
    - `metrics/classification.py`: từ ma trận nhầm lẫn tính Acc, Acc^loc, Acc^eff, và macro-F1 dạng 2TP/(2TP + FP + FN) trên đủ 10 lớp.
    - Evaluator (cũng trong `classification.py`) nhận một model cấp phát sẵn làm tham số, không import `learning` (ENG-7), và nạp state bằng `load_state_dict`. P_i và π truyền vào lúc tạo đối tượng. Không tiêu thụ RNG.
    - `metrics/consensus.py`: drift và consensus distance, dạng khoảng cách RMS.
  - Chấp nhận:
    - T-INV-10: macro-F1 khớp `sklearn.metrics.f1_score(average="macro", labels=range(10), zero_division=0)` trên nhãn ngẫu nhiên, trên dự đoán chỉ một lớp, và trên dự đoán thiếu lớp (không có NaN). Acc^loc và Acc^eff khớp accuracy có trọng số từng mẫu, sai số ≤ 1e-12.
    - Test: Evaluator không sửa state (checksum trước và sau) và không đổi `torch.get_rng_state()` (INV-5).
    - Ξ khớp công thức RMS trên ví dụ tính tay.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_classification.py tests/unit/test_consensus.py`
  - Phụ thuộc: T01 (test dùng MLP tí hon, không cần T09a)
  - File: `src/dfl/metrics/classification.py`, `src/dfl/metrics/consensus.py`, `tests/unit/test_classification.py`, `tests/unit/test_consensus.py`

- [x] **T10b · Gossip + đường server** ⚠ — TV2 · M · ∥ T09b, T11a
  - Mô tả:
    - Engine chạy với topology động: gọi `graph_at` mỗi round. Node không được ghép vẫn chạy Adapt và dùng W_ii = 1.
    - Đường server cho B3/E1c: server ảo chỉ số N nhận N snapshot, tổng hợp theo $n_k/n$, rồi gửi lại N snapshot qua Transport.
  - Chấp nhận:
    - T-INV-1, với lr = 0 và khởi tạo khác nhau. Topology tĩnh (ring, ER, FC): Ξ_t ≤ ρ^t Ξ_0 (1 + 1e-6) + 1e-12 và trung bình model được bảo toàn (MH). Gossip: mỗi W đối xứng, ngẫu nhiên kép, W² = W, và E[W] khớp trong sai số lấy mẫu. Test tự tính Ξ bằng công thức RMS làm oracle độc lập, không import `metrics/consensus.py`, nên T10b không phải đợi T09b.
    - T-INV-3: FC + size-weighted, đường server, và một FedAvg tham chiếu viết ngay trong test cho kết quả allclose (atol 1e-6). Khi kích thước node bằng nhau, FC + MH cũng vậy.
    - T-INV-6 qua engine, N = 20: ring 40, FC 380, T3 20, ER đúng 2·|E| của đồ thị, đường server 2N message/round; byte = message × `nbytes`; không có message tự gửi; W mà engine dùng chỉ khác 0 trên cạnh của `graph_at(r)` (INV-4).
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_engine_consensus.py tests/integration/test_engine_fedavg.py tests/integration/test_engine_messages.py`
  - Phụ thuộc: T10a
  - File: `src/dfl/sim/engine.py`, `tests/integration/test_engine_consensus.py`, `tests/integration/test_engine_fedavg.py`, `tests/integration/test_engine_messages.py`

- [x] **T11a · Manifest, `code_version`, RunLogger** — TV3 · M · ∥ T09b, T10b
  - Mô tả:
    - `io/manifest.py` dựng `ManifestRecord`:
      - `code_version`: hash dãy cặp (đường dẫn, nội dung) của thư mục gói `dfl` đang được nạp (`dfl.__file__`), không của thư mục làm việc, chuẩn hóa xuống dòng, cùng file khóa;
      - `run_id`: hash của cấu hình chuẩn hóa + `code_version` + máy;
      - `git_commit`, `dirty` (tính cả file `*.py`, `*.yaml` bị ignore trong `src/`, `configs/`; `__pycache__` và `*.egg-info` không tính);
      - kiểm môi trường theo `requirements.lock` để ra `env_match`;
      - máy: tên CPU kèm hash cờ CPU.
    - `io/run_logger.py`: tạo attempt nguyên tử, thứ tự start → log → finish, JSON `allow_nan=False`.
  - Chấp nhận:
    - `code_version` không đổi khi chỉ sửa docs/analysis/configs hay đổi CRLF ↔ LF. Nó đổi khi sửa hay đổi tên một file `.py` trong `src/`, hay khi sửa file khóa. `run_id` đổi khi cấu hình hay máy đổi.
    - Một gói lệch phiên bản so với file khóa thì `env_match: false`; công cụ dev cài thêm không làm lệch.
    - Ghi vào attempt đã có thì `FileExistsError`; gọi sai thứ tự thì `RuntimeError`; ghi NaN thì báo lỗi.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_manifest.py tests/unit/test_run_logger.py`
  - Phụ thuộc: T02b, T02c
  - File: `src/dfl/io/manifest.py`, `src/dfl/io/run_logger.py`, `tests/unit/test_manifest.py`, `tests/unit/test_run_logger.py` (các trường của ManifestRecord đã có sẵn trong contracts.py)

- [ ] **T15b · R@x, AUC, độ dao động, mức sụt** — TV3 · S · ∥ T10b, T10c
  - Mô tả: `analysis/convergence.py` theo đúng SPEC REQ-04 và ADR-007: lưới 100 bước; chuỗi `acc_mean`; lần vượt bền vững đầu tiên; "≤ 100" và "> ngân sách"; AUC theo quy tắc hình thang. Độ dao động và mức sụt tính theo từng node.
  - Chấp nhận:
    - Unit test đủ các ca của REQ-04: đạt; không đạt; đạt ở mốc đầu; đạt rồi tụt rồi đạt lại (lấy lần bền vững); chỉ vượt ở mốc cuối (không đạt); lưới khác lưới chuẩn thì báo lỗi, trừ baseline B1, B1-4k (lưới bội của 100 tới hết ngân sách của chính nó, chỉ vẽ làm đường tham chiếu) và lượt `diverged` có lưới là phần đầu của lưới chuẩn; với lượt `diverged`: khi đó R@x là "không đạt", AUC là "không xác định (phân kỳ)", và lượt được đếm vào "k/n seed phân kỳ" (INT-3).
    - Độ dao động và mức sụt tính theo node rồi mới lấy trung bình (G24), có test cho thấy khác với khi tính trên `acc_mean`. Có thêm, mỗi độ đo một test: R@x quy ra round và byte, bytes@x, tải lớn nhất trên một node (REQ-05), và ở round cuối std, min, max − min, trung bình 10% node kém nhất (REQ-09).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_convergence.py`
  - Phụ thuộc: T01
  - File: `analysis/convergence.py`, `tests/unit/test_convergence.py`

- [ ] **T15c · Thống kê + quy tắc quyết định** — TV3 · S · ∥ T10b, T10c
  - Mô tả: `analysis/stats.py` gồm:
    - mean ± CI 95% theo phân phối t, dùng bảng giá trị tới hạn cho df ≤ 30 có ghi nguồn (plan K11);
    - hiệu số ghép cặp;
    - quy tắc "ủng hộ / gợi ý / không ủng hộ / chưa kết luận" cho 5 seed và 3 seed;
    - kiểm định dấu trên thứ tự cho R@x bị kiểm duyệt;
    - độ dốc của H2 theo từng seed;
    - mức khuyến nghị (ADR-007).
  - Chấp nhận:
    - Giá trị tới hạn t khớp `scipy.stats.t` trong test (scipy có trong dev qua scikit-learn). Bản thân `analysis` không import scipy (contract `no-dev-deps`).
    - Các ca biên đúng quy tắc của ADR-007: 5/5 seed cùng chiều và CI không chứa 0 thì "ủng hộ"; 3 seed tối đa là "gợi ý"; hòa giữa hai giá trị bị kiểm duyệt tính là không đúng chiều; có seed phân kỳ thì "chưa kết luận".
    - Độ dốc H2: OLS theo JS cho từng seed, CI t với df = 4, kết luận đúng quy tắc của ADR-007.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_stats.py`
  - Phụ thuộc: T01
  - File: `analysis/stats.py`, `tests/unit/test_stats.py`

- [x] **T10c · Observer + phản biện engine** ⚠ — TV2 · M
  - Mô tả:
    - `sim/observer.py`: đánh giá theo lưới mốc (mỗi 100 bước; smoke đánh giá mỗi round; round cuối luôn được đánh giá).
    - Ở mỗi mốc: gọi Evaluator (T09b) cho mọi node để lấy Acc_i, Acc^loc_i, Acc^eff, macro-F1; tính acc($\bar w$); đo drift trước tổng hợp và consensus distance sau tổng hợp; dựng `RoundRecord`.
    - Engine kiểm mọi tham số hữu hạn sau mỗi round, trước khi ghi bản ghi nào; có test cho round có tham số NaN. Sửa sau phản biện nếu vượt khoảng 300 dòng thì tách thành task mới.
  - Chấp nhận:
    - T-INV-11: với FC + MH, sau tổng hợp mọi node trùng model, std của Acc bằng 0, và Acc^loc của mọi node tính từ cùng một vector recall.
    - T-INV-12: tắt Observer, hay gọi `graph_at` thêm lần, thì quỹ đạo huấn luyện không đổi, trùng từng bit.
    - Mỗi `RoundRecord` có đủ trường của `contracts.py`, kể cả `acc_per_node` và `acc_local_per_node` (N phần tử); không có NaN.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_observer.py`; sau đó chạy doubt-driven cho `engine.py` và `node.py` (H P3.3, bước 4; tối đa 3 vòng), và ghi kết quả reconcile vào SESSION_LOG trước khi commit
  - Phụ thuộc: T10b, T09b
  - File: `src/dfl/sim/observer.py`, `src/dfl/sim/engine.py`, `tests/integration/conftest.py`, `tests/integration/test_observer.py`, `tests/integration/test_engine_review.py` (không cần sửa node.py; test hồi quy của phản biện nằm ở file riêng)

- [x] **T10d · Sửa engine theo phản biện doubt-driven (3 vòng)** ⚠ — TV2 · M
  - Mô tả: các bản sửa sau phản biện của T10c vượt khoảng 300 dòng, nên theo T10c là một task riêng. Các bản sửa đan xen trong `engine.py`, nên gộp vào cùng commit với T10c, để không có trạng thái trung gian chưa được test (SESSION_LOG).
  - Chấp nhận:
    - Mỗi phát hiện hợp lệ có test hồi quy đã FAIL trước khi sửa (H P3.4); đánh đổi và phát hiện không phải lỗi được ghi vào SESSION_LOG.
    - `make check-full` xanh; các đột biến của T10a–c vẫn bị bắt.
  - Kiểm chứng: `make check-full`; `pytest -q tests/integration/test_engine_review.py`
  - Phụ thuộc: T10c
  - File: `src/dfl/sim/engine.py`, `tests/integration/test_engine_review.py`, `tests/integration/test_engine_consensus.py`, `tests/integration/test_engine_messages.py`

- [ ] **T15a · Analysis: đọc kết quả + kiểm INT-2, INT-3, INT-6** — TV3 · M · ∥ T10c, T11b
  - Mô tả:
    - `analysis/aggregate.py`: đọc `results/` qua chỉ mục của nhóm, theo từng `code_version`; mỗi lượt một dòng; xuất CSV.
    - `analysis/checks.py` báo lỗi khi:
      - trộn nhiều `code_version`;
      - có lượt `dirty` hay `env_match: false` (trừ `--draft`);
      - lệch biến cố định (kể cả dataset và máy);
      - tập seed không khai báo trước;
      - ô đường server được so sánh mà không qua kiểm;
      - gặp NaN;
      - một `code_version` thiếu dòng trong `CODE_VERSIONS.md`.
    - Đếm attempt `failed` và `diverged`.
  - Chấp nhận:
    - Mỗi luật của INT-2, INT-3, INT-6 có một unit test báo lỗi rõ ràng và một test cho trường hợp hợp lệ.
    - Chỉ import `dfl.contracts` và tự đọc JSON/JSONL; contract `analysis-reads-only` xanh.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_aggregate.py tests/unit/test_checks.py`
  - Phụ thuộc: T01 (test dùng `rounds.jsonl` giả theo `contracts.py`; khi đã có kết quả smoke thật thì chạy lại test với file thật). Phần đọc chỉ mục thí nghiệm theo TypedDict do task sweep thêm vào contracts.py; nếu chưa có thì viết theo schema ở SPEC §5.3 rồi kiểm lại khi có
  - File: `analysis/aggregate.py`, `analysis/checks.py`, `tests/unit/test_aggregate.py`, `tests/unit/test_checks.py`

- [x] **T11b · `python -m dfl run` + smoke e2e** — TV2 · M
  - Mô tả:
    - `sim/run.py` lắp một lượt từ cấu hình: data → partition → placement → topology → mixing → engine → observer → logger, và ghi ra `--out`. Manifest có `partition_retries`, `node_class_counts`, `objective_weights`, lệch mục tiêu, và spectral gap (topology tĩnh; `None` với gossip, W = I và B1).
    - Lệnh `run` trong `cli.py` đọc cấu hình (không import torch), đặt `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, rồi mới import `sim.run`; `sim.run` gọi `torch.set_num_threads` trước mọi phép tính.
    - Đưa `check-inv` vào `check-full` (CONSTRAINTS, T11).
  - Chấp nhận:
    - ENG-4: `python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke` xong dưới 60 s, đủ 4 file đúng schema, `rounds.jsonl` có 6 dòng; manifest ghi `cpu_threads` bằng số luồng của cấu hình.
    - T-INV-5 (SPEC §9.2): cùng seed và `code_version` thì `rounds.jsonl` giống hệt; khác seed thì khác cách chia và khác $w^0$ ở mọi tensor; cùng seed thì cùng partition, hoán vị và $w^0$ giữa các topology và thuật toán; `torch.get_rng_state()` không đổi sau một lượt.
    - `make check-inv` xanh đủ 12 T-INV.
  - Kiểm chứng: `make check-full`; `make check-inv`; `pytest -q tests/e2e/test_smoke_run.py tests/integration/test_seeds.py`
  - Phụ thuộc: T10c, T11a, T09a, T03, T04c
  - File: `src/dfl/sim/run.py`, `src/dfl/cli.py`, `tests/e2e/test_smoke_run.py`, `tests/integration/test_seeds.py`, `Makefile`

- [x] **T11d · Quy tắc ghi `results/`, phân kỳ, lỗi** — TV3 · S
  - Mô tả: trong `sim/run.py`: ghi vào `results/` chỉ khi cây sạch (theo định nghĩa ở T11a) và môi trường khớp `requirements.lock`; khóa `flock` theo run_id (`fcntl` chỉ import khi ghi vào `results/`); từ chối run_id đã xong. Phân kỳ thì ghi `SummaryDiverged`, lỗi thì ghi `SummaryFailed`, với `allow_nan=False`.
  - Chấp nhận:
    - ENG-10: vào `results/` thì từ chối khi cây bẩn, môi trường lệch, hay run_id đã xong; ra `--out` thì chạy được và manifest ghi đúng `dirty`, `env_match`.
    - Lượt có mất mát NaN (cấu hình η rất lớn trên dữ liệu tổng hợp) ghi `SummaryDiverged` với round phân kỳ, và không có dòng NaN nào trong `rounds.jsonl`; một ngoại lệ trong lượt ghi `SummaryFailed`.
    - Hai tiến trình cùng chạy một run_id: tiến trình thứ hai thấy khóa và thoát, không tạo attempt thứ hai.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_results_rules.py`
  - Phụ thuộc: T11b, T11c (cùng sửa `sim/run.py`, nên làm sau)
  - File: `src/dfl/sim/run.py`, `tests/integration/test_results_rules.py`

- [ ] **T11e · Tập đánh giá + đánh giá cuối** — TV1 · M · ∥ T11d
  - Mô tả: cấu hình chọn tập đánh giá (`test`, hoặc `validation` cho E0a), kích thước tập con của learning curve và seed của tập con (luồng dữ liệu cố định). Sau round cuối, Observer đánh giá mọi node và acc($\bar w$) trên đủ tập đánh giá (10,000 ảnh test hoặc 5,000 ảnh validation), rồi ghi vào các trường `final_*` của `SummaryOk`, kèm `final_eval_split` và `final_eval_size` (REQ-03).
  - Chấp nhận:
    - Cấu hình `validation` thì không ảnh test nào được nạp hay đánh giá: test đếm số lần gọi bộ nạp test bằng 0.
    - `SummaryOk` có đủ trường `final_*` (N phần tử) trên đủ tập; learning curve vẫn dùng tập con.
    - Tập con giống nhau ở mọi seed của lượt, khác nhau giữa test và validation.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_final_eval.py`
  - Phụ thuộc: T11b, T03, T09b
  - File: `src/dfl/sim/observer.py`, `src/dfl/sim/run.py`, `src/dfl/config.py`, `tests/integration/test_final_eval.py`

- [ ] **T14a · Lệnh kiểm sanity + chọn η** — TV1 · S · ∥ T11b
  - Mô tả: `analysis/sanity.py` kiểm SAN-1…5 (R7; SAN-2 so với B1-4k; SAN-5 là điều kiện cấu trúc), đọc `results/` và trả mã lỗi khác 0 khi không đạt.
  - Chấp nhận:
    - Mỗi SAN có unit test cho ba ca: đạt, không đạt, thiếu lượt.
    - SAN-5 kiểm được trên mọi nhóm, với tham số là tên nhóm thí nghiệm.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_sanity.py`
  - Phụ thuộc: T15a, T15b
  - File: `analysis/sanity.py`, `tests/unit/test_sanity.py`

- [ ] **T14e · Quy tắc chọn η** — TV1 · S · ∥ T14a
  - Mô tả: `analysis/select_lr.py` cài các quy tắc của ADR-007.
    - η chung: trung bình 5 mốc cuối trên tập con validation, lấy trường hợp xấu nhất qua các ô; hòa trong 0.2 điểm % thì xét AUC ở ring-α0.1.
    - Quy tắc cho τ = 1 và τ = 50, cùng điều kiện kích hoạt 24 lượt bổ sung.
    - η riêng của B1: accuracy validation lớn nhất trong 3 lượt B1 của E0a, hòa thì lấy AUC lớn hơn.
    - μ tính từ η theo QS-2.
  - Chấp nhận:
    - Có unit test cho các ca: hòa; lượt phân kỳ tính là 0; τ = 1/50 hòa thì giữ η chung; η riêng của B1; μ theo QS-2.
    - Chỉ đọc lượt có `final_eval_split = validation`; gặp lượt `test` thì báo lỗi (G8, G25).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_select_lr.py`
  - Phụ thuộc: T15a, T15b
  - File: `analysis/select_lr.py`, `tests/unit/test_select_lr.py`

- [x] **T11c · Baseline và biến thể trộn + profile fast** — TV2 · M
  - Mô tả: thêm tùy chọn cấu hình cho các baseline và tạo `configs/profiles/fast.yaml`.
    - B1 và B1-4k: một node giữ đủ 55k mẫu, không qua topology hay mixing (W = [1], không có message, spectral gap `None`); τ·R = 17,200 (10 epoch là 17,188 bước, làm tròn lên bội của 100) và 4,000; 100 chia hết cho τ; lịch đánh giá là các bội của 100 tới hết ngân sách của chính lượt đó.
    - B2: W = I.
    - B3: đường server.
    - Biến thể trộn cho E4: uniform (W uniform, π ∝ $d_j+1$) và size-corrected (MH, `loss_scale` $c_i = n_i/\bar n$ tính một lần trước khi huấn luyện, π = $n_j/n$; chỉ với μ = 0).
  - Chấp nhận:
    - Mỗi baseline và biến thể chạy được trên dữ liệu tổng hợp, ghi đúng số bước, số message (B1 và B2 không có; B3 có 2N/round) và `objective_weights`.
    - Demo C3: `python -m dfl run --config configs/profiles/fast.yaml --out build/fast` chạy được cho ring và FC trên IID.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_baselines.py`
  - Phụ thuộc: T11b, T10b
  - File: `src/dfl/config.py`, `src/dfl/sim/run.py`, `configs/profiles/fast.yaml`, `tests/integration/test_baselines.py`

- [x] **Checkpoint C3 (M3)** — sau T11c, T11d, T11e (plan §7). Nhóm duyệt ngày 2026-10-02 (plan §13); còn mở, làm trước E0a: T11e; smoke trên laptop demo (QS-8); sửa P0-1, I1, I2, I6, S1 và thêm test P0/P1 theo docs/decisions/REVIEW-C3-P3.7.md, REVIEW-C3-P4.1.md (đã sửa theo P3.4: P0-1 ở 7ae771f, S1 ở 27b890a, I1 ở 63bbada, I2 ở d583df7; đã thêm test P0-2…P0-4 ở c709814; đã cập nhật README; còn: phần còn lại của I6, test P1):
  - `make check-full` và `make check-inv`, đủ 12 T-INV;
  - pilot trên MNIST thật, ngoài lưới, ghi ra `build/`: B1 và DFedAvg-FC-IID với profile fast, chỉ đánh giá trên validation (không nhìn test), để thấy sớm nếu accuracy thấp bất thường so với SAN-1, SAN-2 (F6, P2.2 vòng 2);
  - demo profile fast cho ring và FC;
  - P3.7 (subagent `test-engineer`);
  - review diff;
  - cài thử và chạy smoke trên laptop demo (QS-8);
  - tăng `MIN_TESTS`;
  - nhóm duyệt.

## Tuần 7 — Sweep, hiệu năng, hình, E0a (mọi code `src/` phải xong trước T14b)

- [x] **T12 · Sweep** — TV3 · M · ∥ T16a
  - Mô tả: `sim/sweep.py` và lệnh `sweep`.
    - YAML thí nghiệm → lưới → run_id; chỉ mục `results/experiments/<exp>.json` theo từng `code_version`.
    - Bỏ qua run_id đã xong. Tự chạy lại attempt bị ngắt. Chỉ chạy lại attempt `failed` khi có `--retry-failed`.
    - Kiểm cây sạch và môi trường. Khóa theo run_id, và bỏ qua lượt đang bị khóa.
    - Mỗi lượt một tiến trình con mới (không `fork`), có biến môi trường số luồng; tiến trình con kiểm lại `code_version`.
    - Có `--workers`, `--dry-run`, và `--only key=value` để chạy trước một phần lưới (ví dụ các ô α = 100 của E1 để có SAN-4 sớm).
    - `fcntl` chỉ import khi ghi vào `results/`. Định dạng chỉ mục `results/experiments/<exp>.json` là một TypedDict trong `contracts.py` (thêm test ở `test_contracts_smoke.py`), để T15a đọc theo đúng schema.
  - Chấp nhận:
    - ENG-9 đủ các ý ở SPEC §9.4: idempotent và chạy lại được sau khi bị ngắt; attempt `failed` không tự chạy lại; từ chối khi cây bẩn hay môi trường lệch; `code_version` đổi giữa chừng thì ghi `failed` và dừng; commit tài liệu không làm đổi `code_version`; `--dry-run` đếm đúng.
    - Kết quả không phụ thuộc `--workers` (1 và 2 worker cho `rounds.jsonl` giống hệt, trên profile smoke). Hai sweep dùng chung một run_id chỉ chạy lượt đó một lần. `sweep --out build/<dir>` chạy được (cho demo) và vẫn đòi cây sạch.
    - Ghi vào `results/` thì sweep từ chối khi không chạy từ một worktree check out một commit sạch (SPEC §5.3). Tiến trình con được tạo bằng `subprocess` hoặc `spawn`, có `PYTHONPATH=<worktree>/src`, và báo lỗi nếu `dfl.__file__` không nằm trong worktree (test dựng một worktree tạm).
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_sweep.py`
  - Phụ thuộc: T11b, T11d
  - File: `src/dfl/sim/sweep.py`, `src/dfl/cli.py`, `tests/integration/test_sweep.py`, `src/dfl/contracts.py`, `tests/unit/test_contracts_smoke.py`

- [ ] **T13 · Đo hiệu năng, chốt giao thức** — TV2 · M · ∥ T16a
  - Mô tả: đo dưới tải thật (H P3.5): thời gian một bước, một lần đánh giá, một lượt fast và full; số luồng {1, 2, 4, 8} kết hợp số worker. Đo thêm process runtime (N = 8, tiến trình cục bộ và Docker: thời gian một round, RAM) và thời gian một bước của M0–M3 (ADR-011), để chốt R của E8. Xác nhận lịch đánh giá, tập con và số luồng đã ghi ở `configs/base.yaml` (ADR-007) còn chạy vừa ngân sách, và tạo profile full. Muốn đổi bất kỳ thứ nào trong đó thì phải hỏi nhóm và có ADR mới (Ask first) trước E0a. Tối ưu nào cần sửa `src/` thì tách thành task riêng, có test, và phải xong trước T14b. Cập nhật ngân sách ở `tasks/plan.md` §10.
  - Chấp nhận:
    - `docs/PERF.md` có bảng đo (trước và sau mỗi tối ưu) và số lõi rảnh thực tế.
    - Ngân sách Tier 1 + 2 tính theo số đo mới. Đạt nếu Tier 1 + 2 vừa 3 tuần (tuần 8–10) với số lõi rảnh đo được; không đạt thì đề xuất cắt theo ADR-007 cho nhóm quyết. Không đổi lịch sau E0a.
  - Kiểm chứng: `make check-task`; lệnh đo ghi trong `docs/PERF.md` chạy lại được
  - Phụ thuộc: T11c, T11d, T12, T22b, T09c
  - File: `docs/PERF.md`, `configs/base.yaml`, `configs/profiles/full.yaml`, `tasks/plan.md`

- [ ] **T16a · Khung vẽ hình + PROVENANCE + P1** — TV3 · M · ∥ T13
  - Mô tả: khung chung của `analysis/plots.py` gồm bảng màu, đơn vị, và một `code_version` mỗi hình.
    - `figures/PROVENANCE.json` ghi lệnh, `code_version`, `analysis_version`, commit và run_id.
    - Ghi vào `figures/` khi `analysis/` bẩn thì từ chối. Không có `--draft` thì chỉ đọc `results/`.
    - `--draft` nới đúng các điều ở SPEC INT-1, và gắn nhãn DRAFT.
    - Có P1 (learning curve) và P7 từ manifest. Hình chưa đủ dữ liệu (nhóm thí nghiệm chưa chạy) thì bỏ qua và in tên ra; không ghi đè hình cũ bằng hình rỗng.
  - Chấp nhận:
    - e2e: `analysis.plots --draft` vẽ được từ kết quả smoke. Không có `--draft` thì từ chối vì `--results` nằm ngoài `results/`; kết quả test không phụ thuộc trạng thái git.
    - PROVENANCE có `analysis_version`; ghi vào `figures/` khi `analysis/` bẩn thì báo lỗi.
  - Kiểm chứng: `make check-full`; `pytest -q tests/unit/test_plots.py tests/e2e/test_plots_smoke.py`
  - Phụ thuộc: T15a, T15b, T05b, T11b
  - File: `analysis/plots.py`, `analysis/provenance.py`, `tests/unit/test_plots.py`, `tests/e2e/test_plots_smoke.py`

- [ ] **T14b · E0a: chọn η (24 lượt)** — TV3 · M
  - Mô tả:
    - Chuẩn bị: thẻ tiền đăng ký E0a (mẫu H§5.3), `configs/experiments/e0a_lr.yaml`, dòng đầu tiên của `docs/experiments/CODE_VERSIONS.md`, và worktree của lưới.
    - Chạy sweep với seed 100, chỉ đánh giá trên validation.
    - Chọn η chung và η riêng của B1 bằng T14a. μ không được chọn bằng lượt chạy: nó tính từ η theo công thức của QS-2 (giữ ημτ ∈ {0.1, 1} ở τ = 20). Điền cả ba vào ADR-007, kèm ngày và run ID; đây là bước điền đã đăng ký trước, và nhóm duyệt diff của ADR (Ask first).
  - Chấp nhận:
    - 24 run_id có attempt xong, cùng `code_version`; nhóm đã duyệt kết quả `--dry-run`, vì lưới hơn 10 lượt.
    - η chung, μ và η của B1 ghi vào ADR-007 trước khi chạy E0b; không lượt nào đánh giá trên test.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e0a_lr.yaml`; `python -m analysis.select_lr --results results/`
  - Phụ thuộc: T12, T13, T14a, T14e, T11e, T22b, T09c (Q1 đã trả lời: process runtime là code của lưới); G1 (ADR đã Accepted); `make check-inv` xanh; cây sạch
  - File: `docs/experiments/E0a.md`, `configs/experiments/e0a_lr.yaml`, `docs/experiments/CODE_VERSIONS.md`, `docs/decisions/ADR-007-evaluation-protocol.md`, `configs/base.yaml` (η, μ, η của B1 cho E0b trở đi)

## Tuần 8–9 — Tier 1 (C4)

- [ ] **T14c · E0b: sanity + baseline (43 lượt)** — TV3 · S
  - Mô tả: thẻ E0b và `configs/experiments/e0b_sanity.yaml`. Chạy B1, B1-4k, B2, B3, DFedAvg IID × T1–T4, và phép kiểm độ nhạy η/2, 2η. Kiểm SAN-1…3 và SAN-5, rồi ghi kết quả vào thẻ.
  - Chấp nhận:
    - 43 run_id xong. SAN-1…3 và SAN-5 đạt, hoặc làm theo quy trình SPEC §9.3 (dừng, debug, báo nhóm).
    - Kết quả η/2, 2η chỉ được mô tả trong thẻ; không đổi η.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e0b_sanity.yaml`; lệnh kiểm SAN-1…3 và SAN-5 của `analysis.sanity`
  - Phụ thuộc: T14b
  - File: `docs/experiments/E0b.md`, `configs/experiments/e0b_sanity.yaml`

- [ ] **T17a · E1: ảnh hưởng Non-IID (70 lượt)** — TV3 chạy, TV1 phân tích · S · ∥ T19a, T20a
  - Mô tả: thẻ E1 và cấu hình. Chạy các ô α = 100 trước để kiểm SAN-4, rồi đủ lưới. Đối chiếu H1 theo phép so sánh chính đã khai báo; vẽ P1, P2, P4.
  - Chấp nhận:
    - 70 run_id xong. SAN-4 và SAN-5 đạt (hoặc làm theo quy trình §9.3) trước khi chạy E2.
    - H1 có kết luận theo quy tắc ADR-007, ghi vào thẻ kèm hình và run ID.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e1_noniid.yaml`; lệnh kiểm SAN-4 của `analysis.sanity`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T14c, T15c, T16a, T16b
  - File: `docs/experiments/E1.md`, `configs/experiments/e1_noniid.yaml`

- [ ] **T20a · E4a: quantity skew (6 lượt)** — TV3 chạy, TV1 phân tích · S · ∥ T17a, T19a
  - Mô tả: P3 β = 0.5 × {ER, FC} × MH × 3 seed.
  - Chấp nhận:
    - 6 run_id xong. Lưới có ≤ 10 lượt, nên không cần duyệt `--dry-run`.
    - Báo cáo kèm kích thước node và lệch mục tiêu (G11, G19).
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e4a_quantity.yaml` (bỏ bước 2); `python -m analysis.aggregate --results results/ --out build/analysis`
  - Phụ thuộc: T14c
  - File: `docs/experiments/E4a.md`, `configs/experiments/e4a_quantity.yaml`

- [ ] **T16b · Hình P2–P5** — TV1 · S · ∥ T14c
  - Mô tả: P2, P3, P4, P5 (KH §3.8), cần cho E1 và E2. Làm trong lúc E0b đang chạy.
  - Chấp nhận:
    - Mỗi hình sinh được từ kết quả mẫu, có đơn vị và bảng màu an toàn, và có mục trong PROVENANCE.
    - Hình trộn `code_version`, lệch biến cố định hay lệch tập seed thì báo lỗi (dùng `checks.py`).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_plots.py -k "p2 or p3 or p4 or p5"`
  - Phụ thuộc: T16a, T15c
  - File: `analysis/plots.py`, `tests/unit/test_plots.py`

- [ ] **T16c · Hình P6, P8, P9 và bảng T1** — TV1 · S · ∥ T17a
  - Mô tả: P6, P8, P9 và bảng T1 (KH §3.8). Gossip vẽ riêng theo tốc độ trộn kỳ vọng (P8). Các so sánh thuật toán dùng trục bước cục bộ và trục byte (G10).
  - Chấp nhận:
    - Mỗi hình và bảng T1 sinh được từ kết quả mẫu, có đơn vị và bảng màu an toàn, và có mục trong PROVENANCE.
    - P8 không đặt gossip chung trục spectral gap; P6 và P9 có trục bước cục bộ và trục byte.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_plots.py -k "p6 or p8 or p9 or t1"`
  - Phụ thuộc: T16b
  - File: `analysis/plots.py`, `tests/unit/test_plots.py`

- [ ] **T19a · E3a: ba thuật toán (24 lượt)** — TV3 · S · ∥ T17a, T20a
  - Mô tả: D-PSGD (τ = 1) và DFedProx (μ theo QS-2) × {ring, FC} × {IID, α = 0.1} × 3 seed. So theo bước cục bộ và theo byte (G10); vẽ P9.
  - Chấp nhận:
    - 24 run_id xong; REQ-08 có đủ ba thuật toán.
    - Kết luận tối đa ở mức "gợi ý" (3 seed), và chỉ cho khuyến nghị tạm.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e3a_algorithms.yaml`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T14c, T16c
  - File: `docs/experiments/E3a.md`, `configs/experiments/e3a_algorithms.yaml`

- [ ] **T18a · E2: topology × Non-IID (30 lượt)** — TV3 · S
  - Mô tả: {ER, gossip $K_N$} × {α = 1, α = 0.1, #C = 2} × 5 seed. Kiểm định chính H2 trên 6 mức Dirichlet kèm IID; H3 và một phần H4; RQ2 để khám phá. Vẽ P3, P5, P8.
  - Chấp nhận:
    - 30 run_id xong.
    - H2 kết luận đúng quy tắc: 5 độ dốc, CI t với df = 4. RQ2 chỉ mô tả, vì spectral gap chỉ có ba cụm giá trị.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e2_topology.yaml`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T17a (SAN-4), T16b, T16c
  - File: `docs/experiments/E2.md`, `configs/experiments/e2_topology.yaml`

- [ ] **Checkpoint C4** — sau T18a, T19a, T20a và T27b, tức đủ 269 lượt của Tier 1 (EXP-2: 197 lượt lưới lõi và 72 lượt E8): `make check-full`; hình Tier 1 từ `results/`; SAN-1…5 đã đạt; sao lưu `results/`; review; nhóm duyệt trước khi mở Tier 2

## Tuần 10 — Tier 2 (thứ tự ngược với thứ tự cắt: E3b → E1c → E4 → E2b) (C5, M4)

- [ ] **T19b · E3b: tần suất giao tiếp + prox (42 lượt)** — TV3 · S
  - Mô tả:
    - μ = 0 với τ ∈ {5, 50} × {ring, FC} × {IID, α = 0.1}.
    - μ nhỏ (μ = 0.1 nếu η = 0.05; nói chung tính từ η theo QS-2, ghi ở ADR-007) với τ = 20.
    - μ lớn (μ = 1 nếu η = 0.05, theo QS-2) với τ = 50 × α = 0.1.
    - Phục vụ RQ3, H5, H7; vẽ P6, P9. Các ô τ = 20 và τ = 1 lấy lại với seed {0, 1, 2}.
  - Chấp nhận:
    - 42 run_id xong.
    - Kết luận ghi kèm hạn chế: η dùng chung cho mọi τ, và accuracy chỉ đo sau tổng hợp (INT-4).
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e3b_tau_mu.yaml`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T19a, C4
  - File: `docs/experiments/E3b.md`, `configs/experiments/e3b_tau_mu.yaml`

- [ ] **T17b · E1c: đối chứng lệch mục tiêu (15 lượt)** — TV3 chạy, TV1 phân tích · S · ∥ T19b
  - Mô tả: B3 (đường server) × α ∈ {100, 1, 0.5, 0.1, 0.05} × 3 seed, để kiểm H6 (G19).
  - Chấp nhận:
    - 15 run_id xong.
    - Ô B3 được kiểm như mọi lượt DFL khi so sánh (INT-2); H6 kết luận theo quy tắc.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e1c_objective.yaml`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T17a, C4
  - File: `docs/experiments/E1c.md`, `configs/experiments/e1c_objective.yaml`

- [ ] **T20b · E4: quantity + trọng số trộn (15 lượt)** — TV3 chạy, TV1 phân tích · S
  - Mô tả:
    - β = 0.5: ER × {uniform, size-corrected}, và FC × size-weighted qua đường server.
    - α = 0.1: ER × {uniform, size-corrected}.
    - Phục vụ RQ4, H6.
  - Chấp nhận:
    - 15 run_id xong.
    - Khuyến nghị về trọng số trộn chỉ dựa trên MH so với uniform. Size-corrected được báo cáo riêng, như một thay đổi bước học theo node (ADR-005).
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e4_quantity.yaml`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T20a, T17b, T18a, T11c, C4
  - File: `docs/experiments/E4.md`, `configs/experiments/e4_quantity.yaml`

- [ ] **T18b · E2b: gossip công bằng (9 lượt)** — TV3 · S · ∥ T20b
  - Mô tả: T3b × {IID, α = 0.1, #C = 2} × 3 seed, để kiểm H4.
  - Chấp nhận:
    - 9 run_id xong (≤ 10 lượt).
    - H4 so theo byte với ring và ER (G10).
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e2b_gossip.yaml` (bỏ bước 2); `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T18a, C4
  - File: `docs/experiments/E2b.md`, `configs/experiments/e2b_gossip.yaml`

- [ ] **T19c · Lượt có điều kiện: η riêng cho τ = 1 hoặc τ = 50 (≤ 24 lượt)** — TV3 · S · chỉ làm khi quy tắc của ADR-007 kích hoạt ở T14b
  - Mô tả: chạy lại các ô D-PSGD của E3a và/hoặc các ô μ = 0, τ = 50 của E3b với η riêng, làm kết quả bổ sung.
  - Chấp nhận:
    - Các lượt xong. Báo cáo ghi rõ đây là kết quả bổ sung; kết luận chính vẫn dùng η chung.
    - Nếu điều kiện không kích hoạt thì đánh dấu task là không áp dụng và ghi lý do.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e3_extra_lr.yaml`
  - Phụ thuộc: T14b (điều kiện); phần τ = 1 (Tier 1) phụ thuộc T19a, phần τ = 50 phụ thuộc T19b
  - File: `docs/experiments/E3extra.md`, `configs/experiments/e3_extra_lr.yaml`

- [ ] **Checkpoint C5 (M4)** — sau T19b, T17b, T20b, T18b, T19c, hoặc sau quyết định cắt phần Tier 2 còn lại (EXP-5; task bị cắt được đánh dấu "không làm" kèm lý do):
  - `make check-full`;
  - toàn bộ P1–P9 và T1 từ `results/`;
  - doubt-driven cho mỗi kết luận chính (INT-5);
  - P3.7 lần hai;
  - sao lưu `results/`;
  - nhóm duyệt.

## Tuần 11–12 — Demo, báo cáo, thẩm định (M5, M6)

- [ ] **T26 · Demo trên laptop** — TV1 · M · ∥ T24
  - Mô tả: `configs/profiles/demo.yaml` và `configs/experiments/demo.yaml` chạy 2 lượt, ring và FC ở α = 0.1. Nếu laptop có Docker (QS-8, Q12) thì demo thêm một lượt nhỏ, mỗi node một container (`make check-docker`). `docs/DEMO.md` ghi các bước: cài không dùng file khóa, tải MNIST trước, chạy sweep ra `build/demo`, vẽ hình bằng `--draft`. Đo thời gian trên laptop thật (QS-8).
  - Chấp nhận:
    - ENG-11: cả chạy lẫn vẽ hình xong trong ≤ 5 phút trên laptop, không cần mạng.
    - Chạy được trên hệ điều hành của laptop; ghi `env_match: false` như dự kiến.
  - Kiểm chứng: chạy theo `docs/DEMO.md` trên laptop, bấm giờ
  - Phụ thuộc: T12, T16a
  - File: `configs/profiles/demo.yaml`, `configs/experiments/demo.yaml`, `docs/DEMO.md`

- [ ] **T24 · Báo cáo và khuyến nghị** — TV3 viết chính, cả nhóm cùng viết · M
  - Mô tả: báo cáo tiếng Việt. Có sơ đồ kiến trúc (REQ-11). Khuyến nghị ghi mức bằng chứng: chính khi "ủng hộ", tạm khi "gợi ý" (REQ-13). Có mục hạn chế đủ theo INT-4, phụ lục attempt `failed`/`diverged` (INT-3), và phụ lục các `code_version` bị thay thế (INT-6).
  - Chấp nhận:
    - Mọi nhận định dẫn chiếu hình hay bảng và run ID; mọi hình trong báo cáo có trong PROVENANCE.
    - Kết quả trái giả thuyết vẫn được báo cáo; H nào bị cắt vì ngân sách thì ghi "chưa kết luận" (EXP-5).
  - Kiểm chứng: review của nhóm; P4.3 (H§3.4)
  - Phụ thuộc: C5
  - File: `docs/report/bao_cao.md`, `docs/report/kien_truc.md`, `docs/report/phu_luc.md`

- [ ] **T25 · Thẩm định cuối + tag v1.0** — TV2 · S
  - Mô tả: REP-1: clone sạch, check out `git_commit`, cài từ file khóa, chạy lại 2 run_id ngẫu nhiên, rồi `cmp`. Review fan-out bằng `code-reviewer`, `test-engineer`, và `security-auditor` (phạm vi hẹp). Ghi `docs/VALIDATION.md`, rồi tag v1.0.
  - Chấp nhận:
    - REP-1 đạt: `rounds.jsonl` và `summary.json` trùng từng byte.
    - Đạt checklist H§3.4 và Definition of Done; nhóm duyệt Gate G3.
  - Kiểm chứng: lệnh REP-1 ghi trong `docs/VALIDATION.md`; `make check-full`. Review ở T25 mà cần sửa `src/` thì chỉ sửa trên nhánh chính và ghi vào mục hạn chế; không chạy lại lưới, trừ khi đó là lỗi làm sai kết quả (khi đó theo SPEC §8)
  - Phụ thuộc: T24, T26
  - File: `docs/VALIDATION.md`

## Đa tiến trình, Docker, thang model và E8 — theo trả lời Q1 (ADR-010, ADR-011; nhóm duyệt 2026-10-01)

T22a, T22b và T09c là code của lưới: phải xong trước T14b (E0a), để lưới, E8 và demo cùng một `code_version` (K7). T22c (Docker) là tùy chọn (Q12) và nằm ngoài `src/`, nên không chặn E0a.

- [x] **T22a · `TorchDistTransport` (gloo)** ⚠ — TV2 · M · ∥ T09c, T11a
  - Mô tả: Transport cho một tiến trình node ở `sim/torchdist.py` (ADR-010).
    - Mã hóa state thành một tensor liền (thứ tự khóa cố định) và giải mã lại.
    - Mỗi round, trao đổi đúng một message với mỗi láng giềng của G_t theo thứ tự không deadlock: id nhỏ gửi trước. Dùng `isend`/`irecv` nếu kiểm được trên gloo; nếu không thì ghi UNVERIFIED và dùng `send`/`recv`.
    - CommLedger đếm message và byte node gửi và nhận.
  - Chấp nhận:
    - Unit: mã hóa rồi giải mã trùng từng bit mọi tensor (float32, float64, int64); `nbytes` không đổi.
    - Integration, 4 tiến trình cục bộ (`spawn`) trên ring và FC: mỗi node nhận đúng snapshot của láng giềng, trùng từng bit với cái đã gửi; số message và byte của từng node bằng InMemoryTransport trên cùng đồ thị; không tự gửi; không treo (có timeout).
    - Primitive đã chọn có URL tài liệu torch 2.14, hoặc ghi UNVERIFIED kèm lý do (bảng backend không có `isend`/`irecv` cho gloo).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_torchdist_codec.py tests/integration/test_torchdist.py`
  - Phụ thuộc: T10c
  - File: `src/dfl/sim/torchdist.py`, `tests/unit/test_torchdist_codec.py`, `tests/integration/test_torchdist.py`

- [x] **T22b · Process runtime: mỗi node một tiến trình, cộng observer** ⚠ — TV2 · M
  - Mô tả: `sim/process_runtime.py` (ADR-010).
    - Vòng lặp của tiến trình node: cùng một round ATC bằng `Node`, `combine`, `make_snapshot` và TorchDistTransport; G_t tự tính từ seed chung (G18); trước round đầu gọi `check_union_connected` như `run_rounds` (INV-4 của gossip); kiểm hữu hạn và kiểm inbox như engine; dừng mọi tiến trình khi phân kỳ hay lỗi.
    - Vòng lặp của observer: nhận state và số đếm ở mốc đánh giá qua kênh đo (không tính phí), dựng RoundRecord, ghi kết quả bằng RunLogger.
    - `python -m dfl run --runtime processes` khởi chạy N + 1 tiến trình cục bộ; mỗi tiến trình chạy `python -m dfl node --rank k --world-size W --master-addr …`, đúng lệnh mà container Docker dùng (T22c). TimingRecord thêm `comm_s`.
  - Chấp nhận:
    - T-INV-13 (khi SPEC đã thêm, Ask first): với smoke N = 4 trên ring, gossip và FC, `rounds.jsonl` trùng từng byte với bản in-process cùng cấu hình và seed.
    - Một node phân kỳ thì mọi tiến trình dừng gọn (không treo), và SummaryDiverged có cùng `diverged_at_round` như bản in-process.
    - Message và byte của lượt bằng bản in-process (T-INV-6 ở N = 4); kênh tới observer không được đếm.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_process_runtime.py`
  - Phụ thuộc: T22a, T11b
  - File: `src/dfl/sim/process_runtime.py`, `src/dfl/sim/run.py`, `src/dfl/contracts.py` (TimingRecord.comm_s), `tests/unit/test_contracts_smoke.py`, `tests/integration/test_process_runtime.py`

- [ ] **T22c · Docker: mỗi node một container (tùy chọn, Q12)** — TV3 · M · ∥ T12, T13
  - Mô tả:
    - `docker/Dockerfile`: python:3.12-slim, cài torch CPU và file khóa, chạy bằng user không phải root.
    - `docker/compose.py`: sinh file Compose cho N node và một observer trên một bridge network; `data/` gắn chỉ đọc.
    - `make check-docker` chạy smoke trong container và so với bản in-process (T-INV-13 trong Docker); ghi số byte mạng của container (`docker stats`).
    - `docs/DOCKER.md`: build, chạy, tài nguyên.
    - Thêm `make check-docker` vào bảng của CONSTRAINTS.md cần nhóm duyệt (Ask first).
  - Chấp nhận:
    - `make check-docker` (N = 4, smoke): `rounds.jsonl` trùng từng byte với bản in-process; byte mạng của container ≥ byte của CommLedger, và tỉ lệ được ghi lại.
    - Image cài đúng file khóa (`pip check` sạch trong container); lúc chạy không cần mạng ngoài (MNIST gắn từ `data/`).
    - Một lượt fast N = 8 trong Docker chạy xong; thời gian và RAM ghi vào `docs/DOCKER.md`.
  - Kiểm chứng: `make check-docker` (chạy tay, ghi kết quả vào `docs/DOCKER.md`); `make check-task`; `pytest -q tests/unit/test_compose.py`
  - Phụ thuộc: T22b
  - File: `docker/Dockerfile`, `docker/compose.py`, `Makefile`, `docs/DOCKER.md`, `tests/unit/test_compose.py`

- [x] **T09c · Thang model M0–M3 (ADR-011)** — TV1 · M · ∥ T22a
  - Mô tả: `learning/models.py` thêm M0 (logistic regression), M2 (2NN-MLP) và M3 (CNN của McMahan et al. 2017) bên cạnh SmallCNN, cùng `build_model(name, generator)`. Cấu hình có trường `model`; mặc định là SmallCNN, nên thí nghiệm lõi không đổi.
  - Chấp nhận:
    - Số tham số đúng 7,850 / 46,730 / 199,210 / 1,663,370; không có BatchNorm hay Dropout.
    - Tạo model không đổi `torch.get_rng_state()`; khác seed thì khác w⁰ ở mọi tensor; đầu ra shape (B, 10).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_models.py`
  - Phụ thuộc: T09a, T02b
  - File: `src/dfl/learning/models.py`, `src/dfl/config.py`, `tests/unit/test_models.py`

- [ ] **T27a · E8: thẻ tiền đăng ký + cấu hình** — TV3 · S · ∥ T14c
  - Mô tả: thẻ `docs/experiments/E8.md` ghi H8 và quy tắc kết luận (ADR-011). `configs/experiments/e8_model_complexity.yaml` có lưới {M0, M1, M2, M3} × {T1, T3, T4} × {IID, α = 0.1} × 3 seed = 72 lượt, với N = 8, τ = 20, R theo T13, `runtime: processes`, `deploy: local`; nếu có T22c thì thêm 4 ô {M1, M3} × {T1, T4} × IID, seed 0, với `deploy: docker`. Thẻ được commit trên nhánh `grid` trước khi chạy (EXP-1).
  - Chấp nhận:
    - Thẻ có trước mọi lượt E8 trong lịch sử git.
    - `--dry-run` báo 72 lượt, và nhóm đã duyệt, vì lưới hơn 10 lượt.
  - Kiểm chứng: `python -m dfl sweep --experiment configs/experiments/e8_model_complexity.yaml --dry-run`
  - Phụ thuộc: T22b, T09c, T14b
  - File: `docs/experiments/E8.md`, `configs/experiments/e8_model_complexity.yaml`

- [ ] **T27b · E8: chạy 72 lượt (tiến trình cục bộ; 4 ô thêm trong Docker nếu có)** — TV3 · S · ∥ T17a, T19a
  - Mô tả: chạy theo quy trình thí nghiệm, từ worktree của lưới, với `deploy: local`, cùng 4 ô Docker nếu T22c đã xong; ghi tải của máy chủ trong lúc chạy. Chạy lại 2 ô M1 bằng in-process và so `rounds.jsonl` (T-INV-13 trên dữ liệu thật, N = 8).
  - Chấp nhận:
    - 72 run_id có attempt xong (`ok` hoặc `diverged`), cùng một `code_version`.
    - Hai bản in-process trùng từng byte với bản process runtime; nếu có 4 ô Docker thì số liệu `docker stats` được ghi lại.
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e8_model_complexity.yaml`; `cmp` hai cặp `rounds.jsonl`
  - Phụ thuộc: T27a
  - File: `docs/experiments/E8.md`

- [ ] **T27c · E8: phân tích, hình P10–P11** — TV1 · M · ∥ T18a
  - Mô tả: `analysis/comm_cost.py` tính byte/round, `comm_s`, tỉ lệ giao tiếp trên tính toán theo số tham số và topology. Hai hình mới: P10 (byte/round và `comm_s` theo số tham số, trục log) và P11 (accuracy theo byte tích lũy). Kết luận H8 ghi vào thẻ.
  - Chấp nhận:
    - P10, P11 sinh bằng một lệnh từ `results/`, mỗi hình một `code_version`.
    - Mỗi H8 có kết luận theo quy tắc ghi trong thẻ.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_comm_cost.py`; `python -m analysis.plots --results results/ --out figures/`
  - Phụ thuộc: T27b, T16a
  - File: `analysis/comm_cost.py`, `analysis/plots.py`, `tests/unit/test_comm_cost.py`, `docs/experiments/E8.md`

## Mở rộng — chỉ làm sau M4, và phải hỏi trước (SPEC §1.4, QĐ-4)

- [ ] T21 E6 Fashion-MNIST và E5 quy mô N ∈ {10, 50} (E5 chỉ đổi cấu hình)
- [ ] T23 Gossip bất đồng bộ, E7
