# Checklist triển khai — Đề 2 (Non-IID × DFL)

> Kế hoạch, quyết định, đồ thị phụ thuộc, rủi ro: `tasks/plan.md`. Trạng thái: nhóm đã duyệt (Gate G2, 2026-09-30).
> **Cách làm một task** (H P3.0, lệnh `/build`):
> 1. Chọn task chưa làm đầu tiên mà mọi phụ thuộc đã xong.
> 2. Viết test FAIL trước, rồi code tối thiểu cho test PASS.
> 3. `make check-task` phải xanh.
> 4. Đánh dấu `[x]` và ghi `tasks/SESSION_LOG.md`.
> 5. Commit đúng một commit, chỉ `git add` file của task. Ngoại lệ: task thí nghiệm có hai commit, xem quy trình thí nghiệm ở dưới.
>
> Test bất biến: chỉ task chủ của T-INV-k (plan §8) đặt tên test dạng `test_t_inv_<k>_…` (SPEC §7). Task khác có thể có test thành phần liên quan nhưng không dùng tên đó. `make check-inv` (có từ P1.8) kiểm đủ 12 ID.
> Task nào cần thêm hay đổi schema trong `src/dfl/contracts.py` thì thêm file đó vào commit của mình và sửa `tests/unit/test_contracts_smoke.py` trước.
>
> "File" liệt kê tối đa 5 file, không tính `tasks/todo.md` và `tasks/SESSION_LOG.md`. Mọi task còn phải đạt Definition of Done (H§4.2).
> Phụ trách theo SPEC §3.1 và §10 dòng 1 (QS-9, nhóm chốt ở P2.1).
> ⚠ = rủi ro cao, làm sớm. ∥ = làm song song được với các task ghi sau nó.
>
> **Kiểm chứng một task thí nghiệm** (gọi tắt là "quy trình thí nghiệm"):
> 1. Commit thứ nhất: thẻ tiền đăng ký và cấu hình, trước khi chạy (EXP-1; `make check-task` xanh). Worktree của lưới lấy đúng commit này.
> 2. Lưới hơn 10 lượt thì trình kết quả `python -m dfl sweep --experiment <file> --dry-run` cho nhóm (Ask first).
> 3. Chạy `python -m dfl sweep --experiment <file> --workers <k>` từ worktree của lưới (SPEC §5.3).
> 4. Chạy lại `--dry-run`: phải báo 0 lượt sẽ chạy và 0 attempt `failed` chưa xử lý.
> 5. Chạy lệnh kiểm SAN-5 của `analysis.sanity` trên nhóm vừa chạy, rồi các lệnh `analysis` ghi ở task.
> 6. Commit thứ hai: kết quả và kết luận ghi vào thẻ, cùng `tasks/`. Lịch sử git cho thấy thẻ có trước kết quả.

## Điều kiện trước pha 3

- [x] P1.9 đã commit (`07b183c`)
- [ ] Gate G1: nhóm và GVHD duyệt SPEC, ADR-001…009 (Proposed → Accepted), trả lời Q1
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

- [ ] **T02b · Cấu hình + YAML base/smoke** — TV3 · M · ∥ T02a, T02c, T07, T04a
  - Mô tả: dataclass cấu hình frozen, đọc bằng `yaml.safe_load`, điền mặc định, kiểm kiểu và giá trị; JSON chuẩn hóa cho `run_id` (khóa sắp xếp, số thực theo `repr`). Tạo `configs/base.yaml` (mặc định KH §3.5, lưới 40 mốc, tập con 2,000, 1 luồng) và `configs/profiles/smoke.yaml` (SPEC §5.2).
  - Chấp nhận:
    - Cấu hình sai thì raise `ValueError` nêu ràng buộc bị vi phạm: kiểu sai (PyYAML đọc `5e-2` thành chuỗi), #C = k với N < 10, 100 không chia hết cho τ ở profile có lưới 100 bước.
    - Hai YAML tương đương (khác thứ tự khóa, `0.05` so với `5e-2`) cho cùng JSON chuẩn hóa. JSON này không gồm tên nhóm, `--out`, `--workers`.
    - `base.yaml` và `smoke.yaml` đọc được và hợp lệ.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_config.py`
  - Phụ thuộc: T01
  - File: `src/dfl/config.py`, `tests/unit/test_config.py`, `configs/base.yaml`, `configs/profiles/smoke.yaml`

- [ ] **T02c · `requirements.lock` + `.gitattributes`** — TV3 · M · ∥ T02a, T02b · cần QS-1
  - Mô tả: sinh `requirements.lock` trong venv chỉ có gói chạy (SPEC §4); thêm `.gitattributes` (`* text=auto eol=lf`); sửa lệnh cài đặt trong README và CLAUDE.md sang `-r`/`-c requirements.lock`.
  - Chấp nhận:
    - Venv mới cài đúng §4 thì `pip check` sạch và `make check-full` xanh. File khóa không có pytest, ruff, import-linter hay scikit-learn.
    - `git ls-files --eol '*.py'` cho `i/lf w/lf`.
    - Nhóm duyệt diff của CLAUDE.md, vì đây là file quy tắc.
  - Kiểm chứng: `make check-full`; cài thử trong venv tạm theo §4; `grep -Ei '^(pytest|ruff|import-linter|scikit-learn)' requirements.lock` không ra dòng nào
  - Phụ thuộc: T01; QS-1
  - File: `requirements.lock`, `.gitattributes`, `README.md`, `CLAUDE.md`

- [ ] **T07 · Mixing** ⚠ — TV2 · S · làm đầu tiên; ∥ T02a–c, T04a
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

- [ ] **T06 · Topology** — TV2 · S · ∥ T03, T04b
  - Mô tả: `network/topology.py` dựng T0 star, T1 ring, T2 ER (p = 0.3, lấy mẫu lại tới khi liên thông), T3 perfect matching trên $K_N$ mỗi round, T3b maximal matching trên cạnh ER, T4 FC, và lịch W = I cho B2. Đồ thị tĩnh được kiểm liên thông lúc dựng. `graph_at` là hàm thuần của (seed, round): lịch nhận một hàm `rng_for(round) -> numpy.random.Generator` do `sim` dựng từ `seeding.numpy_rng(seed, Stream.GOSSIP, round)` (ADR-009). `topology.py` không import `seeding`, vì `seeding` import torch (contract `pure-no-torch`). Không dùng `default_rng([seed, r])` như H P3.2 gợi ý (plan K10).
  - Chấp nhận:
    - Ring bậc 2; FC bậc N − 1; ER liên thông và tất định theo seed. T3: mỗi node tối đa một đối tác, N lẻ thì đúng một node đứng ngoài. T3b chỉ dùng cạnh ER và là maximal matching.
    - Gọi `graph_at(r)` nhiều lần, gọi xen `graph_at(r')`, hay dùng hai lịch tạo độc lập với cùng (seed, N) đều cho cùng đồ thị (SPEC REQ-01).
    - E[W] của T3 qua 10⁴ matching khớp ½I + (J − I)/(2(N − 1)) khi N chẵn, trong ±4 sai số chuẩn.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_topology.py`
  - Phụ thuộc: T02a, T07
  - File: `src/dfl/network/topology.py`, `tests/unit/test_topology.py`

- [ ] **T03 · Datasets + lệnh `download`** — TV1 · M · ∥ T06
  - Mô tả: `data/datasets.py` nạp MNIST qua torchvision thành tensor trong RAM và chuẩn hóa bằng mean/std của tập train. Validation 5k và tập con đánh giá 2,000 ảnh được chọn phân tầng bằng luồng dữ liệu cố định. Có bộ dữ liệu tổng hợp cùng shape cho test và smoke. Tạo khung CLI (`cli.py`, `__main__.py`) với `python -m dfl download mnist`. `cli.py` chỉ import module dùng torch bên trong từng lệnh con, để `run` còn đặt được biến môi trường số luồng trước khi torch được nạp (T11b).
  - Chấp nhận:
    - Trên dataset giả: tách 55k/5k phân tầng, đếm lớp đúng. Ảnh là float32 shape (n, 1, 28, 28), nhãn là int64. Cùng `DATA_SEED` thì cùng cách tách ở mọi seed của lượt.
    - Test không dùng mạng (ENG-8). Nếu dùng `.data`/`.targets` thì ghi UNVERIFIED như trong `contracts.py`.
    - `python -m dfl download mnist` tải được vào `data/` trên máy chủ (kiểm tay, cần mạng).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_datasets.py`; chạy tay `python -m dfl download mnist`
  - Phụ thuộc: T02a, T02b
  - File: `src/dfl/data/datasets.py`, `src/dfl/cli.py`, `src/dfl/__main__.py`, `tests/unit/test_datasets.py`

- [ ] **T04b · Partitioner P0–P3** — TV1 · M
  - Mô tả: `data/partition.py`, chỉ dùng numpy, cài P0–P3 đúng NIID-Bench với min_size = 64. Lấy mẫu lại tối đa 1,000 lần, quá thì raise `ValueError`; #C = k với N < 10 cũng raise `ValueError`. Nhận `numpy.random.Generator` qua tham số, không import `seeding`. Test golden so với hàm tham chiếu sao từ spike (H P3.1); phép so này chỉ kiểm bản cài khớp spike (SPEC REQ-02).
  - Chấp nhận:
    - T-INV-8: rời nhau, phủ đủ, mọi node ≥ min_size; #C = k đúng k nhãn và mọi nhãn có chủ; golden trùng tuyệt đối; N < 10 báo lỗi.
    - Trên mảng nhãn tổng hợp theo số đếm lớp của MNIST (55k, không tải dữ liệu), α = 1000 cho JS < 0.01, và JS (trung bình 3 seed) tăng đơn điệu khi α giảm.
    - `PartitionResult.retries` đúng; vượt 1,000 lần thì `ValueError`; contract `pure-no-torch` có thêm `dfl.data.partition`.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_partition.py -k t_inv_8`; `lint-imports`
  - Phụ thuộc: T02a, T04a
  - File: `src/dfl/data/partition.py`, `tests/reference/niid_bench_reference.py`, `tests/unit/test_partition.py`, `pyproject.toml`

- [ ] **T04c · Hoán vị vị trí node + P2' (shards)** — TV1 · S · ∥ T06
  - Mô tả: hoán vị partition vào vị trí node bằng Generator của luồng placement (KH §3.2), và P2' (shards của McMahan) theo QS-11, chỉ dùng trong `describe`.
  - Chấp nhận:
    - Cùng Generator thì cùng hoán vị; hoán vị là một hoán vị hợp lệ của 0..N−1 và không đổi kích thước hay nhãn của các phần.
    - P2': 2N mảnh sắp theo nhãn, mỗi node 2 mảnh, rời nhau và phủ đủ; test golden theo thứ tự tiêu thụ số ngẫu nhiên ghi trong docstring.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_partition_extras.py`
  - Phụ thuộc: T04b
  - File: `src/dfl/data/partition.py`, `tests/unit/test_partition_extras.py`

## Tuần 4 — C1, C2 (M2), engine bắt đầu

- [ ] **T05a · `dfl describe`: dữ liệu** — TV1 · M · ∥ T08
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

- [ ] **T08 · Transport + CommLedger + snapshot** — TV2 · M · ∥ T05a, T05b
  - Mô tả:
    - `network/transport.py`: in-memory, không copy payload. CommLedger đếm message và byte, tổng và theo từng node, gồm cả gửi lẫn nhận. Nhận server ảo có chỉ số N; cấm node tự gửi cho mình.
    - `make_snapshot` trong `sim/node.py`: clone + detach, `nbytes` tính từ tensor thật.
  - Chấp nhận:
    - T-INV-6 với N = 20: ring 40, FC 380, T3 20 message/round, ER đúng 2·|E|, đường server 2N. Byte = Σ `nbytes`; một state có shape của SmallCNN cho 4 × 46,730 byte. Tải = byte gửi + byte nhận. Không có message tự gửi.
    - Node chỉ đọc được inbox của chính nó. `end_round` giải phóng tham chiếu mà giữ nguyên số đếm.
    - `transport.py` không import torch (contract `pure-no-torch` xanh).
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_transport.py tests/unit/test_snapshot.py`
  - Phụ thuộc: T06
  - File: `src/dfl/network/transport.py`, `src/dfl/sim/node.py`, `tests/unit/test_transport.py`, `tests/unit/test_snapshot.py`

- [ ] **T08b · `dfl describe`: topology** — TV3 · S · ∥ T10a
  - Mô tả: mở rộng `describe` với bảng topology: spectral gap của T1, T4, và T2 trên 20 seed; số message/round dự kiến = 2·|E|, là thuộc tính của đồ thị, không phải phép đếm chi phí; T0 star; tốc độ trộn kỳ vọng của T3; ước lượng cho T3b từ 10⁴ matching. Chi phí thật chỉ do CommLedger đếm trong lượt chạy (INV-2); T-INV-6 (T08) kiểm hai con số khớp nhau.
  - Chấp nhận:
    - Ring 0.0326 (±1e-4), FC 1, ER trung bình trong [0.15, 0.25] trên 20 seed. Con số 0.312 của T3 được ghi là tốc độ trộn, không phải spectral gap.
    - Có test tích hợp ở N nhỏ. Lệnh chạy ở N = 20 in bảng so với KH §3.3 (demo của C2).
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_describe.py`
  - Phụ thuộc: T05a, T06, T07
  - File: `src/dfl/sim/describe.py`, `tests/integration/test_describe.py`

- [ ] **Checkpoint C2 (M2)** — sau T08, T08b: `make check-full`; bảng spectral gap sinh từ code; review diff; tăng `MIN_TESTS`; nhóm duyệt

- [ ] **T10a · Engine ATC lõi** ⚠ — TV2 · M · ∥ T09a
  - Mô tả:
    - `sim/node.py` và `sim/engine.py` chạy một round ATC cho topology tĩnh:
      1. Adapt: τ bước qua `LocalTrainer`.
      2. Mỗi node tạo đúng một snapshot.
      3. Combine: cộng theo id người gửi tăng dần vào bộ đệm mới.
      4. Chỉ ghi vào model sau khi mọi node đã tổng hợp xong.
    - W lấy từ `MixingPolicy`; B2 dùng W = I.
    - Test dùng MLP tí hon, dữ liệu tổng hợp 4 node, và một trainer theo Protocol, nên không cần T09 (plan K2).
    - `tests/conftest.py` có fixture `autouse` kiểm `torch.get_rng_state()` không đổi, chỉ áp cho test nằm trong `tests/integration/` và `tests/e2e/` (lọc theo đường dẫn của test); đây là hạ tầng của T-INV-5, test T-INV-5 thuộc T11b.
  - Chấp nhận:
    - Viết trước và thấy FAIL trước khi có code: T-INV-2 (W = I trùng khớp huấn luyện chỉ cục bộ) và T-INV-4 (đảo thứ tự duyệt node thì kết quả trùng từng bit, checksum mọi snapshot không đổi).
    - Chỉ trộn tensor float; không có phép tại chỗ trên snapshot; node chỉ đọc model của mình và inbox (INV-1, INV-3).
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_engine_core.py -k "t_inv_2 or t_inv_4"`
  - Phụ thuộc: T06, T07, T08
  - File: `src/dfl/sim/node.py`, `src/dfl/sim/engine.py`, `tests/conftest.py`, `tests/integration/conftest.py`, `tests/integration/test_engine_core.py`

- [ ] **T09a · SmallCNN + huấn luyện cục bộ** — TV1 (QS-9) · M · ∥ T10a
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

- [ ] **T09b · Độ đo phân loại + consensus + Evaluator** — TV1 (QS-9) · M · ∥ T10b, T11a
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

- [ ] **T10b · Gossip + đường server** ⚠ — TV2 · M · ∥ T09b, T11a
  - Mô tả:
    - Engine chạy với topology động: gọi `graph_at` mỗi round. Node không được ghép vẫn chạy Adapt và dùng W_ii = 1.
    - Đường server cho B3/E1c: server ảo chỉ số N nhận N snapshot, tổng hợp theo $n_k/n$, rồi gửi lại N snapshot qua Transport.
  - Chấp nhận:
    - T-INV-1, với lr = 0 và khởi tạo khác nhau. Topology tĩnh (ring, ER, FC): Ξ_t ≤ ρ^t Ξ_0 (1 + 1e-6) + 1e-12 và trung bình model được bảo toàn (MH). Gossip: mỗi W đối xứng, ngẫu nhiên kép, W² = W, và E[W] khớp trong sai số lấy mẫu. Test tự tính Ξ bằng công thức RMS làm oracle độc lập, không import `metrics/consensus.py`, nên T10b không phải đợi T09b.
    - T-INV-3: FC + size-weighted, đường server, và một FedAvg tham chiếu viết ngay trong test cho kết quả allclose (atol 1e-6). Khi kích thước node bằng nhau, FC + MH cũng vậy.
    - Test tích hợp (không mang tên T-INV): đường server của engine gửi đúng 2N message/round, và Transport là nơi đếm.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_engine_consensus.py tests/integration/test_engine_fedavg.py`
  - Phụ thuộc: T10a
  - File: `src/dfl/sim/engine.py`, `tests/integration/test_engine_consensus.py`, `tests/integration/test_engine_fedavg.py`

- [ ] **T11a · Manifest, `code_version`, RunLogger** — TV3 · M · ∥ T09b, T10b
  - Mô tả:
    - `io/manifest.py` dựng `ManifestRecord`:
      - `code_version`: hash dãy cặp (đường dẫn, nội dung), chuẩn hóa xuống dòng và file khóa;
      - `run_id`: hash của cấu hình chuẩn hóa + `code_version` + máy;
      - `git_commit`, `dirty` (tính cả file bị ignore trong `src/`, `configs/`);
      - kiểm môi trường theo `requirements.lock` để ra `env_match`;
      - máy: tên CPU kèm hash cờ CPU.
    - `io/run_logger.py`: tạo attempt nguyên tử, thứ tự start → log → finish, JSON `allow_nan=False`.
  - Chấp nhận:
    - `code_version` không đổi khi chỉ sửa docs/analysis/configs hay đổi CRLF ↔ LF. Nó đổi khi sửa hay đổi tên một file `.py` trong `src/`, hay khi sửa file khóa. `run_id` đổi khi cấu hình hay máy đổi.
    - Một gói lệch phiên bản so với file khóa thì `env_match: false`; công cụ dev cài thêm không làm lệch.
    - Ghi vào attempt đã có thì `FileExistsError`; gọi sai thứ tự thì `RuntimeError`; ghi NaN thì báo lỗi.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_manifest.py tests/unit/test_run_logger.py`
  - Phụ thuộc: T02b, T02c
  - File: `src/dfl/io/manifest.py`, `src/dfl/io/run_logger.py`, `tests/unit/test_manifest.py`, `tests/unit/test_run_logger.py`, `src/dfl/contracts.py` (nếu cần đổi schema)

- [ ] **T15b · R@x, AUC, độ dao động, mức sụt** — TV3 · S · ∥ T10b, T10c
  - Mô tả: `analysis/convergence.py` theo đúng SPEC REQ-04 và ADR-007: lưới 100 bước; chuỗi `acc_mean`; lần vượt bền vững đầu tiên; "≤ 100" và "> ngân sách"; AUC theo quy tắc hình thang. Độ dao động và mức sụt tính theo từng node.
  - Chấp nhận:
    - Unit test đủ các ca của REQ-04: đạt; không đạt; đạt ở mốc đầu; đạt rồi tụt rồi đạt lại (lấy lần bền vững); chỉ vượt ở mốc cuối (không đạt); lưới khác lưới chuẩn thì báo lỗi, trừ lượt `diverged` có lưới là phần đầu của lưới chuẩn: khi đó R@x là "không đạt", AUC là "không xác định (phân kỳ)", và lượt được đếm vào "k/n seed phân kỳ" (INT-3).
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

- [ ] **T10c · Observer + phản biện engine** ⚠ — TV2 · M
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
  - File: `src/dfl/sim/observer.py`, `src/dfl/sim/engine.py`, `src/dfl/sim/node.py`, `tests/integration/test_observer.py`

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
  - Phụ thuộc: T01 (test dùng `rounds.jsonl` giả theo `contracts.py`; khi đã có kết quả smoke thật thì chạy lại test với file thật)
  - File: `analysis/aggregate.py`, `analysis/checks.py`, `tests/unit/test_aggregate.py`, `tests/unit/test_checks.py`

- [ ] **T11b · `python -m dfl run` + smoke e2e** — TV2 · M
  - Mô tả:
    - `sim/run.py` lắp một lượt từ cấu hình: data → partition → placement → topology → mixing → engine → observer → logger, và ghi ra `--out`.
    - Lệnh `run` trong `cli.py` đọc cấu hình (không import torch), đặt `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, rồi mới import `sim.run`; `sim.run` gọi `torch.set_num_threads` trước mọi phép tính.
    - Đưa `check-inv` vào `check-full` (CONSTRAINTS, T11).
  - Chấp nhận:
    - ENG-4: `python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke` xong dưới 60 s, đủ 4 file đúng schema, `rounds.jsonl` có 6 dòng; manifest ghi `cpu_threads` bằng số luồng của cấu hình.
    - T-INV-5 (SPEC §9.2): cùng seed và `code_version` thì `rounds.jsonl` giống hệt; khác seed thì khác cách chia và khác $w^0$ ở mọi tensor; cùng seed thì cùng partition, hoán vị và $w^0$ giữa các topology và thuật toán; `torch.get_rng_state()` không đổi sau một lượt.
    - `make check-inv` xanh đủ 12 T-INV.
  - Kiểm chứng: `make check-full`; `make check-inv`; `pytest -q tests/e2e/test_smoke_run.py tests/integration/test_seeds.py`
  - Phụ thuộc: T10c, T11a, T09a, T03, T04c
  - File: `src/dfl/sim/run.py`, `src/dfl/cli.py`, `tests/e2e/test_smoke_run.py`, `tests/integration/test_seeds.py`, `Makefile`

- [ ] **T11d · Quy tắc ghi `results/`, phân kỳ, lỗi** — TV3 · S · ∥ T11c
  - Mô tả: trong `sim/run.py`: ghi vào `results/` chỉ khi cây sạch (tính cả file bị ignore trong `src/`, `configs/`) và môi trường khớp `requirements.lock`; khóa `flock` theo run_id (`fcntl` chỉ import khi ghi vào `results/`); từ chối run_id đã xong. Phân kỳ thì ghi `SummaryDiverged`, lỗi thì ghi `SummaryFailed`, với `allow_nan=False`.
  - Chấp nhận:
    - ENG-10: vào `results/` thì từ chối khi cây bẩn, môi trường lệch, hay run_id đã xong; ra `--out` thì chạy được và manifest ghi đúng `dirty`, `env_match`.
    - Lượt có mất mát NaN (cấu hình η rất lớn trên dữ liệu tổng hợp) ghi `SummaryDiverged` với round phân kỳ, và không có dòng NaN nào trong `rounds.jsonl`; một ngoại lệ trong lượt ghi `SummaryFailed`.
    - Hai tiến trình cùng chạy một run_id: tiến trình thứ hai thấy khóa và thoát, không tạo attempt thứ hai.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_results_rules.py`
  - Phụ thuộc: T11b
  - File: `src/dfl/sim/run.py`, `tests/integration/test_results_rules.py`

- [ ] **T14a · Lệnh kiểm sanity + chọn η** — TV1 · S · ∥ T11b, T11c
  - Mô tả: `analysis/sanity.py` có hai phần:
    - kiểm SAN-1…5 (R7; SAN-2 so với B1-4k; SAN-5 là điều kiện cấu trúc), đọc `results/` và trả mã lỗi khác 0 khi không đạt;
    - chọn η theo quy tắc E0a: trung bình 5 mốc cuối trên tập con validation, lấy trường hợp xấu nhất qua các ô, hòa trong 0.2 điểm % thì xét AUC ở ring-α0.1; kèm quy tắc cho τ = 1 và τ = 50, và điều kiện kích hoạt 24 lượt bổ sung;
    - chọn η riêng của B1 (ADR-007): η có accuracy validation lớn nhất trong 3 lượt B1 của E0a, hòa trong 0.2 điểm % thì lấy AUC validation lớn hơn.
  - Chấp nhận:
    - Unit test cho từng SAN (đạt, không đạt, thiếu lượt) và cho quy tắc chọn η: hòa; lượt phân kỳ tính là 0; τ = 1/50 hòa thì giữ η chung; η riêng của B1.
    - Khi chọn η, chỉ đọc lượt có `final_eval_split = validation`.
  - Kiểm chứng: `make check-task`; `pytest -q tests/unit/test_sanity.py`
  - Phụ thuộc: T15a, T15b
  - File: `analysis/sanity.py`, `tests/unit/test_sanity.py`

- [ ] **T11c · Baseline và biến thể trộn + profile fast** — TV2 · M
  - Mô tả: thêm tùy chọn cấu hình cho các baseline và tạo `configs/profiles/fast.yaml`.
    - B1 và B1-4k: một node giữ đủ 55k mẫu; τ·R = 17,200 (10 epoch là 17,188 bước, làm tròn lên bội của 100) và 4,000; 100 chia hết cho τ; lịch đánh giá là các bội của 100 tới hết ngân sách của chính lượt đó.
    - B2: W = I.
    - B3: đường server.
    - Biến thể trộn cho E4: uniform (W uniform, π ∝ $d_j+1$) và size-corrected (MH, `loss_scale` $c_i = n_i/\bar n$ tính một lần trước khi huấn luyện, π = $n_j/n$; chỉ với μ = 0).
  - Chấp nhận:
    - Mỗi baseline và biến thể chạy được trên dữ liệu tổng hợp, ghi đúng số bước, số message (B1 và B2 không có; B3 có 2N/round) và `objective_weights`.
    - Demo C3: `python -m dfl run --config configs/profiles/fast.yaml --out build/fast` chạy được cho ring và FC trên IID.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_baselines.py`
  - Phụ thuộc: T11b, T10b
  - File: `src/dfl/config.py`, `src/dfl/sim/run.py`, `configs/profiles/fast.yaml`, `tests/integration/test_baselines.py`

- [ ] **Checkpoint C3 (M3)** — sau T11c (plan §7):
  - `make check-full` và `make check-inv`, đủ 12 T-INV;
  - demo profile fast cho ring và FC;
  - P3.7 (subagent `test-engineer`);
  - review diff;
  - cài thử và chạy smoke trên laptop demo (QS-8);
  - tăng `MIN_TESTS`;
  - nhóm duyệt.

## Tuần 7 — Sweep, hiệu năng, hình, E0a (mọi code `src/` phải xong trước T14b)

- [ ] **T12 · Sweep** — TV3 · M · ∥ T16a
  - Mô tả: `sim/sweep.py` và lệnh `sweep`.
    - YAML thí nghiệm → lưới → run_id; chỉ mục `results/experiments/<exp>.json` theo từng `code_version`.
    - Bỏ qua run_id đã xong. Tự chạy lại attempt bị ngắt. Chỉ chạy lại attempt `failed` khi có `--retry-failed`.
    - Kiểm cây sạch và môi trường. Khóa theo run_id, và bỏ qua lượt đang bị khóa.
    - Mỗi lượt một tiến trình con mới (không `fork`), có biến môi trường số luồng; tiến trình con kiểm lại `code_version`.
    - Có `--workers` và `--dry-run`.
  - Chấp nhận:
    - ENG-9 đủ các ý ở SPEC §9.4: idempotent và chạy lại được sau khi bị ngắt; attempt `failed` không tự chạy lại; từ chối khi cây bẩn hay môi trường lệch; `code_version` đổi giữa chừng thì ghi `failed` và dừng; commit tài liệu không làm đổi `code_version`; `--dry-run` đếm đúng.
    - Kết quả không phụ thuộc `--workers` (1 và 2 worker cho `rounds.jsonl` giống hệt, trên profile smoke). Hai sweep dùng chung một run_id chỉ chạy lượt đó một lần. `sweep --out build/<dir>` chạy được (cho demo) và vẫn đòi cây sạch.
    - Ghi vào `results/` thì sweep từ chối khi không chạy từ một worktree check out đúng một commit sạch (SPEC §5.3); tiến trình con được tạo bằng `subprocess` hoặc `spawn` (kiểm bằng test đọc start method).
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_sweep.py`
  - Phụ thuộc: T11b, T11d
  - File: `src/dfl/sim/sweep.py`, `src/dfl/cli.py`, `tests/integration/test_sweep.py`

- [ ] **T13 · Đo hiệu năng, chốt giao thức** — TV2 · M · ∥ T16a
  - Mô tả: đo dưới tải thật (H P3.5): thời gian một bước, một lần đánh giá, một lượt fast và full; số luồng {1, 2, 4, 8} kết hợp số worker. Xác nhận lịch đánh giá, tập con và số luồng đã ghi ở `configs/base.yaml` (ADR-007) còn chạy vừa ngân sách, và tạo profile full. Muốn đổi bất kỳ thứ nào trong đó thì phải hỏi nhóm và có ADR mới (Ask first) trước E0a. Cập nhật ngân sách ở `tasks/plan.md` §10.
  - Chấp nhận:
    - `docs/PERF.md` có bảng đo (trước và sau mỗi tối ưu) và số lõi rảnh thực tế.
    - Ngân sách Tier 1 + 2 tính theo số đo mới. Đạt nếu Tier 1 + 2 vừa 3 tuần (tuần 8–10) với số lõi rảnh đo được; không đạt thì đề xuất cắt theo ADR-007 cho nhóm quyết. Không đổi lịch sau E0a.
  - Kiểm chứng: `make check-task`; lệnh đo ghi trong `docs/PERF.md` chạy lại được
  - Phụ thuộc: T11c, T11d, T12
  - File: `docs/PERF.md`, `configs/base.yaml`, `configs/profiles/full.yaml`, `tasks/plan.md`

- [ ] **T16a · Khung vẽ hình + PROVENANCE + P1** — TV3 · M · ∥ T13
  - Mô tả: khung chung của `analysis/plots.py` gồm bảng màu, đơn vị, và một `code_version` mỗi hình.
    - `figures/PROVENANCE.json` ghi lệnh, `code_version`, `analysis_version`, commit và run_id.
    - Ghi vào `figures/` khi `analysis/` bẩn thì từ chối. Không có `--draft` thì chỉ đọc `results/`.
    - `--draft` nới đúng các điều ở SPEC INT-1, và gắn nhãn DRAFT.
    - Có P1 (learning curve) và P7 từ manifest.
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
  - Kiểm chứng: quy trình thí nghiệm với `configs/experiments/e0a_lr.yaml`; lệnh chọn η của `analysis.sanity`
  - Phụ thuộc: T12, T13, T14a; G1 (ADR đã Accepted); `make check-inv` xanh; cây sạch
  - File: `docs/experiments/E0a.md`, `configs/experiments/e0a_lr.yaml`, `docs/experiments/CODE_VERSIONS.md`, `docs/decisions/ADR-007-evaluation-protocol.md`

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

- [ ] **Checkpoint C4** — sau T18a, T19a, T20a, tức đủ 197 lượt của Tier 1 (EXP-2): `make check-full`; hình Tier 1 từ `results/`; SAN-1…5 đã đạt; sao lưu `results/`; review; nhóm duyệt trước khi mở Tier 2

## Tuần 10 — Tier 2 (thứ tự ngược với thứ tự cắt: E3b → E1c → E4 → E2b) (C5, M4)

- [ ] **T19b · E3b: tần suất giao tiếp + prox (42 lượt)** — TV3 · S
  - Mô tả:
    - μ = 0 với τ ∈ {5, 50} × {ring, FC} × {IID, α = 0.1}.
    - μ = 0.1 (theo QS-2) với τ = 20.
    - μ = 1 với τ = 50 × α = 0.1.
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
  - Mô tả: `configs/profiles/demo.yaml` và `configs/experiments/demo.yaml` chạy 2 lượt, ring và FC ở α = 0.1. `docs/DEMO.md` ghi các bước: cài không dùng file khóa, tải MNIST trước, chạy sweep ra `build/demo`, vẽ hình bằng `--draft`. Đo thời gian trên laptop thật (QS-8).
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

## T22 — theo Q1 và QS-10 (không thuộc phần mở rộng tùy chọn)

- [ ] **T22 · `TorchDistTransport` (gloo)** — TV2 · M
  - Mô tả: backend Transport đa tiến trình ở `dfl.sim` (4–8 tiến trình, backend gloo, gửi/nhận tensor đã tuần tự hóa). Lịch: nếu Q1 yêu cầu P2P thật thì phải xong trước T14b (tuần 6–7), và code thuộc lưới. Nếu tới hết tuần 5 chưa có trả lời thì làm ở tuần 8–10, ngay sau T13 (QS-10), chỉ trên nhánh chính: mọi sweep của lưới vẫn chạy từ worktree của commit lưới, nên `code_version` của lưới không đổi (SPEC §5.3, §8).
  - Chấp nhận:
    - Với N = 4 trên dữ liệu tổng hợp, kết quả khớp bản in-memory (allclose), cùng số message và byte do CommLedger đếm.
    - Primitive gửi/nhận không chặn đã được đối chiếu với tài liệu torch 2.14 (ADR-002 ghi `isend`/`irecv` chưa kiểm); nếu không có trong tài liệu thì ghi UNVERIFIED và dùng `send`/`recv` theo thứ tự cố định.
  - Kiểm chứng: `make check-task`; `pytest -q tests/integration/test_torchdist.py`
  - Phụ thuộc: T11b; Q1 hoặc QS-10
  - File: `src/dfl/sim/torchdist.py`, `src/dfl/config.py`, `tests/integration/test_torchdist.py`

## Mở rộng — chỉ làm sau M4, và phải hỏi trước (SPEC §1.4, QĐ-4)

- [ ] T21 E6 Fashion-MNIST và E5 quy mô N ∈ {10, 50} (E5 chỉ đổi cấu hình)
- [ ] T23 Gossip bất đồng bộ, E7
