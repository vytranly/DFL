# SPEC — Đề 2: Non-IID × Decentralized Federated Learning

> **Trạng thái:** Draft 0.3, chờ nhóm duyệt (Gate G1) · **Ngày:** 2026-09-30 · 0.2 (P1.8): chốt QS-3 và QS-5; kiểm tra kiến trúc bằng import-linter; thêm `CONSTRAINTS.md` và `Makefile` · 0.3 (P1.9): sửa theo các vòng phản biện, xem `docs/decisions/REVIEW-P1.9.md`
> **Nguồn:** `docs/KE_HOACH_DE_TAI_2.md` v0.2 (viết tắt **KH**) · `docs/HANDOFF_GUIDELINE.md` (**H**) · `docs/decisions/OPEN_QUESTIONS.md` (**OQ**) · `CLAUDE.md`.
> **Thứ tự ưu tiên khi mâu thuẫn:** nội dung yêu cầu của đề (cột "Yêu cầu" của REQ-01…14 ở H§2.1) là gốc; SPEC không được ghi đè. Sau đó: ADR đã Accepted, `CONSTRAINTS.md` và phần Boundaries của `CLAUDE.md` > SPEC (ADR còn Proposed có cùng thứ hạng với SPEC, và hai bên phải khớp nhau) > bẫy G1–G25 (H§2.4) > KH > phần còn lại của H (kể cả cột "Thiết kế" và "Kiểm chứng" của H§2.1). SPEC chỉ được khác bẫy, KH hay H khi chỗ khác đó có trong §10; gặp mâu thuẫn chưa có trong §10 thì dừng và hỏi (H§3.0), không tự chọn một cách hiểu.
> Chỗ ghi **(QS-k)** đang dùng giá trị mặc định, chờ nhóm quyết ở §11.2.

## ASSUMPTIONS

```text
ASSUMPTIONS (giả định khi viết SPEC — sửa ngay nếu sai; nếu không, các task sẽ làm theo):
1. "Hệ thống P2P" = mô phỏng đơn tiến trình, đồng bộ; node chỉ trao đổi model qua Transport có đếm byte,
   không có server (A1, ADR-002). Chờ GVHD trả lời Q1; hết tuần 3 chưa có trả lời thì hỏi lại. Nếu GVHD
   yêu cầu P2P thật, T22 (4–8 tiến trình, torch.distributed/gloo) thành bắt buộc và phải xong trước E0a,
   để code_version không đổi giữa lưới; TorchDistTransport đặt ở `sim` (phải import torch). Hết tuần 5
   (trước E0a) vẫn chưa có trả lời thì vẫn chạy lưới bằng mô phỏng đơn tiến trình, để không trễ lịch.
   Im lặng không phải là đồng ý: cách hiểu "P2P" vẫn là rủi ro, được ghi vào mục hạn chế, và nhóm hỏi
   lại ở buổi gặp kế tiếp. Khi đó T22 làm ngay sau T13, trước M4, không còn là tùy chọn (QS-10, nhóm chốt).
2. Nhóm 3 người, 12 tuần (QĐ-1). Mọi thí nghiệm chạy trên máy chủ CPU hiện tại (8 nhân, không GPU,
   dùng chung với người khác: lúc đo tải trung bình khoảng 7); laptop cá nhân chỉ dùng để demo (QĐ-3).
   Không viết nhánh code cho GPU (QĐ-5).
3. Mọi node cùng kiến trúc model và cùng w⁰ theo seed chung (A3, ghi là hạn chế trong báo cáo);
   mọi node tham gia mọi round, đồng bộ, không mô phỏng rớt node (A4).
4. Đánh giá trên tập test toàn cục chung, kèm acc^loc theo phân bố nhãn của từng node (A5).
   Macro-F1 tính cho từng node trên test chung; báo cáo mean, std và node kém nhất (Q3).
5. Phương pháp (cách chia dữ liệu, topology, ma trận trộn, thuật toán, độ đo, lưới, thống kê) lấy nguyên
   KH §3 v0.2. SPEC dẫn chiếu, không chép lại công thức; các điểm SPEC chuẩn hóa lại nằm ở §10.
6. Phạm vi đúng bảng KH §2.4 (xem §1.4). Phần mở rộng chỉ làm sau mốc M4, theo thứ tự QĐ-4.
7. η chưa chốt: chọn ở E0a (seed 100, chỉ trên validation) theo quy tắc ở ADR-007, dựa trên KH §3.5 (dự kiến 0.05),
   và ghi vào ADR-007 trước khi chạy E0b.
8. Báo cáo viết tiếng Việt; mọi hình sinh tự động từ log (A6). Hạn nộp và định dạng báo cáo chờ GVHD (Q2).
→ Sửa ngay nếu có điểm sai. Các điểm SPEC cần nhóm quyết: §11.2 (QS-1…QS-11).
```

---

## 1. Objective

### 1.1 Xây gì và vì sao

Một bộ mô phỏng **Decentralized Federated Learning (DFL) không server**, tất định, chạy trên CPU, cùng pipeline phân tích, để trả lời Đề 2 (ĐATH HK261; yêu cầu REQ-01…14 ở H§2.1): dữ liệu **Non-IID** ảnh hưởng thế nào tới accuracy/macro-F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định và phương sai giữa các node; **topology** (ring, Erdős–Rényi, random gossip, fully-connected) và **thuật toán** (D-PSGD, DFedAvg, DFedProx) nào chống chịu tốt hơn. Dataset chính: MNIST (PyTorch + torchvision). Sản phẩm cuối: kết quả thực nghiệm tái lập được, hình/bảng sinh tự động, báo cáo tiếng Việt có khuyến nghị.

### 1.2 Người dùng và kịch bản

| Người dùng | Cần gì |
|---|---|
| Nhóm 3 thành viên (TV1–TV3, xem §3.1) | Chạy thí nghiệm, phân tích, viết báo cáo, demo |
| GVHD | Xem kiến trúc, kết quả và demo; đánh giá |
| Agent lập trình (pha 2–3) | Triển khai từng task theo SPEC, test trước |

- **US-1 Một lượt:** `python -m dfl run --config <yaml>` tạo một thư mục attempt có `manifest.json`, `rounds.jsonl`, `timing.jsonl`, `summary.json`.
- **US-2 Một nhóm thí nghiệm:** `python -m dfl sweep` chạy song song; bị ngắt thì chạy lại đúng lệnh đó: `run_id` đã có attempt xong (`ok` hoặc `diverged`) được bỏ qua, attempt bị ngắt được chạy lại, còn attempt `failed` chỉ chạy lại khi có `--retry-failed` (§5.3).
- **US-3 Hình và bảng:** một lệnh sinh toàn bộ P1–P9 và bảng T1 từ `results/`.
- **US-4 Xem mức Non-IID và spectral gap trước khi có engine** (mốc M2): `python -m dfl describe`.
- **US-5 Demo trên laptop** trong vài phút: ring vs fully-connected ở α = 0.1, kèm bộ hình đã sinh từ kết quả thật (QĐ-3).
- **US-6 Kiểm toán:** từ một hình hay một nhận định trong báo cáo lần ra `run_id`, rồi manifest (cấu hình, seed, `code_version`, máy), rồi chạy lại được.

### 1.3 Mục tiêu, câu hỏi nghiên cứu, giả thuyết

- **O1** môi trường DFL tái lập được · **O2** định lượng ảnh hưởng của Non-IID · **O3** so sánh topology và thuật toán dưới cùng điều kiện Non-IID · **O4** khuyến nghị, mỗi khuyến nghị dẫn chiếu hình và run ID (KH §2.1).
- **RQ1** Non-IID tăng thì accuracy, tốc độ hội tụ, phương sai giữa node đổi thế nào? · **RQ2** Topology nào chống chịu tốt nhất; mức suy giảm có giải thích được bằng spectral gap hay consensus distance không? · **RQ3** Ở cùng tổng số bước cục bộ, τ và μ đánh đổi accuracy với chi phí giao tiếp ra sao? · **RQ4** Quantity skew có tác động như label skew không; trọng số trộn có quan trọng không? (KH §2.2)
- **H1–H7** (KH §2.3) là giả thuyết đăng ký trước để kiểm chứng, không phải điều mặc định đúng. Mỗi nhóm thí nghiệm có thẻ tiền đăng ký `docs/experiments/E<k>.md` (mẫu H§5.3) trước khi chạy; kết quả trái giả thuyết vẫn được báo cáo.

### 1.4 Phạm vi (KH §2.4 — không thêm tính năng ngoài bảng này)

| Trong phạm vi (bắt buộc) | Mở rộng (chỉ sau M4, thứ tự QĐ-4) | Ngoài phạm vi (QĐ-5) |
|---|---|---|
| Mô phỏng DFL đồng bộ, đơn tiến trình, message-passing có đếm byte | CIFAR-10 nếu T13 cho thấy chạy nổi trên CPU; Fashion-MNIST (E6); quy mô N ∈ {10, 50} (E5) | Differential privacy, mã hóa, tấn công/phòng thủ Byzantine |
| MNIST; IID, Dirichlet, #C = k, quantity skew | `TorchDistTransport` (gloo, 4–8 tiến trình); thành bắt buộc nếu Q1 yêu cầu, hoặc nếu tới hết tuần 5 GVHD chưa trả lời Q1 (QS-10; khi đó làm ngay sau T13, không đợi M4) | Nén / lượng tử hóa model |
| Ring, Erdős–Rényi liên thông, random pairwise gossip, fully-connected; star chỉ làm tham chiếu | Random k-regular, đồ thị mũ (T5); vị trí dữ liệu theo độ trung tâm (E7); gossip bất đồng bộ | Mạng thật nhiều máy, đo độ trễ mạng; GPU / Colab |
| D-PSGD, DFedAvg, DFedProx; quét τ; baseline tập trung, chỉ cục bộ, FedAvg có server | | Giao diện web / dashboard |
| Accuracy (toàn cục và theo phân bố cục bộ), macro-F1, bước/byte để đạt ngưỡng, độ ổn định, drift, consensus distance, phương sai giữa node | | Model khác nhau giữa các node |

Bốn loại phân phối đề nêu (REQ-02) ứng với các cách chia như sau: Dirichlet distribution → P1; pathological partition → P2 (#C = k của NIID-Bench: mỗi node đúng k lớp, kích thước node không đều); quantity-based skew → P3; label skew → P1 và P2. Trong tài liệu, "pathological partition" còn có dạng shards của McMahan 2017 (P2'): sắp theo nhãn rồi chia mảnh, kích thước node đều, số lớp mỗi node thường là 2 nhưng có thể là 1 hoặc nhiều hơn. Hai dạng không tương đương, nên báo cáo phải nói rõ dự án dùng dạng #C = k. NIID-Bench còn gọi #C = k là "quantity-based label imbalance", nên cách đọc "quantity-based skew" của đề cũng chưa chắc. Cách ánh xạ này cần GVHD xác nhận (Q9, §11.1). Để không phải đổi `code_version` nếu GVHD muốn dạng shards, P2' được cài sẵn ở T04b và chỉ dùng trong `describe`, không chạy trong lưới trừ khi GVHD hay nhóm yêu cầu (QS-11). P4 ($p_{major}$) là tùy chọn, chỉ làm khi nhóm yêu cầu.

Code cho phần mở rộng nếu viết sau E0a sẽ làm đổi `code_version` (§5.3). Khi đó kết quả mở rộng chỉ so được với lưới lõi ở mức định tính, hoặc phải tự chạy lại các ô tham chiếu dưới `code_version` mới. Ví dụ là tham số dataset để nạp Fashion-MNIST cho E6, hay topology T5. Muốn so định lượng E6 hay T5 với lưới lõi thì phần code đó phải có trước E0a (Ask first). E5 chỉ đổi N trong cấu hình, nên không cần code mới.

---

## 2. Tech Stack

| Thành phần | Phiên bản (ghim trong `pyproject.toml`) | Ghi chú |
|---|---|---|
| Python | 3.12 (`>=3.12,<3.13`) | |
| torch / torchvision | 2.14.0 / 0.29.0, bản CPU | Cài từ `https://download.pytorch.org/whl/cpu` trước các gói khác (§4) |
| networkx | 3.7 | Topology, spectral gap |
| numpy | 2.5.3 | `Generator` (PCG64), `SeedSequence` |
| pandas · matplotlib · pyyaml | 3.0.6 · 3.11.2 · 6.0.3 | Phân tích · hình · cấu hình (chỉ `yaml.safe_load`) |
| pytest · pytest-cov · ruff | 9.1.1 · 7.1.0 · 0.16.9 | Dev. ruff: dòng ≤ 110, py312, lint E, F, W, I, B, UP cộng các luật sàn của `CONSTRAINTS.md`; loại `docs/` |
| import-linter | 2.15 | Dev, kiểm ranh giới import theo §3.2 (`lint-imports`) |
| scikit-learn | 1.9.1 | Dev, chỉ làm oracle cho macro-F1 trong test |
| setuptools | ≥ 80 | Build backend, layout `src/` |

- **Máy chạy thí nghiệm:** máy chủ Linux, Intel Xeon Platinum 8176, 8 nhân, không GPU. **Laptop demo:** chưa biết hệ điều hành (QS-8).
- **`requirements.lock`** (QS-1, tạo ở T02) ghim mọi gói *chạy* (phụ thuộc trong `[project] dependencies` và phụ thuộc bắc cầu của chúng), không gồm công cụ dev. Nó thuộc `code_version` (§5.3), nên nâng ruff hay pytest không làm kết quả thành stale.
- Không thêm hay nâng phụ thuộc khi chưa hỏi (§8). API framework dùng trong code phải đối chiếu tài liệu chính thức đúng phiên bản trên (P1.7).

---

## 3. Capability map

### 3.1 Module (id ổn định, không đổi tên trong suốt dự án)

| Module id | Mã nguồn | Trách nhiệm | Không được | Phụ trách (QĐ-1) | Task (KH §8) |
|---|---|---|---|---|---|
| `data` | `src/dfl/data/{datasets,partition,heterogeneity}.py` | Nạp MNIST bằng torchvision thành tensor trong RAM, chuẩn hóa, tách validation 5k phân tầng; P0–P3 đúng thuật toán NIID-Bench (min_size = 64); hoán vị vị trí node theo seed; JS, TV, số lớp ≥ 5%, CV(n), lệch mục tiêu, ma trận node × lớp | `partition`/`heterogeneity` import torch; biết topology hay model | TV1 | T03, T04a, T04b, T05a |
| `network` | `src/dfl/network/{topology,mixing,transport}.py` | T0–T4, T3b (T5 tùy chọn); `graph_at(round)` là hàm thuần của (seed, round); ER lấy mẫu lại tới khi liên thông; MH / uniform / size-weighted; kiểm tra ngẫu nhiên hàng/kép; spectral gap; trọng số mục tiêu π (`objective_weights`, ADR-005); Transport in-memory + CommLedger, nơi duy nhất đếm message và byte (tổng và theo từng node, kể cả server ảo có chỉ số N của B3/E1c), đọc số byte từ `Snapshot.nbytes` | Import torch; tổng hợp model; copy payload theo số láng giềng; message node tự gửi cho mình | TV2 | T06, T07, T08 |
| `learning` | `src/dfl/learning/{models,local_train}.py` | SmallCNN 46,730 tham số (không BatchNorm, không dropout); τ bước SGD (momentum 0, weight decay 0), minibatch B = 32 lấy có hoàn lại, số hạng prox μ với anchor = $w_i^t$ (clone + detach), hệ số mất mát $c_i$; ngẫu nhiên qua `torch.Generator` riêng từng node | Truy cập node khác, Transport hay topology | TV1 (QS-9) | T09a |
| `metrics-io` | `src/dfl/metrics/{classification,consensus}.py`, `src/dfl/io/{run_logger,manifest}.py` | Độ đo trong lúc chạy: $\mathrm{Acc}_i$, $\mathrm{Acc}^{loc}_i$, $\mathrm{Acc}^{eff}$, macro-F1 (từ ma trận nhầm lẫn), drift trước tổng hợp, consensus distance sau tổng hợp; ghi manifest/rounds/timing/summary theo attempt; tính `code_version`, `run_id` | Tự đếm byte (INV-2); tiêu thụ RNG của huấn luyện (INV-5); ghi đè attempt cũ; ghi thời gian vào `rounds.jsonl` (INV-6) | TV1: độ đo (T09b); TV3: io (T11a) (QS-9) | T09b, T11a |
| `sim` | `src/dfl/sim/{node,engine,observer,sweep}.py`, `src/dfl/cli.py`, `src/dfl/__main__.py` | Trạng thái node; vòng ATC; Observer theo lịch đánh giá (trước và sau tổng hợp); baseline B1–B3, trong đó B3 và E1c đi qua đường server (ADR-005); lắp ráp một lượt từ cấu hình; sweep idempotent, mỗi lượt một tiến trình con mới (§5.2); CLI `run`, `sweep`, `describe`, `download`; `TorchDistTransport` nếu làm T22 (phải import torch nên đặt ở đây, không ở `network`) | Đưa đại lượng toàn cục vào huấn luyện (INV-1); đọc model của node khác ngoài inbox | TV2: engine, `run`, baseline, hiệu năng, T22 (T10a–c, T11b, T11c, T13); TV1: `describe` dữ liệu (T05a); TV3: `describe` topology, sweep (T08b, T12) | T05a, T08b, T10a–T13 |
| `analysis` | `analysis/{aggregate,convergence,stats,plots}.py` (ngoài `src/`, nên sửa không làm đổi `code_version`) | Tự đọc JSON/JSONL của `results/` bằng code riêng; gom các lượt thành DataFrame/CSV; độ đo suy ra từ `rounds.jsonl` (R@x, AUC, độ dao động và mức sụt theo từng node); CI (phân phối t), kiểm định H2; P1–P9, bảng T1; từ chối trộn `code_version`, lượt `dirty` hay lệch môi trường, so sánh lệch biến cố định hoặc lệch tập seed (INT-2); chế độ `--draft` cho kết quả smoke và demo | Import bất cứ gì của `dfl` ngoài `dfl.contracts`; sửa hay lọc số liệu | TV3; TV1 phân tích E1, E1c, E4a, E4 | T05b, T14a, T15a–c, T16a–b |

**Nền tảng dùng chung** (không phải module id, QS-6): `src/dfl/contracts.py` (Protocol / TypedDict / dataclass, không có logic — P1.7), `config.py` (dataclass + `yaml.safe_load` + kiểm tra kiểu và giá trị cấu hình), `seeding.py` (nơi duy nhất tạo `SeedSequence`/`Generator`; mỗi luồng ngẫu nhiên có spawn_key cố định, §6). Nền tảng không import module nào ở bảng trên. TV3 phụ trách nền tảng (T02a–c, QS-9). Mã task theo `tasks/todo.md`.

### 3.2 Hướng phụ thuộc (import một chiều, không có vòng)

```text
analysis ──(chỉ dfl.contracts; tự đọc file kết quả)──────────┐
                                                             ▼
sim ──► data · network · learning · metrics-io ──► nền tảng (contracts · config · seeding)
```

- Bốn module lá `data`, `network`, `learning`, `metrics-io` chỉ import nền tảng, không import lẫn nhau. `sim` là nơi duy nhất ghép chúng; dữ liệu đi giữa các module lá dưới dạng mảng/tensor do `sim` truyền vào.
- `src/dfl` không bao giờ import `analysis`. `analysis` chỉ đọc file trong `results/` (hoặc `build/`) và chỉ được import `dfl.contracts`. Code đọc JSON/JSONL nằm trong `analysis/`, nên sửa nó không làm đổi `code_version`.
- Các module thuần `dfl.data.partition`, `dfl.data.heterogeneity`, `dfl.network.topology`, `dfl.network.mixing`, `dfl.network.transport` không nạp torch, kể cả gián tiếp: `__init__.py` để trống; `contracts.py` chỉ import torch dưới `TYPE_CHECKING`, hoặc module thuần không import `contracts` (Protocol so khớp theo cấu trúc, không cần import).
- Kiểm bằng test kiến trúc ENG-6 và ENG-7 (§9.4).

### 3.3 Thứ tự build

| Bước | Module | Task (KH §8) | Mốc / checkpoint |
|---|---|---|---|
| 0 | Nền tảng | P1.7 (`contracts.py`), T02 (config, seeding) | Gate G1 (hết pha 1B) |
| 1 | `data` ∥ `network` (TV1 ∥ TV2) | T03–T05 ∥ T06–T08 | C1 sau T05, C2 sau T08; **M2**: P7 và bảng spectral gap sinh từ code |
| 2 | `learning` + phần độ đo của `metrics-io` | T09 | |
| 3 | `sim`: engine, Observer (viết T-INV-1…4, 11, 12 trước engine) | T10 | |
| 4 | Phần io của `metrics-io` + `sim`: CLI `run`, baseline B1–B3 | T11 | C3; **M3**: lát dọc đầu tiên, IID chạy trên ring và FC |
| 5 | `sim`: sweep, rồi đo hiệu năng | T12 → T13 | |
| 6 | `analysis` (TV3 có thể bắt đầu ngay sau P1.7, dùng `rounds.jsonl` giả) | T15 → T16 | |
| 7 | Thí nghiệm: E0a → ghi η vào ADR-007 → E0b → Tier 1 → Tier 2 | T14, T17–T20 | C4 sau T18, C5 sau T20; **M4** |

Mốc theo KH §7: M2 cuối tuần 4, M3 cuối tuần 6, M4 cuối tuần 10. T13 (tuần 7) đo tốc độ dưới tải thật rồi chốt lịch đánh giá, tập con đánh giá và số luồng. Việc này phải xong trước E0a, vì các tham số đó nằm trong `run_id`; sau E0a chúng không đổi nữa.

---

## 4. Commands

Chạy từ gốc repo, trong `.venv`. Chú thích ở mỗi nhóm lệnh cho biết lệnh có từ task nào.

```bash
# Cài đặt (CPU). Trước T02 chưa có requirements.lock: bỏ dòng `-r` và phần `-c requirements.lock`
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.lock                   # đúng phiên bản mọi gói chạy (QS-1)
pip install -e ".[dev]" -c requirements.lock       # công cụ dev; -c giữ nguyên phiên bản gói chạy

# Tạo lại requirements.lock — chỉ khi đổi phụ thuộc (Ask first), trên máy chủ, trong venv chỉ có gói chạy
python3.12 -m venv /tmp/lockenv && . /tmp/lockenv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e . && pip freeze --exclude-editable > requirements.lock

# Lint và test trực tiếp — dùng được ngay (trừ dòng cuối)
ruff check . && ruff format --check .
pytest -q tests/unit
pytest -q tests/integration tests/e2e      # từ T10–T11; chạy riêng lúc chưa có test thì pytest trả mã 5
                                           # (check-task vẫn xanh vì gom cả tests/unit)

# Kiểm tra theo chi phí — Makefile (P1.8); nội dung và lý do ở CONSTRAINTS.md
make check-fast   # sàn + ruff + import-linter + unit, vài giây
make check-task   # như trên + integration + coverage ≥ 80%; dưới 90 s (chỉ cảnh báo); trước mỗi commit
make check-full   # như trên + mọi test (gồm smoke e2e từ T11) + pip check; ở mỗi checkpoint
make check-inv    # T-INV-1…12 có đủ và xanh; bắt buộc từ C3 và trước mọi lượt Tier 1

# Dữ liệu — T03: tải MNIST vào data/ một lần (cần mạng; laptop demo phải tải trước)
python -m dfl download mnist

# Một lượt — T11
python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke   # dữ liệu tổng hợp, dưới 60 s
python -m dfl run --config <config.yaml>          # ghi results/runs/<run_id>/attempt-<k>/; cây phải sạch;
                                                  # profile smoke không bao giờ được ghi vào results/

# Mô tả dữ liệu và topology, không huấn luyện — T05 (mốc M2)
python -m dfl describe --config configs/base.yaml --seeds 20 --out build/describe

# Lưới thí nghiệm — T12
python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --dry-run    # in tổng / đã xong / failed / sẽ chạy
python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --workers 4
python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --retry-failed   # chạy lại attempt failed (§5.3)

# Phân tích và hình — T15–T16 (analysis/ phải sạch khi ghi vào figures/, INT-1)
python -m analysis.aggregate --results results/ --out build/analysis   # CSV một dòng mỗi lượt, kèm độ đo suy ra
python -m analysis.plots --results results/ --out figures/              # P1–P9 và bảng T1

# Demo trên laptop — QĐ-3, QS-8 (sweep luôn đòi cây sạch, kể cả khi ghi ra --out)
# Laptop cài không dùng requirements.lock (bản torch của nền tảng, ví dụ macOS không có "+cpu"), nên manifest
# ghi env_match: false; ghi ra build/demo thì không cần khóa, nên không nạp fcntl (không có trên Windows)
pip install torch==2.14.0 torchvision==0.29.0 && pip install -e .
python -m dfl sweep --experiment configs/experiments/demo.yaml --out build/demo --workers 2
python -m analysis.plots --results build/demo --out build/demo/figures --draft   # hình có nhãn "DRAFT"
```

Các lệnh `run`, `sweep`, `analysis.plots` đã có trong CLAUDE.md. `download`, `describe`, `sweep --dry-run`, `--retry-failed`, `--out`, `--draft` và `analysis.aggregate` là đề xuất (QS-7); cờ chi tiết chốt ở T11–T12. `--out` ra ngoài `results/` chỉ nới quy tắc cây sạch cho `run`, không cho `sweep`; nó còn cho phép chạy khi môi trường lệch `requirements.lock` (§5.3).

Các cơ chế cài đặt trên theo tài liệu pip: constraints file (`-c`) chỉ quyết định phiên bản của gói, không làm gói được cài (https://pip.pypa.io/en/stable/user_guide/#constraints-files). Trên Python 3.12, `pip freeze` chỉ bỏ qua `pip` (https://pip.pypa.io/en/stable/cli/pip_freeze/). venv của Python 3.12 không cài sẵn setuptools (https://docs.python.org/3.12/whatsnew/3.12.html, gh-95299), nên file khóa chỉ gồm gói chạy.

---

## 5. Project Structure

### 5.1 Cây thư mục

```text
non_iid-dfl/
├── CLAUDE.md  SPEC.md  README.md  pyproject.toml     # đã có
├── CONSTRAINTS.md  Makefile                          # có từ P1.8
├── requirements.lock                                 # T02, QS-1: chỉ gói chạy; thuộc code_version
├── .gitattributes                                    # T02: `* text=auto eol=lf`, để checkout trên Windows không đổi byte
├── configs/
│   ├── base.yaml                                     # mặc định KH §3.5
│   ├── profiles/{smoke,fast,full,demo}.yaml
│   └── experiments/{e0a_lr,e0b_sanity,e1_noniid,e1c_objective,e2_topology,e2b_gossip,
│                    e3a_algorithms,e3b_tau_mu,e4a_quantity,e4_quantity,demo}.yaml
├── src/dfl/                                          # mọi code ảnh hưởng tới kết quả; hash nội dung → code_version
│   ├── __main__.py  cli.py                           # sim (cli.py loại khỏi coverage)
│   ├── contracts.py  config.py  seeding.py           # nền tảng dùng chung
│   ├── data/{datasets,partition,heterogeneity}.py    # data
│   ├── network/{topology,mixing,transport}.py        # network
│   ├── learning/{models,local_train}.py              # learning
│   ├── metrics/{classification,consensus}.py         # metrics-io
│   ├── io/{run_logger,manifest}.py                   # metrics-io
│   └── sim/{node,engine,observer,sweep,run,describe}.py   # sim (run: lắp một lượt; describe: lệnh describe)
├── analysis/{aggregate,checks,convergence,stats,sanity,plots,provenance}.py   # analysis (ngoài src/: sửa không làm đổi code_version)
├── tests/
│   ├── conftest.py                                   # fixture autouse kiểm RNG toàn cục (§7)
│   ├── unit/  integration/  e2e/                     # file test_<chủ đề>.py, tên không trùng giữa các thư mục
│   └── reference/                                    # hàm chia dữ liệu tham chiếu (sao từ docs/spikes/) cho test golden
├── data/        # .gitignore với mẫu có neo `/data/` (không trùng src/dfl/data/) — MNIST tải bằng torchvision
├── results/     # .gitignore — runs/<run_id>/attempt-<k>/ và experiments/<exp>.json; sao lưu sau mỗi nhóm thí nghiệm
├── build/       # .gitignore — smoke, demo, describe, coverage
├── figures/     # chỉ sinh bằng script; có commit
├── docs/        # đề tài, research/, KH, H, decisions/ (ADR, OPEN_QUESTIONS), experiments/{E<k>.md, CODE_VERSIONS.md},
│                #   spikes/, report/, PERF.md (T13), DEMO.md (T26), VALIDATION.md (T25)
└── tasks/{plan.md,todo.md,SESSION_LOG.md}
```

### 5.2 Cấu hình và profile

- `configs/base.yaml` chứa mặc định KH §3.5: N = 20, τ·R = 4,000, B = 32, SmallCNN, Metropolis–Hastings, min_size = 64, lịch đánh giá theo bước cục bộ, tập con test phân tầng 2,000 ảnh. Profile ghi đè ngân sách và quy mô; mỗi file trong `configs/experiments/` khai báo lưới của một nhóm thí nghiệm và profile dùng.
- **Tập đánh giá** là tham số cấu hình: split, kích thước tập con cho learning curve, và seed của tập con. E0a dùng validation cho cả learning curve và đánh giá cuối; mọi nhóm khác dùng test (ADR-007). Seed của tập con là hằng số của cả dự án, không phải seed của lượt (§6), nên mọi lượt so sánh với nhau dùng đúng cùng các ảnh.
- **Lịch đánh giá** (ADR-007) là tham số của profile. Profile `full` và `fast` đánh giá ở các mốc 100, 200, …, τ·R bước cục bộ, giống nhau cho mọi τ ∈ {1, 5, 20, 50}: 100 chia hết cho mọi τ đó, nên mọi mốc rơi đúng vào cuối một round. Profile smoke đánh giá mỗi round. Round cuối luôn được đánh giá, nên `rounds.jsonl` không bao giờ rỗng.
- **Số luồng mỗi lượt** là hằng số của cả lưới (mặc định 1, chốt ở T13), nằm trong cấu hình chuẩn hóa; `--workers` chỉ quyết định số lượt chạy song song. Đo ngày 2026-09-30: một bước huấn luyện SmallCNN (B = 32) mất 10.5 ms với 1 luồng và 10.0 ms với 8 luồng, nên nhiều luồng không có lợi (ADR-007).
  - Sweep chạy mỗi lượt trong một tiến trình con mới: `subprocess` gọi `python -m dfl run`, hoặc `multiprocessing` với start method `spawn`. Không dùng `fork`, vì tài liệu Python 3.12 ghi fork một tiến trình nhiều luồng là không an toàn (https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods).
  - `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` được đặt trong môi trường của tiến trình con trước khi nó khởi động. Khi chạy `dfl run` trực tiếp, CLI đọc cấu hình (không import torch), đặt các biến này, rồi mới import module dùng torch.
  - `dfl run` gọi `torch.set_num_threads(k)` trước mọi phép tính torch, vì tài liệu 2.14 yêu cầu gọi trước code eager hay autograd (https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html). Manifest ghi giá trị đọc lại bằng `torch.get_num_threads()`.
- **B1 (tập trung):** một model trên 55k mẫu, 10 epoch, B = 32 (khoảng 17,190 bước), minibatch lấy có hoàn lại; η riêng chọn trên validation trong {0.01, 0.05, 0.1} (3 lượt, thuộc E0a). **B1-4k:** như B1 nhưng dùng η chung của lưới và 4,000 bước, bằng số bước cục bộ của một node DFL (3 lượt, thuộc E0b). B1-4k là đối chứng của SAN-2 (§9.3; nhóm chốt ở P1.9, QS-3): nó tách được lỗi khỏi hiệu ứng ngân sách và η.
- Mỗi nhóm E0a, E0b, E1, E1c, E2, E2b, E3a, E3b, E4a, E4 có một file YAML riêng (E0a phải xong trước E0b vì η; E3a và E4a thuộc Tier 1, E3b và E4 thuộc Tier 2), cộng `demo.yaml`.

| Profile | N | τ·R | Dữ liệu | Dùng cho |
|---|---|---|---|---|
| `smoke` | 4 | 12 (τ = 2, R = 6) | Tổng hợp, cùng shape MNIST: 2,000 train (200 trong đó làm validation), 500 test dùng toàn bộ làm tập đánh giá; đánh giá mỗi round (6 dòng); chỉ IID/Dirichlet (#C = k cần N ≥ 10, G23) | Test và `make check-full`; dưới 60 s |
| `fast` | 20 | 1,000 (τ = 10) | MNIST; đánh giá mỗi 100 bước | Phát triển |
| `full` | 20 | 4,000 (mặc định τ = 20, R = 200) | MNIST | Kết quả cho báo cáo |
| `demo` | 20 | Nhỏ, dự kiến khoảng 400 | MNIST đã tải sẵn | Demo trên laptop: 2 lượt ring vs FC ở α = 0.1, tổng ≤ 5 phút (QS-8) |

### 5.3 Kết quả

- **Đường dẫn:** `results/runs/<run_id>/attempt-<k>/{manifest.json, rounds.jsonl, timing.jsonl, summary.json}`. Mỗi nhóm thí nghiệm có chỉ mục `results/experiments/<exp>.json` liệt kê các run_id của nó theo từng `code_version` (do sweep ghi). Một `code_version` mới thêm một mục mới; mục cũ không bị xóa hay ghi đè. Nhiều nhóm dùng chung được một run_id, nên ô dùng lại (ví dụ ô của E4 lấy từ E1) không phải chạy lại. `analysis` chọn lượt qua chỉ mục, không quét cả thư mục.
- **Cấu hình chuẩn hóa:** dataclass cấu hình đầy đủ, đã điền mặc định và qua kiểm tra kiểu (PyYAML đọc `5e-2` thành chuỗi, nên mọi trường số phải được kiểm). Chỉ gồm tham số ảnh hưởng tới kết quả, không gồm tên nhóm thí nghiệm, `--out`, `--workers`. Ghi thành JSON, khóa sắp xếp, số thực theo `repr`.
- **`code_version`** = sha256 của dãy cặp (đường dẫn tương đối, nội dung) của mọi file `*.py` trong `src/`, sắp theo đường dẫn, cùng `requirements.lock` (QS-1). Vì đường dẫn nằm trong hash, chuyển code giữa các file cũng làm đổi nó. Trước khi hash, `\r\n` được đổi thành `\n`; Python đọc hai kiểu xuống dòng như nhau, nên đổi kiểu xuống dòng không đổi hành vi. Các dòng của file khóa được bỏ khoảng trắng thừa và sắp xếp trước khi hash. Tính từ file trên đĩa, không cần git, nên phản ánh cả thay đổi chưa commit. Commit chỉ đổi `docs/`, `tasks/`, `analysis/` hay `configs/` không làm đổi nó (cấu hình đi vào `run_id`). **`run_id`** = hash của (cấu hình chuẩn hóa + `code_version` + máy), trong đó "máy" là tên CPU kèm hash của danh sách cờ CPU (G13). Nhờ vậy lượt chạy trên máy khác, hay trên máy ảo đã bị chuyển sang phần cứng khác, là một run_id khác: sweep không bỏ qua nó, và `analysis` không trộn nó (INT-2).
- **Cây sạch** nghĩa là `git status --porcelain -- src configs requirements.lock` rỗng, tính cả file chưa track, và không có file `*.py` hay `*.yaml` nào trong `src/`, `configs/` bị `.gitignore` bỏ qua (`git status --porcelain --ignored`). File bị ignore không hiện trong `git status` nhưng vẫn vào `code_version`. `sweep` luôn đòi cây sạch (CLAUDE.md), kể cả khi ghi ra `--out`. `run` chỉ được ghi vào `results/` khi cây sạch; `run --out` ra ngoài `results/` (ví dụ `build/`) thì được phép khi cây bẩn hay không có git, và manifest ghi `dirty: true`. Profile smoke không bao giờ ghi vào `results/`. `.gitattributes` đặt `eol=lf` để checkout trên Windows không làm cây bẩn (https://git-scm.com/docs/gitattributes: `eol` ghi đè `core.autocrlf`).
- **Môi trường:** trước khi chạy, cả `run` và `sweep` so từng gói trong `requirements.lock` với phiên bản đã cài, đọc bằng `importlib.metadata.version` (https://docs.python.org/3.12/library/importlib.metadata.html). Gói có trong môi trường nhưng không có trong file khóa (công cụ dev) được bỏ qua. Ghi vào `results/` thì mọi gói phải khớp, nếu không thì từ chối. Ghi ra `--out` ngoài `results/` thì chỉ cảnh báo, và manifest ghi `env_match: false`, như với laptop demo.
- **Attempt:**
  - Mỗi run_id có nhiều nhất một attempt đã xong (`ok` hoặc `diverged`). Sweep bỏ qua run_id đó; `run` ghi vào `results/` thì từ chối.
  - Attempt không có `summary.json` (tiến trình bị ngắt) được giữ lại và tự chạy lại thành attempt mới.
  - Attempt `failed` được giữ lại nhưng không tự chạy lại, vì lỗi với cùng `code_version` và cấu hình thường lặp lại y hệt. Ví dụ: vượt 1,000 lần lấy mẫu lại khi chia dữ liệu, cấu hình sai, hay lỗi code. Sửa code thì `code_version` đổi, nên đó là run_id mới. Lỗi do môi trường (đầy đĩa, thiếu RAM) thì chạy lại bằng `sweep --retry-failed`. `--dry-run` in số attempt `failed`.
  - Thư mục attempt được tạo nguyên tử.
  - **Khóa theo từng run_id:** tiến trình chạy một lượt giữ `fcntl.flock(LOCK_EX | LOCK_NB)` trên `results/runs/<run_id>/.lock` từ lúc chọn số attempt tới lúc ghi `summary.json`. Sweep giữ khóa trên `results/experiments/<exp>.lock` khi ghi chỉ mục. Nhờ vậy nhiều sweep của các nhóm khác nhau chạy song song được, và `--workers` chạy nhiều lượt cùng lúc. Một lượt dùng chung giữa hai sweep chỉ chạy ở một nơi: nơi kia thấy khóa đang bị giữ thì bỏ qua, và dùng kết quả khi lượt đó xong. Khóa tự nhả khi mọi file descriptor của nó đóng, kể cả khi tiến trình chết, nên không có khóa bỏ quên phải xóa tay (https://docs.python.org/3.12/library/fcntl.html#fcntl.flock, https://man7.org/linux/man-pages/man2/flock.2.html). `fcntl` chỉ có trên Unix, nên chỉ được import khi ghi vào `results/` (máy chủ Linux); ghi ra `--out` thì không khóa.
  - **Lưới chạy từ một checkout cố định:** sweep ghi vào `results/` phải chạy từ một worktree riêng, check out đúng commit của lưới (`git worktree add -d <đường dẫn> <commit>`, https://git-scm.com/docs/git-worktree), không từ cây làm việc hằng ngày mà ba người vẫn commit vào. Sweep tính `code_version` một lần lúc bắt đầu. Mỗi tiến trình con tính lại, và nếu khác thì ghi attempt `failed` (lý do: `code_version` đổi giữa chừng); khi đó sweep dừng xếp lượt mới.
  - Khi `code_version` đổi, lượt cũ thành "stale": không xóa, không dùng lại, và phải được ghi vào `docs/experiments/CODE_VERSIONS.md` (INT-6).
- **Phân kỳ:** `LocalTrainer` dừng ngay ở bước có mất mát không hữu hạn và trả về giá trị đó; sau mỗi round engine kiểm mọi tham số. Cả hai việc kiểm đều làm trước khi ghi bất kỳ bản ghi nào của round đó. Khi mất mát hay tham số không còn hữu hạn, lượt dừng lại và ghi `status: diverged` kèm round phân kỳ. Phân kỳ là kết quả tất định nên không chạy lại và được tính là đã xong. Bản ghi không bao giờ chứa NaN hay vô cực (ghi JSON với `allow_nan=False`). Lượt có mất mát lớn nhưng hữu hạn vẫn là `ok`; accuracy thấp của nó đi vào thống kê như mọi lượt khác.
- **`manifest.json`:** `run_id`, `code_version`, `git_commit` (hoặc `none` khi không có git), `dirty`, `env_match`, `attempt`, cấu hình chuẩn hóa đầy đủ, seed và các seed con, `partition_retries`, phiên bản python và mọi gói trong `requirements.lock` đã cài, máy (tên CPU kèm hash của cờ CPU), số luồng đọc lại từ torch, thời điểm; spectral gap (topology tĩnh); trọng số mục tiêu π (`objective_weights`) và lệch mục tiêu; ma trận node × lớp (N × 10) để P7 sinh được từ `results/`.
- **`rounds.jsonl`** (tất định, mỗi round có tối đa một dòng): `round`, `local_steps`, `acc_mean`, `acc_std`, `acc_min`, `acc_max`, `acc_local_mean`, `acc_local_std`, `acc_local_min`, `f1_macro_mean`, `f1_macro_std`, `f1_macro_min`, `acc_avg_model`, `acc_eff_mean`, `drift_pre_combine`, `consensus_distance`, `messages_cum`, `bytes_cum`, `max_node_bytes_cum`, `train_loss_mean`; bắt buộc có `acc_per_node`, `acc_local_per_node` (mảng N phần tử).
- **`timing.jsonl`:** `round`, `wall_time_s` và thời gian từng pha (train, combine, eval).
- **`summary.json`:** một trong ba biến thể `ok`, `diverged`, `failed` (xem `src/dfl/contracts.py`). Bản `ok` có đánh giá cuối trên đủ tập đánh giá (`final_eval_split`: 10,000 ảnh test, hoặc 5,000 ảnh validation với E0a), gồm $\mathrm{Acc}_i$, $\mathrm{Acc}^{loc}_i$ và macro-F1 của từng node (mảng N) cùng acc($\bar w$). Cả ba bản đều có tổng message và byte. Không chứa độ đo suy ra (R@x, AUC, độ dao động, mức sụt): `analysis/` tính chúng từ `rounds.jsonl`.
- **Sao lưu:** `results/` chỉ nằm trên máy chủ, nên phải nén và chép ra ngoài máy chủ sau mỗi nhóm thí nghiệm; không chạy `git clean -x` trong repo trên máy chủ.

---

## 6. Code Style

- **Ngôn ngữ:** tên định danh tiếng Anh; docstring và comment tiếng Việt, ngắn, nói *vì sao* và quy tắc đến từ đâu (KH §, ADR, G); không comment lặp lại code; không để code bị comment-out.
- **Hình dạng code:** ưu tiên hàm thuần và `@dataclass(frozen=True)` cho cấu hình/bản ghi (cấu hình không đổi được sau khi đã tính `run_id`); type hint đầy đủ cho tham số và giá trị trả về; không có trạng thái toàn cục thay đổi được.
- **Ngẫu nhiên tường minh:** nhận `numpy.random.Generator` / `torch.Generator` làm tham số; trong `src/` không dùng `np.random.<hàm>` toàn cục, `random.*` hay `torch.manual_seed`, và không gọi hàm ngẫu nhiên của torch mà thiếu `generator=`. Chỉ `seeding.py` được tạo `SeedSequence`/`Generator`. Mỗi luồng là `SeedSequence(entropy=…, spawn_key=(ID_LUỒNG, …))`, trong đó ID_LUỒNG là hằng số nguyên cố định khai báo trong `seeding.py` (không dùng `hash()` của Python, vốn đổi theo tiến trình). Không luồng nào trùng entropy với luồng khác.
  - *Luồng theo seed của lượt* (entropy = seed): partition, hoán vị vị trí, đồ thị ER, init, batch của node i (spawn_key (ID_BATCH, i)), batch của B1 và B1-4k, matching của T3 (spawn_key (ID_GOSSIP, round)), matching của T3b (spawn_key (ID_GOSSIP_ER, round)). Luồng không phụ thuộc topology, thuật toán hay τ, để cùng seed thì cùng partition, cùng hoán vị vị trí và cùng w⁰ (G9).
  - *Luồng dữ liệu cố định* (entropy = hằng số `DATA_SEED` trong `seeding.py`, giống nhau cho mọi lượt): cách tách validation 5k, tập con đánh giá 2,000 ảnh, dữ liệu tổng hợp của profile smoke. Nhờ vậy mọi lượt, ở mọi seed, dùng cùng validation và cùng ảnh đánh giá.
  - Trạng thái RNG toàn cục của torch (`torch.get_rng_state()`, generator CPU mặc định: https://docs.pytorch.org/docs/2.14/generated/torch.get_rng_state.html) phải giống hệt trước và sau một lượt (T-INV-5). `manual_seed` bị cấm nên RNG toàn cục ở mọi tiến trình bắt đầu từ cùng một trạng thái. Một lời gọi quên `generator=` vì thế sẽ cho cùng số ở mọi seed, làm các seed tương quan mà không test nào khác bắt được.
- **Khởi tạo w⁰:** lấy mẫu tường minh từ `torch.Generator` của luồng init rồi chép vào mọi tham số; không dựa vào RNG toàn cục mà các lớp `nn` dùng khi khởi tạo. Để việc tạo module không tiêu thụ RNG toàn cục, T09 chọn một trong hai cách đã đối chiếu tài liệu 2.14: `torch.nn.utils.skip_init` tạo module mà không khởi tạo tham số (https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.skip_init.html), hoặc tạo module trong `torch.random.fork_rng()`, vốn trả RNG về trạng thái cũ khi thoát (https://docs.pytorch.org/docs/2.14/random.html).
- **Module thuần** không import torch (§3.2); mọi `__init__.py` để trống.
- **Mô hình và tổng hợp:** không BatchNorm (G3); mỗi node tạo đúng một snapshot sở hữu riêng mỗi round (clone + detach, không dùng view của `state_dict()`, G22); tổng hợp cộng theo id người gửi tăng dần, chỉ đọc snapshot và ghi ra bộ đệm mới (không có phép tại chỗ như `mul_`/`add_` trên tensor của snapshot); chỉ ghi vào model sau khi mọi node đã tổng hợp xong (INV-3).
- **Lỗi cấu hình:** raise `ValueError` nói rõ ràng buộc bị vi phạm (ví dụ #C = k cần N ≥ 10, G23); không nuốt lỗi; không dùng `assert` để kiểm tra đầu vào (`python -O` bỏ qua `assert`).
- **An toàn:** chỉ `yaml.safe_load`; `torch.load` chỉ với `weights_only=True` (mặc định ở 2.14, G14); không unpickle dữ liệu từ ngoài; dữ liệu chỉ tải qua torchvision.
- **Định dạng:** `ruff format` + `ruff check`; không thêm `# noqa` hay `type: ignore` mà không ghi lý do.
- **Kích thước thay đổi:** mỗi task tối đa khoảng 300 dòng diff, một commit (KH §8).
- **API framework:** docstring ghi URL tài liệu chính thức đúng phiên bản; không kiểm được thì ghi "UNVERIFIED" (P1.7).

Mẫu phong cách — hàm `metropolis_hastings` có thật trong `docs/spikes/spike_topology_partition.py:33`, viết lại theo quy ước trên (docstring nói *vì sao*, kiểm tra đầu vào theo INV-4 và G7). Đã chạy thử trên một bản sao của repo (cùng `pyproject.toml`), đặt đúng đường dẫn `src/dfl/network/mixing.py` và `tests/unit/test_mixing.py`: `ruff check` và `ruff format --check` sạch cho cả mẫu này và mẫu test ở §7, test pass; hàm cho cùng ma trận với bản spike trên ring, ER (p = 0.3, seed 0) và FC, và từ chối đồ thị có hướng, đa đồ thị, self-loop, node đánh số sai. Khi làm T07 vẫn phải viết test trước. Hàm không kiểm tính liên thông, vì matching của gossip vốn không liên thông; tính liên thông của đồ thị tĩnh do `TopologySchedule` kiểm lúc dựng (INV-4, T06). Tên test mẫu theo quy ước `test_t_inv_<k>_…` (§7), vì nó là một phần của T-INV-9. Với W không đối xứng (uniform), spectral gap phải tính bằng `np.linalg.eigvals`; `eigvalsh` trong mẫu test chỉ đúng vì MH đối xứng.

```python
"""Ma trận trộn cho DFL (module network).

Chỉ phụ thuộc numpy/networkx để test chạy trong mili-giây và không kéo theo torch.
"""

import networkx as nx
import numpy as np


def metropolis_hastings(graph: nx.Graph) -> np.ndarray:
    """W[i, j] = 1 / (1 + max(d_i, d_j)) trên mỗi cạnh; phần còn lại của hàng dồn vào W[i, i].

    Vì sao MH: W đối xứng và ngẫu nhiên kép nên trung bình các model được bảo toàn, và mỗi node
    tính được hàng của mình chỉ từ bậc của láng giềng (Xiao & Boyd 2004; KH §3.3, ADR-005).
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("đồ thị giao tiếp phải vô hướng và không có đa cạnh (INV-4)")
    n = graph.number_of_nodes()
    if set(graph.nodes) != set(range(n)):
        raise ValueError("node phải được đánh số 0..N-1 để khớp chỉ số hàng của W")
    if nx.number_of_selfloops(graph) > 0:
        raise ValueError("không được có self-loop: node không tự gửi message cho mình (G7)")
    degree = dict(graph.degree())
    weights = np.zeros((n, n))
    for i, j in graph.edges():
        weights[i, j] = weights[j, i] = 1.0 / (1 + max(degree[i], degree[j]))
    np.fill_diagonal(weights, 1.0 - weights.sum(axis=1))
    return weights
```

---

## 7. Testing Strategy

- **Framework:** pytest 9.1.1 + pytest-cov 7.1.0; scikit-learn chỉ làm oracle (T-INV-10).
- **Tháp kiểm thử** (KH §5.1):

| Mức | Tỉ trọng | Thời gian | Nội dung |
|---|---|---|---|
| Unit (`tests/unit/`) | ~80% | Mili-giây, không I/O | partition, heterogeneity, topology, mixing, ledger, độ đo, gradient prox, tổng hợp state_dict, độ đo suy ra của `analysis` |
| Integration (`tests/integration/`) | ~15% | Vài giây, dữ liệu tổng hợp | Test bất biến trên engine với 4 node và MLP tí hon; sweep (idempotent, attempt, cây bẩn) |
| E2E (`tests/e2e/`) | ~5% | Dưới 60 s | `python -m dfl run` với profile `smoke` sinh đủ 4 file đúng schema §5.3; `analysis.plots --draft` sinh được hình từ kết quả smoke, và không có `--draft` thì từ chối vì lượt smoke là `dirty` (INT-2) |

- **Quy ước:** Arrange-Act-Assert; tên test mô tả hành vi; mỗi test một hành vi; tên file test không trùng giữa các thư mục (pytest import theo tên file); seed cố định; dung sai số thực ghi rõ; so khớp "trùng từng bit" dùng `torch.equal` / `np.array_equal`; không tải dữ liệu qua mạng (dataset giả cùng shape); không phụ thuộc thời gian thực.
- **Oracle và tham chiếu:** `tests/reference/` chứa hàm chia NIID-Bench sao từ `docs/spikes/`, tham số hóa theo mảng nhãn (golden cho T-INV-8); FedAvg có server viết ngay trong test (T-INV-3); `sklearn.metrics.f1_score(average="macro")` (T-INV-10).
- **TDD:** test FAIL trước, PASS sau. Engine được viết qua ba task T10a–c (`tasks/todo.md`), và mỗi task viết T-INV của mình trước rồi mới viết code: T10a có T-INV-2, 4; T10b có T-INV-1, 3; T10c có T-INV-11, 12. Như vậy không dòng code engine nào có trước test bất biến của chính nó (H§7). Không commit test đỏ (CONSTRAINTS F1, F2).
- **Tên test bất biến:** `test_t_inv_<k>_<hành vi>`, ví dụ `test_t_inv_4_node_order_does_not_change_result`, để `make check-inv` kiểm được đủ 12 ID.
- **Kiểm tra thống kê:** chỉ T-INV-1 (phần gossip) được so theo sai số lấy mẫu: hai phía, ±4 sai số chuẩn, t ≤ 3, seed cố định để không flaky.
- **RNG toàn cục trong mọi test tích hợp:** `tests/conftest.py` có một fixture `autouse` kiểm `torch.get_rng_state()` giống hệt trước và sau mỗi test trong `tests/integration/` và `tests/e2e/` (T-INV-5, §6). Mỗi đường code có ít nhất một test tích hợp, để fixture này thấy được nó: B1, B1-4k, B2, đường server của B3; P0–P3 (và P2'); T0–T4, T3b; prox (μ > 0); uniform; size-corrected.
- **Coverage:** ≥ 80% dòng cho `src/dfl` (trừ `cli.py`) và cho `analysis/` (nhóm chốt QS-5 ở P1.8); ngưỡng ghi trong `CONSTRAINTS.md` và `fail_under` của `pyproject.toml`, không được hạ để qua kiểm tra.
- **Kiểm tra kiến trúc:** import-linter (ENG-6, ENG-7, §9.4), cấu hình ở `[tool.importlinter]` trong `pyproject.toml`.

Mẫu test (dùng hàm ở §6):

```python
import networkx as nx
import numpy as np
import pytest

from dfl.network.mixing import metropolis_hastings


def test_t_inv_9_metropolis_hastings_ring_is_doubly_stochastic_with_closed_form_gap() -> None:
    # Arrange
    ring = nx.cycle_graph(20)

    # Act
    weights = metropolis_hastings(ring)

    # Assert
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    moduli = np.sort(np.abs(np.linalg.eigvalsh(weights)))
    assert 1 - moduli[-2] == pytest.approx(2 / 3 * (1 - np.cos(2 * np.pi / 20)))  # ≈ 0.0326, T-INV-9
```

---

## 8. Boundaries

"(toàn vẹn)" đánh dấu quy tắc toàn vẹn nghiên cứu, lấy từ KH §4.7, §5.3 và H§2.4.

**Always**
- Viết test trước (FAIL rồi mới PASS); giữ các T-INV liên quan luôn xanh.
- `make check-task` trước mỗi commit; `make check-full` ở mỗi checkpoint C1–C5.
- Một task = một commit; cập nhật `tasks/todo.md` và `tasks/SESSION_LOG.md`.
- Dẫn URL tài liệu chính thức cho API framework; ghi "UNVERIFIED" nếu không kiểm được.
- Chỉ ghi vào `results/` khi cây làm việc sạch; `sweep` luôn đòi cây sạch (§5.3).
- (toàn vẹn) Commit thẻ tiền đăng ký `docs/experiments/E<k>.md` trước khi chạy nhóm thí nghiệm đó; điền kết quả sau khi chạy, kể cả kết quả trái giả thuyết.
- (toàn vẹn) Mỗi lượt có manifest đầy đủ (§5.3); giữ mọi attempt, kể cả attempt lỗi.
- (toàn vẹn) Chỉ chọn siêu tham số trên validation 5k; ghi η vào ADR-007 trước khi chạy E0b. Kết quả trên tập test (E0b trở đi, kể cả phép kiểm độ nhạy η/2, 2η) chỉ được báo cáo, không bao giờ được dùng để chọn lại η, τ, μ hay mô hình.
- (toàn vẹn) Code của lưới được đóng băng ở một commit từ E0a tới lượt Tier 2 cuối cùng: mọi sweep của lưới chạy từ worktree của commit đó (§5.3, ADR-008). Code mới trên nhánh chính sau E0a (ví dụ T22) không được dùng cho lượt nào của lưới. Nếu phải sửa một lỗi có test tái hiện thì chạy lại mọi nhóm đã có dưới `code_version` mới, bắt đầu từ E0a. η được chọn lại trên validation theo đúng quy tắc cũ, η mới thay η cũ, và ADR-007 ghi cả hai cùng lý do.
- (toàn vẹn) Định nghĩa độ đo (R@x, AUC, độ dao động, mức sụt, độ dốc của H2) có unit test và được commit trước lượt Tier 1 đầu tiên. Đổi định nghĩa sau khi đã thấy kết quả thì phải ghi vào thẻ thí nghiệm kèm lý do, và báo cáo kết quả theo cả định nghĩa cũ lẫn mới.
- (toàn vẹn) Ghi mọi `code_version` đã sinh lượt thuộc lưới vào `docs/experiments/CODE_VERSIONS.md`, kèm lý do khi nó bị thay thế (INT-6).
- (toàn vẹn) So sánh ghép cặp: cùng seed thì cùng partition, cùng hoán vị vị trí, cùng w⁰ cho mọi topology và thuật toán (G9); so D-PSGD với DFedAvg theo tổng số bước cục bộ và theo byte, không theo round (G10).
- (toàn vẹn) Báo cáo điểm của từng seed, "k/n seed đạt" và "k/n seed phân kỳ", không chỉ trung bình; báo cáo kích thước node và lệch mục tiêu kèm mọi kết quả Non-IID (Dirichlet, #C = k, quantity; G11, G19); diễn giải FC kèm acc^loc và drift (G15).
- (toàn vẹn) Đạt SAN-1…3 trước khi chạy E1 trở đi (§9.3). Không đạt mà debug không tìm ra lỗi thì dừng và báo nhóm (quy trình ở §9.3).
- (toàn vẹn) Sao lưu `results/` ra ngoài máy chủ sau mỗi nhóm thí nghiệm (§5.3).

**Ask first**
- Thêm hoặc nâng phụ thuộc (kể cả dev).
- Sửa SPEC, ADR, `CONSTRAINTS.md`.
- Đổi siêu tham số, lịch đánh giá hay ngưỡng đã chốt (KH §3.5, ADR-006/007, `CONSTRAINTS.md` R7): phải có ADR mới và chạy lại cả nhóm thí nghiệm bị ảnh hưởng. Chỉ được làm trong hai trường hợp: trước khi chạy E0b (khi chưa có kết quả nào trên tập test), hoặc để sửa một lỗi có test tái hiện. Còn lại thuộc Never.
- Chạy tiếp khi một SAN không đạt mà debug không tìm ra lỗi (§9.3).
- Di chuyển hoặc xóa bất cứ gì trong `results/`.
- Chạy lưới lớn hơn 10 lượt (báo kết quả `--dry-run` trước).
- Làm bất cứ mục nào ngoài cột "Trong phạm vi" ở §1.4, kể cả phần mở rộng trước M4, P2', P4 và T5.
- Cắt Tier 2, hoặc hạ thí nghiệm lõi từ 5 xuống 3 seed (KH §3.7; phải ghi rõ trong báo cáo, theo EXP-5).
- Đổi schema manifest/rounds/summary sau khi đã có kết quả Tier 1.

**Never**
- Sửa tay `results/` hoặc `figures/`.
- (toàn vẹn) Xóa lượt chạy lỗi; bỏ seed "xấu" hay chọn lọc lượt chạy để kết quả đẹp hơn.
- (toàn vẹn) Trộn kết quả của nhiều `code_version` trong một hình hay một so sánh.
- (toàn vẹn) Chọn hay đổi siêu tham số (η, τ, μ, B, mô hình), lịch đánh giá hay ngưỡng SAN vì kết quả trên tập test, kể cả khi có ADR mới (G8, G25). Điều này gồm cả việc chạy một lượt validation mới *sau khi* đã thấy kết quả test rồi dựa vào đó để đổi: từ E0b trở đi, các thứ này chỉ được đổi để sửa một lỗi có test tái hiện.
- (toàn vẹn) Đưa lượt `dirty`, lượt lệch môi trường hay hình `--draft` vào `figures/` hoặc vào báo cáo.
- Hạ ngưỡng trong `CONSTRAINTS.md` để qua kiểm tra; skip hay xóa test đang fail khi chưa được duyệt.
- Commit dữ liệu MNIST, `results/`, hay secret/token.
- `torch.load(..., weights_only=False)`; `yaml.load` không dùng SafeLoader.
- Chạy `git clean -x` trong repo trên máy chủ (xóa mất `results/`).
- Đưa đại lượng toàn cục ($\bar w$, trung bình toàn mạng) vào huấn luyện (INV-1). Ngoại lệ duy nhất là n̄ của biến thể size-corrected ở E4 (ADR-005), là đại lượng tính từ dữ liệu của node khác. Tri thức chung cố định của DFL đồng bộ không tính là đại lượng toàn cục, và được ghi trong mục hạn chế (ADR-002): id 0..N−1, N, đồng hồ round, seed chung, w⁰, và bậc của láng giềng (MH cần). Ở đường server (B3, E1c, ô size-weighted của E4), server biết $n_k$ của mọi node như FedAvg gốc; đó là baseline có server, không phải DFL.
- Đếm byte ngoài Transport (INV-2); cập nhật model tại chỗ khi tổng hợp (INV-3); để Observer sửa model hay tiêu thụ RNG của huấn luyện (INV-5); ghi thời gian vào `rounds.jsonl` (INV-6).
- Dùng BatchNorm (G3).
- Tự đánh dấu một gate là "đã duyệt" (H§7).

---

## 9. Success Criteria

Mỗi task còn phải đạt Definition of Done ở H§4.2.

### 9.1 Yêu cầu của đề (REQ-01…14, H§2.1)

| REQ | Tiêu chí chấp nhận (kiểm chứng được) | Kiểm chứng | Task |
|---|---|---|---|
| REQ-01 Mô phỏng DFL không server | Node chỉ đọc model của mình và inbox (INV-1); mọi message đi qua Transport, CommLedger là nơi duy nhất đếm (INV-2). Lịch gossip là hàm thuần của (seed chung, round, N), nên mỗi node tự tính được đối tác của mình mà không cần điều phối viên; trong mô phỏng, engine tính giùm (ADR-009). Tri thức chung mà mọi node cần (id, N, đồng hồ round, seed chung, w⁰) được liệt kê ở ADR-002 và ghi trong mục hạn chế | T-INV-2, T-INV-4, T-INV-12; unit test: hai lịch tạo độc lập với cùng (seed, N) cho cùng đồ thị ở mọi round, gọi theo bất kỳ thứ tự nào; ENG-7 | T08, T10 |
| REQ-02 Non-IID nhiều mức | P0 IID; P1 Dirichlet α ∈ {100, 1, 0.5, 0.1, 0.05}; P2 #C = k, k ∈ {2, 1}; P3 quantity β = 0.5 (ánh xạ bốn loại đề nêu ở §1.4); đúng thuật toán NIID-Bench, khác có chủ đích ở min_size = 64; lấy mẫu lại tối đa 1,000 lần, số lần ghi vào manifest, vượt thì lượt ghi `failed` trước khi huấn luyện. Trên MNIST thật (N = 20, 20 seed): JS trung bình tăng nghiêm ngặt theo IID < α = 100 < α = 1 < α = 0.5 < α = 0.1 < α = 0.05, và #C = 2 < #C = 1; mỗi giá trị lệch khỏi bảng KH §3.2 không quá max(10% giá trị đó, 0.002). Bảng KH §3.2 do spike sinh ra, nên phép so này chỉ kiểm bản cài khớp spike trên nhãn thật. Việc spike đúng thuật toán NIID-Bench dựa trên đọc mã (DIGEST §2.4), không có kiểm tra độc lập. Quantity skew có đo trong Tier 1 (E4a) | T-INV-8; `dfl describe`; hình P7 | T04, T05 |
| REQ-03 Accuracy / F1 | Ở mỗi lần đánh giá: $\mathrm{Acc}_i$, $\mathrm{Acc}^{loc}_i$, $\mathrm{Acc}^{eff}$, macro-F1 của từng node (mean, std, min), acc($\bar w$) trên tập con 2,000 ảnh của tập đánh giá (test; validation với E0a); đánh giá cuối trên đủ tập đánh giá, lưu ở trường riêng | T-INV-10, T-INV-11; test schema e2e | T09–T11 |
| REQ-04 Tốc độ hội tụ | R@x (x ∈ {90, 95}; 97 chỉ là phụ) tính trên chuỗi `acc_mean`, theo số bước cục bộ (G10), quy ra round và byte. Mọi lượt dùng cùng lưới mốc 100, 200, …, 4,000 bước (§5.2), nên lưới đo không phụ thuộc τ. Tuy vậy, accuracy chỉ đo sau tổng hợp, nên phần tụt accuracy trong round (lớn dần theo τ) không được thấy; đây là hạn chế (INT-4). Một *lần vượt* là mốc p có acc(p) ≥ x trong khi mốc trước đó < x. Lần vượt *bền vững* khi mọi mốc trong (p, p + 200] ≥ x − 0.5 điểm %, và sau p còn ít nhất một mốc. R@x là bước nội suy tuyến tính giữa mốc trước p và p, lấy ở lần vượt bền vững đầu tiên; lần vượt không bền thì xét lần kế tiếp. Mốc đầu tiên (100) có acc ≥ x cũng là một lần vượt; nếu nó bền vững thì R@x ghi "≤ 100" và tính là 100 khi lấy trung vị (coi "> ngân sách" là +∞). Không có lần vượt bền vững nào (kể cả khi chỉ vượt ở mốc cuối) thì ghi "> ngân sách"; lượt phân kỳ tính là không đạt. Tổng hợp qua seed bằng trung vị kèm "k/n seed đạt". R@x và bytes@x bị kiểm duyệt hai phía ("≤ 100", "> ngân sách"), nên không dùng CI theo phân phối t cho chúng. Giả thuyết về tốc độ dùng kiểm định dấu trên thứ tự ghép cặp: "> ngân sách" lớn hơn mọi giá trị hữu hạn, còn hai giá trị cùng bị kiểm duyệt ở một cặp là hòa, và hòa tính là không đúng chiều (ADR-007). AUC là trung bình `acc_mean` theo bước trên cùng lưới, quy tắc hình thang (KH §3.6); AUC không bị kiểm duyệt nên dùng được CI t | Unit test trong `analysis`: chuỗi đạt; không đạt; đạt ở mốc đầu; đạt rồi tụt rồi đạt lại (lấy lần bền vững); chỉ vượt ở mốc cuối (không đạt); lưới khác lưới chuẩn thì báo lỗi | T15 |
| REQ-05 Chi phí giao tiếp | Message/round, byte/round, byte tích lũy, tải lớn nhất trên một node, bytes@x. Byte = tổng `nbytes` của các message; `nbytes` tính từ tensor thật (SmallCNN float32: 4 × 46,730 = 186,920 byte/message). Tải của một node = byte gửi + byte nhận, cộng dồn. B3/E1c đi qua đường server của engine: server ảo có chỉ số N (không có dữ liệu) nhận N snapshot, tổng hợp theo $n_k/n$, rồi gửi lại N snapshot. Vì mọi message đi qua Transport, CommLedger đếm đúng 2N message/round (INV-2), và tải lớn nhất là của server. T0 (star có hub là một node dữ liệu, trọng số MH) là topology tham chiếu khác, 2(N − 1) message/round. Siêu dữ liệu đi kèm (ví dụ $n_k$ gửi lên server, 1 số nguyên) không được đếm; nó nhỏ hơn 0.005% một message, và được ghi trong mục hạn chế | T-INV-6; P6 | T08, T15 |
| REQ-06 Độ ổn định | Độ dao động và mức sụt tính trên từng node rồi mới lấy trung bình (G24); độ lệch giữa seed; drift trước tổng hợp; consensus distance sau tổng hợp | T-INV-1, T-INV-11; unit test `analysis`; P5 | T10, T15 |
| REQ-07 So sánh topology | T1 ring, T2 ER liên thông (p = 0.3), T3 gossip trên $K_N$, T4 FC; thêm T3b (E2b); T0 (star) được cài và có trong bảng `describe`, nhưng không chạy trong lưới; spectral gap của topology tĩnh ghi trong manifest. Gossip không có spectral gap cố định: nó được mô tả bằng tốc độ trộn kỳ vọng $1-\sqrt{\lambda_2(\mathbb E[W])}$ (0.312 với N = 20; ADR-009), không đặt chung trục với spectral gap | T-INV-9; unit test topology (liên thông, đúng bậc, matching hợp lệ, gọi lại không đổi kết quả); P3, P8 | T06, T07, T18 |
| REQ-08 Thuật toán tổng hợp | Một vòng ATC tham số (τ, μ, $c_i$) cho D-PSGD (τ = 1, R = 4,000), DFedAvg (τ = 20, R = 200), DFedProx (τ = 20, μ theo QS-2); τ·R = 4,000 ở mọi điều kiện. D-PSGD là tương tự phi tập trung gần nhất của FedSGD nhưng dùng minibatch B = 32 (không phải full-batch) và trọng số MH; cách hiểu này nên được GVHD xác nhận (Q8) | T-INV-3, T-INV-7; E3a; P9 | T09, T10, T19 |
| REQ-09 Accuracy theo round, volume, variance | `rounds.jsonl` đủ trường ở §5.3 (có `acc_per_node`, `acc_local_per_node`); phương sai giữa node trên $\mathrm{Acc}_i$ và $\mathrm{Acc}^{loc}_i$ ở round cuối (std, min, max − min, trung bình 10% node kém nhất) | Test schema e2e; T-INV-11; P1, P4, P6 | T11, T15 |
| REQ-10 Biểu đồ so sánh | P1–P9 và bảng T1 (KH §3.8) sinh bằng một lệnh; bảng màu an toàn cho người mù màu; trục có đơn vị | e2e vẽ hình trên kết quả smoke; INT-1, INT-2 | T16 |
| REQ-11 Kiến trúc tổng quan | Sơ đồ KH §4.1 và luồng một round §4.3 có trong báo cáo và khớp capability map §3 | Duyệt ở Gate G1 và khi review báo cáo | P1.5, T24 |
| REQ-12 Baseline IID, biến cố định | B1 tập trung (10 epoch, B = 32, η riêng chọn trên validation, §5.2) và B1-4k; B2 chỉ cục bộ (W = I); B3 FedAvg có server (trọng số $n_k/n$, đường server, REQ-05), chạy trên IID ngay ở Tier 1 (E0b). Đạt SAN-1…4. Biến cố định (N, model, η, B, τ·R, lịch đánh giá, tập đánh giá, số luồng) và tập seed trùng nhau trong mọi so sánh giữa các lượt DFL; baseline chỉ xuất hiện như đường tham chiếu có nhãn, ghi rõ ngân sách và η của nó (INT-2) | §9.3; INT-2 | T11, T14 |
| REQ-13 Nhận xét và khuyến nghị | Mỗi khuyến nghị (topology, τ, chiến lược tổng hợp, trọng số trộn) dẫn chiếu ít nhất một hình/bảng và run ID, và ghi mức bằng chứng: khuyến nghị chính khi kết luận là "ủng hộ" (5 seed), khuyến nghị tạm khi chỉ là "gợi ý" (3 seed), không khuyến nghị khi "chưa kết luận" (ADR-007). Khuyến nghị về trọng số trộn chỉ dựa trên MH so với uniform; size-corrected là thay đổi bước học theo node, báo cáo riêng (ADR-005). Có mục hạn chế | INT-4, INT-5; P4.3 | T24 |
| REQ-14 Dataset | MNIST qua torchvision: 60k train chia thành 55k cho node và 5k validation phân tầng; 10k test. Fashion-MNIST và CIFAR-10 chỉ là mở rộng sau M4 (H§2.1 ghi kiểm chứng bằng E6; khác biệt ghi ở §10) | Unit test T03 trên dataset giả (shape, đếm lớp, phân tầng) | T03 (T21) |

### 9.2 Test bất biến T-INV-1…12 (KH §5.2)

Cả 12 test phải tồn tại và xanh trước khi chạy bất kỳ lượt nào của Tier 1.

| ID | Điều kiện đạt | Mức | Task |
|---|---|---|---|
| T-INV-1 | lr = 0, khởi tạo khác nhau. Với $\Xi_t=\sqrt{\frac1N\sum_i\lVert w_i^t-\bar w_t\rVert^2}$ (khoảng cách RMS, không bình phương): topology tĩnh (ring, ER, FC) có $\Xi_t\le\rho^t\,\Xi_0$ (sai số số học nhỏ) và trung bình model được bảo toàn với MH. Gossip: mỗi W đối xứng, ngẫu nhiên kép, $W^2=W$; trung bình của W qua $10^4$ matching khớp $\frac12 I+\frac{1}{2(N-1)}(J-I)$ (N chẵn) trong sai số lấy mẫu; nếu kiểm Monte Carlo cho $\Xi_t^2$ thì hai phía, ±4 sai số chuẩn, chỉ với t ≤ 3 | integration | T10 |
| T-INV-2 | W = I cho kết quả trùng khớp huấn luyện chỉ cục bộ | integration | T10 |
| T-INV-3 | FC + trọng số size-weighted tương đương FedAvg có server với trọng số $n_k/n$ (allclose); FedAvg ở đây là đường server của engine (B3/E1c) và một bản FedAvg viết ngay trong test. Khi kích thước node bằng nhau, FC + MH cũng vậy | integration | T10 |
| T-INV-4 | Đảo thứ tự duyệt node không làm đổi kết quả, trùng từng bit; mọi snapshot giữ nguyên sau tổng hợp (so checksum trước và sau) | integration | T10 |
| T-INV-5 | Cùng seed + `code_version` thì `rounds.jsonl` giống hệt. Khác seed thì khác cách chia, và khác w⁰ ở *mọi* tensor tham số. Cùng seed thì cùng cách chia, cùng hoán vị và cùng w⁰ giữa các topology và thuật toán (G9). `torch.get_rng_state()` sau một lượt giống hệt trước lượt đó, tức không có lời gọi RNG toàn cục của torch (§6). Fixture `autouse` ở §7 mở rộng phép kiểm này ra mọi test tích hợp và e2e, nên mọi đường code đều được kiểm | e2e (smoke) + integration | T11 |
| T-INV-6 | N = 20: ring 40, FC 380, gossip trên $K_N$ 20 message/round; ER đúng 2·\|E\| của đồ thị đã lấy mẫu; đường server (B3/E1c) 2N; `nbytes` của snapshot bằng tổng numel × element_size của state (SmallCNN float32: 4 × 46,730); byte = tổng `nbytes` của các message; tải một node = byte gửi + byte nhận; không có message tự gửi | unit | T08 |
| T-INV-7 | μ = 0 tương đương SGD thường; gradient của số hạng prox bằng μ(w − anchor); anchor không có gradient | unit | T09 |
| T-INV-8 | Partition rời nhau, phủ đủ, mọi node ≥ min_size; #C = k: mỗi node đúng k nhãn, mọi nhãn có node sở hữu; α = 1000 thì JS < 0.01; JS (trung bình 3 seed) tăng đơn điệu khi α giảm; **golden:** cùng `Generator` thì trùng tuyệt đối với hàm tham chiếu trong `tests/reference/`; #C = k với N < 10 báo lỗi cấu hình | unit | T04 |
| T-INV-9 | Spectral gap của ring(20) = ⅔(1 − cos(2π/20)) ≈ 0.0326; của FC = 1 | unit | T07 |
| T-INV-10 | Macro-F1 = trung bình trên đủ 10 lớp của $2TP_c/(2TP_c+FP_c+FN_c)$, khớp `sklearn.metrics.f1_score(average="macro", labels=range(10), zero_division=0)` trên nhãn ngẫu nhiên, trên bộ dự đoán chỉ ra một lớp, và trên bộ dự đoán không bao giờ ra một lớp nào đó (không có NaN). $\mathrm{Acc}^{loc}_i$ khớp accuracy có trọng số từng mẫu $P_i(y)/P_{eval}(y)$; $\mathrm{Acc}^{eff}$ khớp trọng số $P_{eff}(y)/P_{eval}(y)$, trên cùng tập đánh giá (sai số ≤ 1e-12) | unit | T09 |
| T-INV-11 | FC + MH: sau tổng hợp mọi node có model trùng nhau (allclose), nên std của $\mathrm{Acc}_i$ bằng 0 và $\mathrm{Acc}^{loc}_i$ của mọi node tính từ cùng một vector recall | integration | T10 |
| T-INV-12 | Tắt Observer, hoặc gọi `graph_at` thêm lần, không làm đổi quỹ đạo huấn luyện | integration | T10 |

### 9.3 Sanity, tái lập và toàn vẹn (KH §5.3)

Ngưỡng có số do nhóm chốt ở P1.8 (QS-3); đối chứng của SAN-2 chốt lại ở P1.9. Cả hai được ghi trong `CONSTRAINTS.md` (R7).

- **SAN-1** (E0b) B1 tập trung (với η riêng của B1, §5.2): accuracy trên đủ 10,000 ảnh test ≥ 98.5% (trung bình 3 seed).
- **SAN-2** (E0b) DFedAvg-FC-IID kém B1-4k không quá 1.0 điểm % (accuracy trên đủ 10,000 ảnh test, so trung bình qua seed {0, 1, 2}). Nhóm chốt đối chứng này ngày 2026-09-30 (QS-3). B1 có η riêng và khoảng 17,190 bước. DFedAvg-FC-IID có η chung, chọn theo ô khó nhất, và 4,000 bước mỗi node. Vì thế khoảng cách tới B1 có thể vượt 1.0 điểm % mà không có lỗi nào. B1-4k cùng η, cùng B và cùng số bước, nên một khoảng cách lớn so với nó mới là dấu hiệu lỗi. Khoảng cách tới B1 vẫn được báo cáo, nhưng không phải điều kiện dừng.
- **SAN-3** (E0b) B2 chỉ cục bộ ở α = 0.1 thấp hơn B2 ở IID ít nhất 20 điểm % (accuracy trên test toàn cục).
- **SAN-4** (E1) Trên cùng topology, accuracy ở α = 100 và ở IID chênh nhau không quá 1.0 điểm %. E1 chạy các ô α = 100 trước (ô IID có sẵn ở E0b), để có SAN-4 sớm.
- **SAN-5** (mọi lượt, kiểm trên `rounds.jsonl`) Bước tổng hợp có tác dụng thật trên code chạy lưới, không chỉ trên MLP tí hon của T-INV. Đây là các điều kiện cấu trúc, đúng khi không có lỗi:
  - FC + MH và đường server: `consensus_distance` ≤ 1e-6 × `drift_pre_combine`, và `acc_std` = 0 ở mọi mốc.
  - Mọi lượt dùng W đối xứng, ngẫu nhiên kép (MH, matching của gossip): `consensus_distance` ≤ `drift_pre_combine` (sai số tương đối 1e-9) ở mọi mốc.
  - Mọi lượt DFL có `messages_cum` > 0 ở mốc đầu tiên; B1 và B2 có 0.
  Không đạt SAN-5 là lỗi, xử lý như SAN-1…3; kiểm ở E0b trước khi chạy E1, và trên mọi nhóm sau đó.
- Không đạt SAN-1…3 thì dừng và debug (skill `debugging-and-error-recovery`) trước khi chạy E1; không đạt SAN-4 thì dừng trước khi chạy E2. Khi debug chỉ được sửa lỗi có test tái hiện; không đổi siêu tham số, mô hình hay ngưỡng để qua SAN, vì SAN đo trên tập test.
- **Khi debug không tìm ra lỗi**, tức mọi T-INV xanh và không có test tái hiện nào fail: dừng và báo nhóm (Ask first). Chỉ nhóm được quyết định chạy tiếp. Khi đó báo cáo ghi SAN không đạt như một phát hiện, kèm số đo và các chẩn đoán (B1-4k, drift, acc^loc). Không bao giờ đổi η, mô hình hay ngưỡng để qua SAN.
- **Tín hiệu dừng** (không phải ngưỡng đạt/không đạt):
  - accuracy tăng khi α giảm → nghi lỗi đảo chiều α (G1) trước khi tin;
  - std giữa node bằng 0 trên FC + MH là do cấu trúc (G15), không phải một phát hiện; ngược lại, std khác 0 trên FC + MH là lỗi (SAN-5);
  - kết quả đúng chiều giả thuyết nhưng lớn bất thường cũng phải được kiểm như kết quả ngược chiều, để việc săn lỗi không chỉ nhắm vào kết quả trái kỳ vọng (INT-5).
- **REP-1** Clone sạch → check out `git_commit` ghi trong manifest → cài đặt theo §4 (từ `requirements.lock`, nên kiểm môi trường của `run` đạt) → chạy lại 2 `run_id` chọn ngẫu nhiên trên cùng máy chủ, cùng số luồng → `code_version` trùng, và `rounds.jsonl`, `summary.json` trùng từng byte (`cmp`).
- **INT-1** Mọi hình và bảng trong `figures/` sinh bằng `python -m analysis.plots --results results/ --out figures/`; không sửa tay. Lệnh này từ chối chạy khi `git status --porcelain -- analysis` không rỗng. Nó ghi `figures/PROVENANCE.json`: mỗi hình kèm lệnh, `code_version`, `analysis_version`, commit, và danh sách run_id đã dùng. `analysis_version` là sha256 nội dung `analysis/*.py`, tính như `code_version`. Không có `--draft` thì lệnh chỉ đọc lượt trong `results/`, và báo lỗi khi `--results` trỏ ra ngoài `results/`. Có `--draft` thì lệnh:
  - nhận lượt `dirty`, lượt `env_match: false` và lượt ở ngoài `results/` (smoke, demo);
  - nhận lịch đánh giá riêng của profile, ví dụ smoke đánh giá mỗi round;
  - bỏ qua kiểm INT-6;
  - gắn nhãn "DRAFT" lên mọi hình, và không được ghi vào `figures/`.
- **INT-2** `analysis` báo lỗi rõ ràng (có unit test) trong các trường hợp sau:
  - một hình hay một so sánh trộn nhiều `code_version` (trừ khi được cho phép tường minh, KH §4.7);
  - có lượt `dirty: true` hoặc `env_match: false`, trừ ở chế độ `--draft` (INT-1);
  - các lượt được so sánh khác nhau ở một biến cố định (dataset, N, model, η, B, τ·R, lịch đánh giá, tập đánh giá, số luồng, máy), trừ biến là trục của hình đó;
  - một so sánh ghép cặp gồm các điều kiện có tập seed khác nhau mà không khai báo tập seed chung. Tập seed chung phải được khai báo trước trong thẻ tiền đăng ký, không suy ra sau khi đã biết lượt nào có. Hình chỉ dùng các seed đó và ghi chúng trên hình; ví dụ quét τ dùng seed {0, 1, 2} cho cả ô τ = 20. Một lượt thiếu hay phân kỳ trong tập seed đã khai báo không bị bỏ khỏi tập đó, mà được báo cáo theo INT-3.
  Baseline B1, B1-4k, B2 chỉ được miễn kiểm biến cố định khi vẽ như đường tham chiếu có nhãn; nhãn phải ghi ngân sách và η của baseline. Khi lượt đường server (B3, E1c, ô size-weighted của E4) là một ô được so sánh (H6, E4), nó được kiểm như mọi lượt DFL. SAN-2 được định nghĩa riêng ở trên, không qua INT-2.
- **INT-3** Không attempt nào bị xóa; phụ lục báo cáo liệt kê số attempt `failed`, `diverged` và lý do. Lượt phân kỳ không bị lọc khỏi thống kê: ô có seed phân kỳ báo cáo "k/n seed phân kỳ" cùng điểm của từng seed, không lấy trung bình qua NaN; `analysis` báo lỗi khi gặp NaN (không dùng `skipna` mặc định của pandas).
- **INT-4** Mọi nhận định trong báo cáo dẫn chiếu hình/bảng và run ID; khuyến nghị có mục hạn chế: lr dùng chung cho mọi τ (có thể bất lợi cho τ = 1 và τ = 50) và được chọn theo ô khó nhất (có thể làm nhỏ hiệu ứng Non-IID đo được); gossip trên $K_N$; lệch mục tiêu; khởi tạo chung và tri thức chung khác của DFL đồng bộ (ADR-002); n̄ của size-corrected là đại lượng toàn cục, và size-corrected tương đương bước học theo node; D-PSGD khác FedSGD gốc; quantity skew chỉ có một mức (β = 0.5); accuracy chỉ đo sau tổng hợp, nên phần tụt trong round (lớn dần theo τ) không được thấy và τ lớn có thể được lợi; siêu dữ liệu ($n_k$, bậc láng giềng) không được đếm vào byte; bảy kiểm định chính, mỗi kiểm định có p = 1/32 với 5 seed: nếu mọi giả thuyết H0 đều đúng thì xác suất có ít nhất một kết luận "ủng hộ" sai là tối đa khoảng 20% ($1-(31/32)^7$); hồi quy theo spectral gap (RQ2) chỉ có ba cụm giá trị và lẫn với bậc, số message, byte.
- **INT-5** Mỗi kết luận chính được một reviewer ngữ cảnh mới tìm cách bác bỏ (doubt-driven, KH §5.3) trước khi chốt báo cáo.
- **INT-6** Mọi `code_version` có lượt trong `results/runs/` có một dòng trong `docs/experiments/CODE_VERSIONS.md` gồm: ngày, commit, các nhóm thí nghiệm đã chạy. Nếu nó bị thay thế thì dòng đó ghi thêm lý do, cùng commit sửa lỗi kèm test tái hiện. `analysis.plots` báo lỗi khi thiếu dòng. Phụ lục báo cáo liệt kê mọi `code_version` bị thay thế và lý do, nên một bộ kết quả không thể bị thay lặng lẽ bằng một "bản sửa lỗi".

### 9.4 Kỹ thuật

- **ENG-1** `make check-task` xanh trước mỗi commit; `make check-full` xanh ở mỗi checkpoint C1–C5.
- **ENG-2** Coverage dòng ≥ 80% cho `src/dfl` (trừ `cli.py`) và `analysis/` (chốt ở P1.8).
- **ENG-3** `ruff check .` và `ruff format --check .` sạch.
- **ENG-4** `python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke` xong dưới 60 s trên máy chủ và sinh đủ 4 file đúng schema §5.3, với `rounds.jsonl` có 6 dòng (đánh giá mỗi round); chạy trong `tests/e2e/test_smoke_run.py`.
- **ENG-5** Mục tiêu: `make check-task` xong dưới 90 s; vượt thì chỉ cảnh báo (CONSTRAINTS.md T1).
- **ENG-6** Module thuần không nạp torch, kể cả gián tiếp: contract `pure-no-torch` của import-linter (`lint-imports`) cho `dfl.contracts` và `dfl.network`; `dfl.data.partition`, `dfl.data.heterogeneity` được thêm ở T04. Import dưới `TYPE_CHECKING` không tính (`exclude_type_checking_imports = true`). Riêng `dfl.contracts` còn có test chạy trong tiến trình con (`tests/unit/test_contracts_smoke.py`).
- **ENG-7** Hướng phụ thuộc §3.2 được kiểm bằng các contract `layers`, `no-analysis-in-src`, `analysis-reads-only` (chỉ cho phép `dfl.contracts`) của import-linter. Trong contract `layers`, tầng giữa khai báo `dfl.data | dfl.network | dfl.learning | dfl.metrics | dfl.io`. Dấu `|` của import-linter nghĩa là các module cùng tầng độc lập, không được import lẫn nhau, nên bốn module lá độc lập với nhau. Cấm RNG toàn cục, kiểm bằng ruff `NPY002`, `S311` và `make check-floor`: `torch.manual_seed` trong `src/`, tạo `Generator`/`SeedSequence` ngoài `seeding.py`. Lời gọi RNG của torch thiếu `generator=` thì do T-INV-5 và fixture ở §7 bắt, qua trạng thái RNG toàn cục. Contract `no-dev-deps` cấm `dfl` và `analysis` import gói chỉ có trong nhóm dev (scipy, sklearn, joblib, pytest). Các gói này không nằm trong `requirements.lock` và bị bỏ qua khi kiểm môi trường, nên nếu code chạy dùng chúng, file khóa sẽ thiếu mà không ai thấy.
- **ENG-8** Unit và integration test không truy cập mạng và không cần MNIST thật.
- **ENG-9** Sweep:
  - bỏ qua `run_id` đã có attempt xong (`ok` hoặc `diverged`);
  - tự chạy lại attempt không có `summary.json`; chỉ chạy lại attempt `failed` khi có `--retry-failed`;
  - từ chối khi cây bẩn, hoặc khi ghi vào `results/` mà một gói trong `requirements.lock` lệch phiên bản đã cài;
  - bỏ qua (không chờ) lượt mà một tiến trình khác đang giữ khóa của run_id đó; hai sweep của hai nhóm chạy song song được (§5.3);
  - khi ghi vào `results/`, chạy từ một worktree riêng check out đúng commit của lưới; tiến trình con nào thấy `code_version` khác của sweep thì ghi `failed`, và sweep dừng xếp lượt mới (§5.3);
  - mỗi lượt chạy trong một tiến trình con mới, không dùng `fork` (§5.2);
  - commit chỉ đổi `docs/`, `tasks/`, `analysis/` hay `configs/` không làm đổi `code_version`; cài thêm hay nâng công cụ dev không làm đổi kết quả kiểm môi trường;
  - `--dry-run` in đúng số lượt tổng / đã xong / failed / sẽ chạy.
- **ENG-10** `sweep` luôn đòi cây sạch; `run` chỉ ghi vào `results/` khi cây sạch và môi trường khớp `requirements.lock`, và từ chối tạo attempt mới cho run_id đã xong. Khi `run` ghi ra `--out` ngoài `results/`, các trường `dirty` và `env_match` trong manifest phản ánh đúng trạng thái cây và môi trường.
- **ENG-11** Demo: `configs/experiments/demo.yaml` (2 lượt) và lệnh vẽ hình (`--draft`) xong trong tổng cộng ≤ 5 phút trên laptop demo, không cần mạng (MNIST tải sẵn) (QS-8). Laptop có thể khác phiên bản gói so với máy chủ: sweep ghi ra `build/demo` vẫn chạy, và manifest ghi `env_match: false`.

### 9.5 Hoàn thành thực nghiệm (KH §3.7)

- **EXP-1** Mỗi nhóm thí nghiệm có thẻ tiền đăng ký `docs/experiments/E<k>.md` được commit trước khi chạy.
- **EXP-2** Tier 1: E0a 24, E0b 43, E1 70, E2 30, E3a 24, E4a 6 = **197** `run_id` (bảng dưới). Mọi lượt đều có attempt xong (`ok` hoặc `diverged`) và cùng một `code_version`.
- **EXP-3** Tier 2: E1c 15, E2b 9, E3b 42, E4 15 = **81** `run_id`, cộng tối đa 24 lượt có điều kiện (ADR-007). Mọi lượt đều có attempt xong; các ô dùng lại giữa các nhóm thí nghiệm có cùng `code_version`.

| Nhóm | Tier | Lưới (DFedAvg, τ = 20, MH nếu không ghi khác; ô lấy từ nhóm khác không tính lại) | Seed | Lượt |
|---|---|---|---|---|
| E0a | 1 | η ∈ {0.01, 0.05, 0.1} × {FC-IID, ring-α0.1, ring-#C2} ở τ = 20 (9); η × {FC-IID, ring-α0.1} ở τ = 1 (6) và ở τ = 50 (6); B1 × 3 η (3). Chỉ đánh giá trên validation | 1 (seed 100) | 24 |
| E0b | 1 | B1 (3); B1-4k (3); B2 × {IID, α = 0.1} (6); B3 × IID (3); IID × {T1, T2, T3, T4} (20, 5 seed); độ nhạy {η/2, 2η} × {T1, T4} × {IID, α = 0.1} (8, seed 0) | 3 / 5 / 1 | 43 |
| E1 | 1 | {T1, T4} × {α = 100, 1, 0.5, 0.1, 0.05, #C = 2, #C = 1}; ô IID lấy từ E0b | 5 | 70 |
| E2 | 1 | {T2, T3} × {α = 1, α = 0.1, #C = 2}; ô IID lấy từ E0b, ô T1/T4 lấy từ E1 | 5 | 30 |
| E3a | 1 | {D-PSGD (τ = 1), DFedProx (μ = 1, QS-2)} × {T1, T4} × {IID, α = 0.1}; DFedAvg lấy từ E0b/E1 (seed 0–2) | 3 | 24 |
| E4a | 1 | P3 β = 0.5 × {T2, T4} | 3 | 6 |
| E1c | 2 | B3 (đường server) × α ∈ {100, 1, 0.5, 0.1, 0.05} | 3 | 15 |
| E2b | 2 | T3b × {IID, α = 0.1, #C = 2} | 3 | 9 |
| E3b | 2 | μ = 0: τ ∈ {5, 50} × {T1, T4} × {IID, α = 0.1} (24); μ = 0.1 (QS-2), τ = 20 × {T1, T4} × {IID, α = 0.1} (12); μ = 1, τ = 50 × {T1, T4} × α = 0.1 (6) | 3 | 42 |
| E4 | 2 | β = 0.5: T2 × {uniform, size-corrected} và T4 × size-weighted (9); α = 0.1: T2 × {uniform, size-corrected} (6). Ô MH lấy từ E4a, E1, E2; ô size-weighted ở α = 0.1 lấy từ E1c | 3 | 15 |
| Có điều kiện | 2 | η riêng cho τ = 1: các ô D-PSGD của E3a (12); η riêng cho τ = 50: các ô μ = 0, τ = 50 của E3b (12) | 3 | ≤ 24 |

- **EXP-4** Mỗi giả thuyết H1–H7 có kết luận ủng hộ / gợi ý / không ủng hộ / chưa kết luận theo quy tắc quyết định đăng ký trước (ADR-007 và thẻ `E<k>.md`), kèm hình và run ID, ghi trong thẻ thí nghiệm và trong báo cáo. Kết luận "gợi ý" (3 seed) chỉ cho phép khuyến nghị tạm (REQ-13). Mỗi giả thuyết có đúng một phép so sánh chính (độ đo, các ô, chiều kỳ vọng), khai báo trong thẻ trước khi chạy; mọi phép so sánh khác chỉ để khám phá, được báo cáo nhưng không dùng để kết luận "ủng hộ".
- **EXP-5** Nếu cắt Tier 2 hay hạ số seed: có quyết định của nhóm (Ask first) và báo cáo ghi rõ. Mọi H/RQ/REQ phụ thuộc phần bị cắt (bảng dưới) được ghi "chưa kết luận (cắt vì ngân sách)" và không có khuyến nghị cho chúng.

| Nhóm thí nghiệm | Tier | Phục vụ |
|---|---|---|
| E0a, E0b | 1 | Chọn η; SAN-1…3; baseline B1, B1-4k, B2, B3 (REQ-12) |
| E1 | 1 | RQ1, H1; SAN-4 |
| E2 | 1 | RQ2, H2, H3, một phần H4 |
| E3a | 1 | REQ-08 (đủ ba thuật toán đề yêu cầu); khuyến nghị tạm về chiến lược tổng hợp (3 seed) |
| E4a | 1 | REQ-02 (quantity skew được đo), một phần RQ4 |
| E1c | 2 | H6 (đối chứng lệch mục tiêu, G19) |
| E2b | 2 | H4 (so sánh công bằng với gossip trên $K_N$) |
| E3b | 2 | RQ3, H5, H7 |
| E4 | 2 | RQ4, H6; khuyến nghị về trọng số trộn (chỉ MH so với uniform); size-corrected báo cáo riêng như tái trọng số mục tiêu (ADR-005) |

---

## 10. Khác biệt so với KH v0.2 và H

SPEC được ưu tiên hơn KH và H ở các điểm dưới đây (yêu cầu của đề thì không bị ghi đè). KH sẽ được cập nhật theo bảng này sau khi SPEC được duyệt. Các dòng 13–23 được thêm ở vòng phản biện 1 của P1.9, các dòng 24–30 ở vòng 2, các dòng 31–37 ở vòng 3 (`docs/decisions/REVIEW-P1.9.md`), các dòng 38–39 ở P2.1 (`tasks/plan.md`).

| # | KH v0.2 | SPEC | Lý do |
|---|---|---|---|
| 1 | §7: phân vai cho 4 người | 3 người (QS-9, chốt ở P2.1): TV1 `data`, `learning`, độ đo và phân tích E1/E1c/E4; TV2 `network`, engine, `run`, baseline, hiệu năng, T22; TV3 nền tảng, io, sweep, `analysis`, thí nghiệm, báo cáo | QĐ-1; giảm tải cho TV2 trên đường găng (`tasks/plan.md`) |
| 2 | A2, §3.5, §3.7, §9: có nhánh GPU / Colab | Chỉ CPU: bỏ cờ tất định cho GPU, cấu hình đánh giá 10k trên GPU và ước tính thời gian GPU | QĐ-5, Q4 |
| 3 | §3.5: profile `smoke`, `fast`, `full` | Thêm profile `demo` và `configs/experiments/demo.yaml` | QĐ-3 |
| 4 | §4.7: chỉ có `f1_macro_mean` | Thêm `f1_macro_std`, `f1_macro_min`; macro-F1 từng node ở đánh giá cuối | Q3 |
| 5 | §4.6: `src/dfl/metrics/{…,convergence,communication}.py` | `convergence` chuyển thành `analysis/convergence.py`; bỏ `metrics/communication.py` (số liệu message/byte đọc từ CommLedger) | KH §4.7: sửa code phân tích không được làm đổi `code_version`; INV-2 (QS-4) |
| 6 | §4.6: chưa nêu chỗ đặt sweep và Observer | `sim/sweep.py`, `sim/observer.py` | `cli.py` bị loại khỏi coverage, nên logic sweep không được nằm ở đó |
| 7 | §4.6: `e0_sanity.yaml`, `e3_tau_mu.yaml` | Tách `e0a_lr` / `e0b_sanity` và `e3a_algorithms` / `e3b_tau_mu` | E0a phải xong trước E0b; E3a thuộc Tier 1, E3b thuộc Tier 2 |
| 8 | §4.7: chỉ sweep từ chối khi cây bẩn | `run` cũng vậy khi ghi vào `results/`; `run --out` ra `build/` thì được, manifest ghi `dirty: true`; `sweep` luôn đòi cây sạch, kể cả khi ghi ra `--out` (đúng CLAUDE.md) | Kết quả trong `results/` phải check out lại được (QS-7) |
| 9 | §3.2: "lưu ma trận node × lớp" (chưa nói lưu ở đâu) | Lưu trong `manifest.json` | P7 sinh được từ `results/` |
| 10 | (chưa có) | CLI `describe`, `download`, `sweep --dry-run`, `analysis.aggregate` | M2 cần P7 và spectral gap trước khi có engine; demo không cần mạng; quy tắc "lưới > 10 lượt" (QS-7) |
| 11 | (chưa có) | Kiểm tra kiến trúc ENG-6, ENG-7 bằng import-linter (thêm `network/transport.py` vào nhóm không nạp torch, vì payload là đối tượng mờ); `analysis` kiểm tra biến cố định (INT-2); đối chiếu JS trên MNIST thật với bảng KH §3.2 (REQ-02) | Kiểm được hướng phụ thuộc, REQ-12 và cách chia trên nhãn thật |
| 12 | §3.4, ADR-006, H§2.4 G16: μ ∈ {0.1, 1}, lập luận ở η = 0.05 | Nếu η chọn ở E0a khác 0.05 thì giữ cường độ ημτ ∈ {0.1, 1} ở τ = 20 (QS-2) | H5 và lý do loại μ = 0.01 (G16) đều dựa trên ημτ |
| 13 | H§2.1 REQ-02: thiết kế "§3.2 P0–P4" | P0–P3 bắt buộc, P4 tùy chọn; bốn loại phân phối đề nêu được ánh xạ ở §1.4 | P4 ($p_{major}$) không có trong đề |
| 14 | H§2.1 REQ-14: kiểm chứng bằng E6 | Unit test T03; E6 (Fashion-MNIST) là mở rộng sau M4 | Đề cho chọn MNIST hoặc Fashion-MNIST; Q5, QĐ-4 |
| 15 | §4.7: `results/<exp>/<run_id>/attempt-<k>/`; `code_version` = tree hash của commit (`git rev-parse HEAD:src`); sweep chạy lại mọi lượt lỗi | `results/runs/<run_id>/…` và chỉ mục theo nhóm; `code_version` = sha256 nội dung file trên đĩa (chuẩn hóa xuống dòng), không cần git; mỗi run_id nhiều nhất một attempt xong; attempt `failed` chỉ chạy lại khi có `--retry-failed`; khóa `flock`; `run` và `sweep` kiểm môi trường theo `requirements.lock` chỉ gồm gói chạy; thêm `git_commit`, `env_match` vào manifest | Dùng lại ô giữa các nhóm; lượt chạy từ cây bẩn không mang nhầm `code_version` của commit sạch; chạy được khi không có git; lỗi tất định không bị chạy lại mãi; nâng công cụ dev không làm kết quả thành stale; tái lập được |
| 16 | §4.7: `status` ∈ {ok, failed} | Thêm `diverged`: dừng lượt, không chạy lại, được tính là xong, không có NaN trong bản ghi | Phân kỳ là kết quả tất định; tránh NaN lọt khỏi thống kê |
| 17 | §3.5: số luồng CPU theo cấu hình worker | Số luồng mỗi lượt là hằng số của lưới (mặc định 1), nằm trong `run_id`; mỗi lượt một tiến trình con mới (không `fork`), biến môi trường OMP/OpenBLAS/MKL đặt trước khi tiến trình con khởi động | Đo: 8 luồng không nhanh hơn 1 luồng; kết quả không phụ thuộc `--workers`; biến môi trường chỉ có tác dụng khi đặt trước |
| 18 | §3.5 và bẫy G12 (dày lúc đầu, thưa về sau): mốc đánh giá mỗi 20 bước tới 1,000 rồi mỗi 100 bước (khoảng 80 mốc, mật độ khác nhau theo τ); §3.6: R@x "3 điểm kế tiếp" | Mốc 100, 200, …, 4,000 (40 mốc) cho mọi τ (nhóm chấp nhận ở P1.9, kể cả việc R@90 của ô IID thô hơn); R@x theo lần vượt bền vững đầu tiên với cửa sổ 200 bước (REQ-04); round cuối luôn được đánh giá; lịch là tham số của profile | Ở lưới cũ τ = 50 có 50 mốc còn τ ≤ 20 có 80, nên R@x, AUC và độ dao động lệch theo τ; lưới chung bỏ được độ lệch đó và giảm một nửa chi phí đánh giá; profile smoke (12 bước) vẫn có dòng đánh giá |
| 19 | §3.4: B1 "10 epoch" (không nêu lr) | B1 có η riêng chọn trên validation (3 lượt ở E0a) | SAN-1 không được fail chỉ vì η được chọn cho ô Non-IID khó nhất |
| 20 | §3.7: Tier 1 = 176, Tier 2 = 87 lượt | Tier 1 = 197 (thêm E4a 6 lượt; η cho τ = 50: 6 lượt; η của B1: 3 lượt; B1-4k: 3 lượt; B3 trên IID: 3 lượt), Tier 2 = 81, cộng tối đa 24 lượt có điều kiện; bảng chi tiết ở EXP-2 | Quantity skew được đo ngay trong Tier 1 (REQ-02); kiểm η ở cả hai đầu τ = 1 và τ = 50; KH không chạy B3 ở Tier 1 dù REQ-12 cần nó; B1-4k tách được lỗi khỏi hiệu ứng ngân sách ở SAN-2 |
| 21 | §3.7: khoảng 10 phút/lượt, Tier 1 + 2 khoảng 44–57 giờ-CPU | Đo ngày 2026-09-30: khoảng 20 phút-lõi/lượt với lưới 40 mốc (huấn luyện 14, đánh giá 6); Tier 1 khoảng 64 giờ-lõi, Tier 1 + 2 khoảng 92 giờ-lõi (100 nếu có đủ 24 lượt có điều kiện; khoảng 130 kể cả 30% dự phòng). Thời gian thực bằng giờ-lõi chia số lõi rảnh: với 1 lõi rảnh là 4–5.5 ngày | Số đo thật; máy chủ dùng chung (tải khoảng 7 trên 8 nhân) nên số lõi rảnh mới là rủi ro chính (ADR-007) |
| 22 | §4.2, §4.4: `analysis` đọc kết quả qua `dfl.io` | `analysis` chỉ import `dfl.contracts`, tự đọc JSON/JSONL | Sửa code đọc không làm đổi `code_version` |
| 23 | §4.5 INV-1: không có ngoại lệ | Ngoại lệ duy nhất: n̄ (kích thước node trung bình) của biến thể size-corrected ở E4, được tính một lần trước khi huấn luyện | $c_i$ cần một đại lượng toàn cục; ghi là hạn chế (ADR-005) |
| 24 | §5.3: SAN-2 so DFedAvg-FC-IID với B1 | SAN-2 so với B1-4k (η chung, 4,000 bước; 3 lượt ở E0b), giữ ngưỡng 1.0 điểm % (nhóm chốt 2026-09-30, QS-3); khoảng cách tới B1 chỉ báo cáo; thêm quy trình khi SAN không đạt mà không tìm ra lỗi | B1 có η riêng và ngân sách gấp khoảng 4 lần, nên SAN-2 so với B1 có thể fail mà không có lỗi và không có lối ra hợp lệ |
| 25 | §3.5: E0a 1 seed (không nêu seed nào); "accuracy validation cuối"; "η tốt nhất" của τ = 1 chưa định nghĩa | E0a dùng seed 100, ngoài {0…4} (nhóm chấp nhận ở P1.9); accuracy validation = trung bình `acc_mean` ở 5 mốc cuối trên tập con validation; η tốt nhất của τ = 1 và τ = 50 theo cùng quy tắc, hòa thì giữ η chung (ADR-007) | Không chọn η trên chính partition và w⁰ của seed đánh giá; một điểm cuối bị nhiễu dao động (ring-#C2); quy tắc kích hoạt 24 lượt bổ sung phải đăng ký trước |
| 26 | §3.5: E0b kiểm độ nhạy η/2, 2η, "kết luận H2 phải giữ nguyên hướng" | Chỉ để mô tả: báo cáo giá trị của từng cấu hình; vì chỉ có 1 seed nên không kết luận về độ bền của H2; không bao giờ dùng để chọn lại η | Phép kiểm chạy trên tập test (G8, G25); dấu của một hiệu tương tác trên 1 seed có thể đổi chỉ vì nhiễu |
| 27 | §3.4: B3 "cập nhật giống hệt FC size-weighted; chỉ khác chi phí" | B3 và E1c chạy qua đường server của engine (server ảo chỉ số N), nên 2N message/round được Transport đếm | Chi phí star phải đi qua Transport (INV-2), không tính tay |
| 28 | §3.3: ER 108 message/round, gossip 0.312 trong cột spectral gap | ER trung bình khoảng 115 message/round (108 là đồ thị của seed 0; 72–154 tùy đồ thị); 0.312 là tốc độ trộn kỳ vọng của gossip, không phải spectral gap (spectral gap của $\mathbb E[W]$ là 0.526) | Đo lại trên 2,000 đồ thị ER liên thông; tránh đặt gossip lên trục spectral gap (ADR-005, ADR-009) |
| 29 | §3.1: chưa nêu tri thức chung của các node | Liệt kê tri thức chung (id, N, đồng hồ round, seed chung, w⁰) là giả định của DFL đồng bộ, ghi trong mục hạn chế | "Không có điều phối viên" phải nói rõ được giả định gì (ADR-002) |
| 30 | §4.7: hình chỉ sinh bằng script | `analysis/` phải sạch khi ghi vào `figures/`; PROVENANCE ghi `analysis_version`; chế độ `--draft` cho smoke và demo; `docs/experiments/CODE_VERSIONS.md` (INT-6) | Độ đo suy ra nằm ngoài `code_version`, nên cần dấu vết riêng; lượt smoke luôn `dirty`; bộ kết quả bị thay thế phải được báo cáo |
| 31 | §5.3: sanity SAN-1…4 | Thêm SAN-5: kiểm trên `rounds.jsonl` của lượt thật rằng bước tổng hợp có tác dụng (FC + MH đồng thuận, consensus distance không lớn hơn drift, có message) | T-INV chỉ chạy trên MLP tí hon; lỗi nối cấu hình vào engine có thể biến mọi lượt DFL thành B2 mà SAN-1…4 vẫn đạt |
| 32 | §4.7: `run_id` = hash(cấu hình + `code_version`) | `run_id` còn gồm máy (tên CPU + hash cờ CPU); INT-2 kiểm thêm dataset và máy | Lượt từ máy khác không bị bỏ qua nhầm hay trộn lặng lẽ (G13) |
| 33 | §4.7: sweep chạy từ cây làm việc; khóa chung | Lưới vào `results/` chạy từ worktree riêng check out đúng commit; tiến trình con kiểm lại `code_version`; khóa theo từng run_id, nên các nhóm chạy song song được | Lưới chạy nhiều ngày trong khi nhóm vẫn commit; một khóa chung chặn `--workers` và chặn E1 chạy song song với E3a/E4a |
| 34 | §3.6: R@x tổng hợp bằng trung vị, không đạt là +∞ | Giả thuyết về tốc độ dùng kiểm định dấu trên thứ tự ghép cặp; không dùng CI t cho R@x và bytes@x | R@x bị kiểm duyệt hai phía; CI t không xác định khi có +∞ |
| 35 | §3.7 thứ tự cắt, và bẫy G19 (đối chứng E1c) | Nếu phải cắt E1c thì H6 ghi "chưa kết luận (cắt vì ngân sách)", và G19 chỉ còn được giảm thiểu bằng việc báo cáo TV lệch mục tiêu kèm mọi kết quả Non-IID | Thứ tự cắt đã chốt ở ADR-007; lệch khỏi bẫy phải được ghi (đầu SPEC) |
| 36 | §2.3, §3.8: kiểm định cho H1–H7 | Mỗi giả thuyết có đúng một phép so sánh chính, khai báo trước trong thẻ; phần còn lại chỉ để khám phá (EXP-4) | Nhiều phép so sánh làm tăng xác suất kết luận sai (INT-4) |
| 37 | §3.2: P2' (shards) tùy chọn | Cài sẵn P2' ở T04b, chỉ dùng trong `describe`, không chạy trong lưới trừ khi GVHD hay nhóm yêu cầu (QS-11) | Nếu GVHD muốn dạng shards cho "pathological partition" (Q9), thêm lượt sau E0a mà không phải đổi `code_version` |
| 38 | H P3.3: viết cả sáu T-INV của engine trước rồi mới viết engine | Engine chia ba task T10a–c; mỗi task viết T-INV của mình trước (§7) | Gộp lại thì vượt khoảng 300 dòng một task; test đỏ không commit được (CONSTRAINTS F1, F2) |
| 39 | §4.6: cây thư mục dự kiến | Thêm `sim/run.py`, `sim/describe.py`, `analysis/{checks,sanity,provenance}.py`, `tests/conftest.py`, `configs/experiments/e4a_quantity.yaml`, `docs/{PERF,DEMO,VALIDATION}.md`, `docs/report/` (§5.1) | Logic không được nằm trong `cli.py`; mỗi task tối đa 5 file (`tasks/plan.md` K8) |

---

## 11. Open Questions

### 11.1 Chờ GVHD (OQ, tin nhắn đã soạn)

| ID | Câu hỏi | Ảnh hưởng |
|---|---|---|
| Q1 | Mô phỏng đơn tiến trình có message-passing có được coi là "hệ thống P2P" không? | §1.4 (T22 thành bắt buộc nếu không), ADR-002 |
| Q2 | Ngày nộp, ngày bảo vệ, định dạng báo cáo, có cần slide/demo không? | Lịch KH §7, T24, T25, QS-8 |
| Q8 (đề xuất hỏi thêm) | Dùng D-PSGD (minibatch B = 32, trọng số MH) làm bản phi tập trung của FedSGD có được không? | REQ-08, ADR-006 |
| Q9 (đề xuất hỏi thêm) | Bốn loại phân phối của đề được ánh xạ như SPEC §1.4 ("pathological partition" = #C = k của NIID-Bench, "quantity-based skew" = P3) có đúng ý đề không, hay đề muốn dạng shards của McMahan? | REQ-02, §1.4, QS-11 |

### 11.2 Cần nhóm quyết (SPEC đang dùng giá trị mặc định)

| ID | Câu hỏi | Mặc định đang dùng | Phương án khác | Ảnh hưởng |
|---|---|---|---|---|
| QS-1 | File nào là "file khóa phụ thuộc" trong `code_version`? | `requirements.lock` = `pip freeze --exclude-editable` trong một venv chỉ có gói chạy, trên máy chủ (§4); commit vào repo, tạo ở T02, tạo lại mỗi khi đổi phụ thuộc chạy. Công cụ dev không có trong file này, nên nâng chúng không làm kết quả thành stale. Cài đặt dùng `-r requirements.lock` và `-c requirements.lock` | `pyproject.toml`: không thêm file, nhưng mọi sửa cấu hình ruff/pytest/coverage sẽ làm toàn bộ kết quả thành stale (Tier 1 + 2 khoảng 92 giờ-lõi); hoặc `pip freeze` của venv dev: cùng nhược điểm khi nâng công cụ dev | ADR-008, T02, T12 |
| QS-2 | Nếu E0a chọn η ≠ 0.05, DFedProx giữ gì? | Giữ cường độ ημτ ∈ {0.1, 1} ở τ = 20, tức μ = {0.1, 1} × 0.05/η; ghi vào ADR-007 cùng η | Giữ μ ∈ {0.1, 1} như ADR-006 (khi η = 0.01, μ = 0.1 gần như không có tác dụng — đúng lý do đã loại μ = 0.01) | ADR-006/007, E3a, E3b, H5 |
| QS-3 | Ngưỡng sanity có số | **Đã chốt (P1.8, 2026-09-30):** SAN-2 ≤ 1.0 điểm %; SAN-3 ≥ 20 điểm %; SAN-4 ≤ 1.0 điểm %. **Chốt thêm (P1.9, 2026-09-30):** đối chứng của SAN-2 là B1-4k (cùng η chung, 4,000 bước), không phải B1 | — | §9.3, `CONSTRAINTS.md` (R7), T14 |
| QS-4 | Độ đo suy ra đặt ở đâu? | `analysis/convergence.py`; bỏ `src/dfl/metrics/convergence.py` và `communication.py` | Giữ cây KH §4.6 (mỗi lần sửa code R@x/AUC làm toàn bộ kết quả thành stale) | §3, §5, T15 |
| QS-5 | Coverage có tính `analysis/` không? | **Đã chốt (P1.8, 2026-09-30):** có, ≥ 80% cho cả `src/dfl` (trừ `cli.py`) và `analysis/`; đã thêm `pythonpath = ["."]` vào cấu hình pytest | — | `CONSTRAINTS.md`, `pyproject.toml` |
| QS-6 | Ranh giới module | `contracts`/`config`/`seeding` là nền tảng dùng chung, không có id; `cli`, `__main__`, sweep, Observer thuộc `sim`; bốn module lá không import lẫn nhau | Thêm một id riêng (ví dụ `core`) | §3, P1.7 |
| QS-7 | CLI bổ sung | `describe`, `download`, `sweep --dry-run`, `analysis.aggregate`, `--out` (chỉ `run --out` được chạy khi cây bẩn, manifest ghi `dirty: true`; `sweep` luôn đòi cây sạch) | Bỏ bớt (ví dụ vẽ P7 bằng một script riêng) | §4, T05, T11, T12 |
| QS-8 | Demo trên laptop | 2 lượt ring vs FC, α = 0.1, N = 20, ngân sách khoảng 400 bước, MNIST tải sẵn, tổng ≤ 5 phút; chốt thông số ở T13 sau khi đo trên laptop thật. Laptop chạy hệ điều hành gì? (Makefile không chạy trên Windows; demo chỉ cần lệnh `python -m`) | Chỉ trình chiếu hình có sẵn, không chạy trực tiếp | §5.2, ENG-11, T11, T13 |
| QS-9 | Tải công việc của TV2: T06–T11, cả đường găng tới M3, đang dồn cho một người | **Đã chốt (P2.1, 2026-09-30):** TV1 nhận T09a, T09b; TV3 nhận nền tảng (T02a–c) và io (T11a); TV2 giữ T06–T08, T10a–c, T11b, T11c, T13, T22 | Giữ phân vai QĐ-1 | §3.1, §10 dòng 1, `tasks/plan.md` |
| QS-10 | Tới hết tuần 5 GVHD vẫn chưa trả lời Q1 thì làm gì với T22? | **Đã chốt (2026-09-30):** vẫn chạy lưới bằng mô phỏng đơn tiến trình. T22 làm ngay sau T13 (TV2, tuần 8–10), không còn là tùy chọn, để có bản đa tiến trình khi bảo vệ. T22 viết ở cây làm việc hằng ngày; lưới chạy từ worktree riêng nên không bị ảnh hưởng (§5.3) | Giữ T22 là mở rộng tùy chọn, chấp nhận rủi ro REQ-01 bị hiểu khác | ASSUMPTIONS 1, ADR-002, §1.4 |
| QS-11 | Có cài sẵn P2' (shards của McMahan) trước E0a không? | **Đã chốt (2026-09-30):** có, cài ở T04b, chỉ dùng trong `describe`, không chạy trong lưới trừ khi GVHD hay nhóm yêu cầu. Code phải có trước E0a thì mới thêm lượt sau được mà không đổi `code_version` | Không cài; nếu GVHD muốn shards sau E0a thì phải đổi `code_version` và chạy lại | §1.4, REQ-02, T04b |

Sau khi SPEC được duyệt: cập nhật KH theo §10.
