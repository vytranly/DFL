# SPEC — Đề 2: Non-IID × Decentralized Federated Learning

> **Trạng thái:** Draft 0.2, chờ nhóm duyệt (Gate G1) · **Ngày:** 2026-09-30 · 0.2 (P1.8): chốt QS-3 và QS-5; kiểm tra kiến trúc bằng import-linter; thêm `CONSTRAINTS.md` và `Makefile`
> **Nguồn:** `docs/KE_HOACH_DE_TAI_2.md` v0.2 (viết tắt **KH**) · `docs/HANDOFF_GUIDELINE.md` (**H**) · `docs/decisions/OPEN_QUESTIONS.md` (**OQ**) · `CLAUDE.md`.
> **Thứ tự ưu tiên khi mâu thuẫn:** ADR đã Accepted và `CONSTRAINTS.md` > SPEC > KH > H. Gặp mâu thuẫn chưa được giải quyết trong SPEC thì dừng và hỏi (H§3.0), không tự chọn một cách hiểu.
> Chỗ ghi **(QS-k)** đang dùng giá trị mặc định, chờ nhóm quyết ở §11.2.

## ASSUMPTIONS

```text
ASSUMPTIONS (giả định khi viết SPEC — sửa ngay nếu sai; nếu không, các task sẽ làm theo):
1. "Hệ thống P2P" = mô phỏng đơn tiến trình, đồng bộ; node chỉ trao đổi model qua Transport có đếm byte,
   không có server (A1, ADR-002). Chờ GVHD trả lời Q1. Nếu GVHD yêu cầu P2P thật, T22 (4–8 tiến trình,
   torch.distributed/gloo) chuyển thành bắt buộc; giao diện Transport giữ nguyên.
2. Nhóm 3 người, 12 tuần (QĐ-1). Mọi thí nghiệm chạy trên máy chủ CPU hiện tại (8 nhân, không GPU);
   laptop cá nhân chỉ dùng để demo (QĐ-3). Không viết nhánh code cho GPU (QĐ-5).
3. Mọi node cùng kiến trúc model và cùng w⁰ theo seed chung (A3, ghi là hạn chế trong báo cáo);
   mọi node tham gia mọi round, đồng bộ, không mô phỏng rớt node (A4).
4. Đánh giá trên tập test toàn cục chung, kèm acc^loc theo phân bố nhãn của từng node (A5).
   Macro-F1 tính cho từng node trên test chung; báo cáo mean, std và node kém nhất (Q3).
5. Phương pháp (cách chia dữ liệu, topology, ma trận trộn, thuật toán, độ đo, lưới, thống kê) lấy nguyên
   KH §3 v0.2. SPEC dẫn chiếu, không chép lại công thức; các điểm SPEC chuẩn hóa lại nằm ở §10.
6. Phạm vi đúng bảng KH §2.4 (xem §1.4). Phần mở rộng chỉ làm sau mốc M4, theo thứ tự QĐ-4.
7. η chưa chốt: chọn ở E0a theo quy tắc KH §3.5 (dự kiến 0.05) và ghi vào ADR-007 trước khi chạy E0b.
8. Báo cáo viết tiếng Việt; mọi hình sinh tự động từ log (A6). Hạn nộp và định dạng báo cáo chờ GVHD (Q2).
→ Sửa ngay nếu có điểm sai. Các điểm SPEC cần nhóm quyết: §11.2 (QS-1…QS-8).
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
- **US-2 Một nhóm thí nghiệm:** `python -m dfl sweep` chạy song song; bị ngắt thì chạy lại đúng lệnh đó, chỉ các `run_id` chưa có attempt `ok` được chạy.
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
| MNIST; IID, Dirichlet, #C = k, quantity skew | `TorchDistTransport` (gloo, 4–8 tiến trình); thành bắt buộc nếu Q1 yêu cầu | Nén / lượng tử hóa model |
| Ring, Erdős–Rényi liên thông, random pairwise gossip, fully-connected; star chỉ làm tham chiếu | Random k-regular, đồ thị mũ (T5); vị trí dữ liệu theo độ trung tâm (E7); gossip bất đồng bộ | Mạng thật nhiều máy, đo độ trễ mạng; GPU / Colab |
| D-PSGD, DFedAvg, DFedProx; quét τ; baseline tập trung, chỉ cục bộ, FedAvg có server | | Giao diện web / dashboard |
| Accuracy (toàn cục và theo phân bố cục bộ), macro-F1, bước/byte để đạt ngưỡng, độ ổn định, drift, consensus distance, phương sai giữa node | | Model khác nhau giữa các node |

Hai cách chia tùy chọn trong KH §3.2 — P2' (shards) và P4 ($p_{major}$) — không bắt buộc; chỉ làm khi nhóm yêu cầu.

---

## 2. Tech Stack

| Thành phần | Phiên bản (ghim trong `pyproject.toml`) | Ghi chú |
|---|---|---|
| Python | 3.12 (`>=3.12,<3.13`) | |
| torch / torchvision | 2.14.0 / 0.29.0, bản CPU | Cài từ `https://download.pytorch.org/whl/cpu` trước `pip install -e ".[dev]"` |
| networkx | 3.7 | Topology, spectral gap |
| numpy | 2.5.3 | `Generator` (PCG64), `SeedSequence` |
| pandas · matplotlib · pyyaml | 3.0.6 · 3.11.2 · 6.0.3 | Phân tích · hình · cấu hình (chỉ `yaml.safe_load`) |
| pytest · pytest-cov · ruff | 9.1.1 · 7.1.0 · 0.16.9 | Dev. ruff: dòng ≤ 110, py312, lint E, F, W, I, B, UP cộng các luật sàn của `CONSTRAINTS.md`; loại `docs/` |
| import-linter | 2.15 | Dev, kiểm ranh giới import theo §3.2 (`lint-imports`) |
| scikit-learn | 1.9.1 | Dev, chỉ làm oracle cho macro-F1 trong test |
| setuptools | ≥ 80 | Build backend, layout `src/` |

- **Máy chạy thí nghiệm:** máy chủ Linux, Intel Xeon Platinum 8176, 8 nhân, không GPU. **Laptop demo:** chưa biết hệ điều hành (QS-8).
- Không thêm hay nâng phụ thuộc khi chưa hỏi (§8). API framework dùng trong code phải đối chiếu tài liệu chính thức đúng phiên bản trên (P1.7).

---

## 3. Capability map

### 3.1 Module (id ổn định, không đổi tên trong suốt dự án)

| Module id | Mã nguồn | Trách nhiệm | Không được | Phụ trách (QĐ-1) | Task (KH §8) |
|---|---|---|---|---|---|
| `data` | `src/dfl/data/{datasets,partition,heterogeneity}.py` | Nạp MNIST bằng torchvision thành tensor trong RAM, chuẩn hóa, tách validation 5k phân tầng; P0–P3 đúng thuật toán NIID-Bench (min_size = 64); hoán vị vị trí node theo seed; JS, TV, số lớp ≥ 5%, CV(n), lệch mục tiêu, ma trận node × lớp | `partition`/`heterogeneity` import torch; biết topology hay model | TV1 | T03–T05 |
| `network` | `src/dfl/network/{topology,mixing,transport}.py` | T0–T4, T3b (T5 tùy chọn); `graph_at(round)` là hàm thuần của (seed, round); ER lấy mẫu lại tới khi liên thông; MH / uniform / size-weighted; kiểm tra ngẫu nhiên hàng/kép; spectral gap; trọng số đồng thuận π; Transport in-memory + CommLedger, nơi duy nhất đếm message và byte (tổng và theo từng node) | Import torch; tổng hợp model; copy payload theo số láng giềng; message node tự gửi cho mình | TV2 | T06–T08 |
| `learning` | `src/dfl/learning/{models,local_train}.py` | SmallCNN 46,730 tham số (không BatchNorm, không dropout); τ bước SGD (momentum 0, weight decay 0), minibatch B = 32 lấy có hoàn lại, số hạng prox μ với anchor = $w_i^t$ (clone + detach), hệ số mất mát $c_i$; ngẫu nhiên qua `torch.Generator` riêng từng node | Truy cập node khác, Transport hay topology | TV2 | T09 |
| `metrics-io` | `src/dfl/metrics/{classification,consensus}.py`, `src/dfl/io/{run_logger,manifest}.py` | Độ đo trong lúc chạy: $\mathrm{Acc}_i$, $\mathrm{Acc}^{loc}_i$, $\mathrm{Acc}^{eff}$, macro-F1 (từ ma trận nhầm lẫn), drift trước tổng hợp, consensus distance sau tổng hợp; ghi manifest/rounds/timing/summary theo attempt; tính `code_version`, `run_id` | Tự đếm byte (INV-2); tiêu thụ RNG của huấn luyện (INV-5); ghi đè attempt cũ; ghi thời gian vào `rounds.jsonl` (INV-6) | TV2 | T09–T11 |
| `sim` | `src/dfl/sim/{node,engine,observer,sweep}.py`, `src/dfl/cli.py`, `src/dfl/__main__.py` | Trạng thái node; vòng ATC; Observer theo lịch đánh giá (trước và sau tổng hợp); baseline B1–B3; lắp ráp một lượt từ cấu hình; sweep idempotent; CLI `run`, `sweep`, `describe`, `download` | Đưa đại lượng toàn cục vào huấn luyện (INV-1); đọc model của node khác ngoài inbox | TV2 (T10–T11), TV3 (T12) | T10–T12 |
| `analysis` | `analysis/{aggregate,convergence,stats,plots}.py` (ngoài `src/`, nên sửa không làm đổi `code_version`) | Gom các lượt thành DataFrame/CSV; độ đo suy ra từ `rounds.jsonl` (R@x, AUC, độ dao động và mức sụt theo từng node); CI (phân phối t), kiểm định H2; P1–P9, bảng T1; từ chối trộn `code_version` và so sánh lệch biến cố định (INT-2) | Import `sim`, `learning`, `data`, `network`; sửa hay lọc số liệu | TV3 (T15–T16); TV1 phân tích E1/E1c/E4 | T15–T16 |

**Nền tảng dùng chung** (không phải module id, QS-6): `src/dfl/contracts.py` (Protocol / TypedDict / dataclass, không có logic — P1.7), `config.py` (dataclass + `yaml.safe_load` + kiểm tra cấu hình), `seeding.py` (`SeedSequence` → seed con theo tên luồng). Nền tảng không import module nào ở bảng trên.

### 3.2 Hướng phụ thuộc (import một chiều, không có vòng)

```text
analysis ──► metrics-io (chỉ dfl.io, để đọc kết quả) ──┐
                                                       ▼
sim ──► data · network · learning · metrics-io ──► nền tảng (contracts · config · seeding)
```

- Bốn module lá `data`, `network`, `learning`, `metrics-io` chỉ import nền tảng, không import lẫn nhau. `sim` là nơi duy nhất ghép chúng; dữ liệu đi giữa các module lá dưới dạng mảng/tensor do `sim` truyền vào.
- `src/dfl` không bao giờ import `analysis`. `analysis` chỉ đọc file trong `results/` (hoặc `build/`) và chỉ được import `dfl.io`, `dfl.contracts`.
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

---

## 4. Commands

Chạy từ gốc repo, trong `.venv`. Chú thích ở mỗi nhóm lệnh cho biết lệnh có từ task nào.

```bash
# Cài đặt (CPU) — dùng được ngay
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Lint và test trực tiếp — dùng được ngay (trừ dòng cuối)
ruff check . && ruff format --check .
pytest -q tests/unit
pytest -q tests/integration tests/e2e      # từ T10–T11; hiện chưa có test nên pytest trả mã 5

# Kiểm tra theo chi phí — Makefile (P1.8); nội dung và lý do ở CONSTRAINTS.md
make check-fast   # sàn + ruff + import-linter + unit, vài giây
make check-task   # như trên + integration + coverage ≥ 80%; dưới 90 s (chỉ cảnh báo); trước mỗi commit
make check-full   # như trên + mọi test (gồm smoke e2e từ T11) + pip check; ở mỗi checkpoint
make check-inv    # T-INV-1…12 có đủ và xanh; bắt buộc từ C3 và trước mọi lượt Tier 1

# Dữ liệu — T03: tải MNIST vào data/ một lần (cần mạng; laptop demo phải tải trước)
python -m dfl download mnist

# Một lượt — T11
python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke   # dữ liệu tổng hợp, dưới 60 s
python -m dfl run --config <config.yaml>          # ghi results/<exp>/<run_id>/attempt-<k>/; cây phải sạch

# Mô tả dữ liệu và topology, không huấn luyện — T05 (mốc M2)
python -m dfl describe --config configs/base.yaml --seeds 20 --out build/describe

# Lưới thí nghiệm — T12
python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --dry-run    # in tổng / đã ok / sẽ chạy
python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --workers 4

# Phân tích và hình — T15–T16
python -m analysis.aggregate --results results/ --out build/analysis   # CSV một dòng mỗi lượt, kèm độ đo suy ra
python -m analysis.plots --results results/ --out figures/              # P1–P9 và bảng T1

# Demo trên laptop — QĐ-3, QS-8
python -m dfl sweep --experiment configs/experiments/demo.yaml --out build/demo --workers 2
python -m analysis.plots --results build/demo --out build/demo/figures
```

Các lệnh `run`, `sweep`, `analysis.plots` đã có trong CLAUDE.md. `download`, `describe`, `sweep --dry-run`, `--out` và `analysis.aggregate` là đề xuất (QS-7); cờ chi tiết chốt ở P1.7 và T11–T12.

---

## 5. Project Structure

### 5.1 Cây thư mục

```text
non_iid-dfl/
├── CLAUDE.md  SPEC.md  README.md  pyproject.toml     # đã có
├── CONSTRAINTS.md  Makefile                          # có từ P1.8
├── requirements.lock                                 # QS-1: pip freeze trên máy chủ, dùng cho code_version
├── configs/
│   ├── base.yaml                                     # mặc định KH §3.5
│   ├── profiles/{smoke,fast,full,demo}.yaml
│   └── experiments/{e0a_lr,e0b_sanity,e1_noniid,e1c_objective,e2_topology,e2b_gossip,
│                    e3a_algorithms,e3b_tau_mu,e4_quantity,demo}.yaml
├── src/dfl/                                          # mọi code ảnh hưởng tới kết quả; hash nội dung → code_version
│   ├── __main__.py  cli.py                           # sim (cli.py loại khỏi coverage)
│   ├── contracts.py  config.py  seeding.py           # nền tảng dùng chung
│   ├── data/{datasets,partition,heterogeneity}.py    # data
│   ├── network/{topology,mixing,transport}.py        # network
│   ├── learning/{models,local_train}.py              # learning
│   ├── metrics/{classification,consensus}.py         # metrics-io
│   ├── io/{run_logger,manifest}.py                   # metrics-io
│   └── sim/{node,engine,observer,sweep}.py           # sim
├── analysis/{aggregate,convergence,stats,plots}.py   # analysis (ngoài src/: sửa không làm đổi code_version)
├── tests/
│   ├── unit/  integration/  e2e/                     # file test_<chủ đề>.py, tên không trùng giữa các thư mục
│   └── reference/                                    # hàm chia dữ liệu tham chiếu (sao từ docs/spikes/) cho test golden
├── data/        # .gitignore — MNIST tải bằng torchvision
├── results/     # .gitignore — results/<exp>/<run_id>/attempt-<k>/
├── build/       # .gitignore — smoke, demo, describe, coverage
├── figures/     # chỉ sinh bằng script; có commit
├── docs/        # đề tài, research/, KH, H, decisions/ (ADR, OPEN_QUESTIONS), experiments/E<k>.md, spikes/, report/
└── tasks/{plan.md,todo.md,SESSION_LOG.md}
```

### 5.2 Cấu hình và profile

- `configs/base.yaml` chứa mặc định KH §3.5: N = 20, τ·R = 4,000, B = 32, SmallCNN, Metropolis–Hastings, min_size = 64, lịch đánh giá theo bước cục bộ, tập con test phân tầng 2,000 ảnh. Profile ghi đè ngân sách và quy mô; mỗi file trong `configs/experiments/` khai báo lưới của một nhóm thí nghiệm và profile dùng.
- Mỗi nhóm E0a, E0b, E1, E1c, E2, E2b, E3a, E3b, E4 có một file YAML riêng (E0a phải xong trước E0b vì η; E3a thuộc Tier 1, E3b thuộc Tier 2), cộng `demo.yaml`.

| Profile | N | τ·R | Dữ liệu | Dùng cho |
|---|---|---|---|---|
| `smoke` | 4 | 12 (τ = 2, R = 6) | Tổng hợp, cùng shape MNIST (2,000 train, 500 test); chỉ IID/Dirichlet (#C = k cần N ≥ 10, G23) | Test và `make check-full`; dưới 60 s |
| `fast` | 20 | 1,000 (τ = 10) | MNIST | Phát triển |
| `full` | 20 | 4,000 (mặc định τ = 20, R = 200) | MNIST | Kết quả cho báo cáo |
| `demo` | 20 | Nhỏ, dự kiến khoảng 400 | MNIST đã tải sẵn | Demo trên laptop: 2 lượt ring vs FC ở α = 0.1, tổng ≤ 5 phút (QS-8) |

### 5.3 Kết quả

- **Đường dẫn:** `results/<exp>/<run_id>/attempt-<k>/{manifest.json, rounds.jsonl, timing.jsonl, summary.json}` (KH §4.7).
- **`code_version`** = hash nội dung: tree hash của `src/` (`git rev-parse HEAD:src`) cùng hash của file khóa phụ thuộc (QS-1). Commit chỉ đổi `docs/`, `tasks/`, `analysis/` hay `configs/` không làm đổi nó (cấu hình đi vào `run_id`). **`run_id`** = hash của (cấu hình chuẩn hóa + `code_version`).
- **Ghi vào `results/`** chỉ khi `src/`, `configs/` và file khóa không có thay đổi chưa commit. Chạy với `--out` ra ngoài `results/` (ví dụ `build/`) thì được phép khi cây bẩn, và manifest ghi `dirty: true` (QS-7).
- **Attempt:** sweep bỏ qua `run_id` đã có attempt `ok`; attempt lỗi được giữ (`status: failed`) và chạy lại thành attempt mới; khi `code_version` đổi, các lượt cũ thành "stale": không xóa, không dùng lại.
- **`manifest.json`:** `run_id`, `code_version`, `dirty`, cấu hình chuẩn hóa đầy đủ (gồm kích thước và seed của tập đánh giá), seed và các seed con, phiên bản python/torch/torchvision/networkx/numpy, máy (CPU), số luồng CPU, thời điểm; spectral gap (topology tĩnh); trọng số đồng thuận π và lệch mục tiêu; ma trận node × lớp (N × 10) để P7 sinh được từ `results/`.
- **`rounds.jsonl`** (tất định, một dòng cho mỗi lần đánh giá): `round`, `local_steps`, `acc_mean`, `acc_std`, `acc_min`, `acc_max`, `acc_local_mean`, `acc_local_std`, `acc_local_min`, `f1_macro_mean`, `f1_macro_std`, `f1_macro_min`, `acc_avg_model`, `acc_eff_mean`, `drift_pre_combine`, `consensus_distance`, `messages_cum`, `bytes_cum`, `max_node_bytes_cum`, `train_loss_mean`; bắt buộc có `acc_per_node`, `acc_local_per_node` (mảng N phần tử).
- **`timing.jsonl`:** `round`, `wall_time_s` và thời gian từng pha (train, combine, eval).
- **`summary.json`:** `status` ∈ {`ok`, `failed`} (kèm thông báo lỗi); `final_*_full_test` trên đủ 10,000 ảnh test, tối thiểu gồm $\mathrm{Acc}_i$, $\mathrm{Acc}^{loc}_i$ và macro-F1 của từng node (mảng N) cùng acc($\bar w$); tổng message và byte. Không chứa độ đo suy ra (R@x, AUC, độ dao động, mức sụt): `analysis/` tính chúng từ `rounds.jsonl`.

---

## 6. Code Style

- **Ngôn ngữ:** tên định danh tiếng Anh; docstring và comment tiếng Việt, ngắn, nói *vì sao* và quy tắc đến từ đâu (KH §, ADR, G); không comment lặp lại code; không để code bị comment-out.
- **Hình dạng code:** ưu tiên hàm thuần và `@dataclass(frozen=True)` cho cấu hình/bản ghi (cấu hình không đổi được sau khi đã tính `run_id`); type hint đầy đủ cho tham số và giá trị trả về; không có trạng thái toàn cục thay đổi được.
- **Ngẫu nhiên tường minh:** nhận `numpy.random.Generator` / `torch.Generator` làm tham số; trong `src/` không dùng `np.random.<hàm>` toàn cục, `random.*` hay `torch.manual_seed`. Seed con sinh từ `SeedSequence` theo tên luồng cố định (partition, placement, graph, init, batch của từng node, gossip), không phụ thuộc topology, thuật toán hay τ, để cùng seed thì cùng partition, cùng hoán vị vị trí và cùng w⁰ (G9).
- **Module thuần** không import torch (§3.2); mọi `__init__.py` để trống.
- **Mô hình và tổng hợp:** không BatchNorm (G3); mỗi node tạo đúng một snapshot sở hữu riêng mỗi round (clone + detach, không dùng view của `state_dict()`, G22); tổng hợp cộng theo id người gửi tăng dần; chỉ ghi vào model sau khi mọi node đã tổng hợp xong (INV-3).
- **Lỗi cấu hình:** raise `ValueError` nói rõ ràng buộc bị vi phạm (ví dụ #C = k cần N ≥ 10, G23); không nuốt lỗi; không dùng `assert` để kiểm tra đầu vào (`python -O` bỏ qua `assert`).
- **An toàn:** chỉ `yaml.safe_load`; `torch.load` chỉ với `weights_only=True` (mặc định ở 2.14, G14); không unpickle dữ liệu từ ngoài; dữ liệu chỉ tải qua torchvision.
- **Định dạng:** `ruff format` + `ruff check`; không thêm `# noqa` hay `type: ignore` mà không ghi lý do.
- **Kích thước thay đổi:** mỗi task tối đa khoảng 300 dòng diff, một commit (KH §8).
- **API framework:** docstring ghi URL tài liệu chính thức đúng phiên bản; không kiểm được thì ghi "UNVERIFIED" (P1.7).

Mẫu phong cách — hàm `metropolis_hastings` có thật trong `docs/spikes/spike_topology_partition.py:33`, viết lại theo quy ước trên (docstring nói *vì sao*, kiểm tra đầu vào theo INV-4 và G7). Đã chạy thử trên một bản sao của repo (cùng `pyproject.toml`), đặt đúng đường dẫn `src/dfl/network/mixing.py` và `tests/unit/test_mixing.py`: `ruff check` và `ruff format --check` sạch cho cả mẫu này và mẫu test ở §7, test pass; hàm cho cùng ma trận với bản spike trên ring, ER (p = 0.3, seed 0) và FC. Khi làm T07 vẫn phải viết test trước.

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
| E2E (`tests/e2e/`) | ~5% | Dưới 60 s | `python -m dfl run` với profile `smoke` sinh đủ 4 file đúng schema §5.3; `analysis.plots` sinh được hình từ kết quả smoke |

- **Quy ước:** Arrange-Act-Assert; tên test mô tả hành vi; mỗi test một hành vi; tên file test không trùng giữa các thư mục (pytest import theo tên file); seed cố định; dung sai số thực ghi rõ; so khớp "trùng từng bit" dùng `torch.equal` / `np.array_equal`; không tải dữ liệu qua mạng (dataset giả cùng shape); không phụ thuộc thời gian thực.
- **Oracle và tham chiếu:** `tests/reference/` chứa hàm chia NIID-Bench sao từ `docs/spikes/`, tham số hóa theo mảng nhãn (golden cho T-INV-8); FedAvg có server viết ngay trong test (T-INV-3); `sklearn.metrics.f1_score(average="macro")` (T-INV-10).
- **TDD:** test FAIL trước, PASS sau. T-INV-1…4, 11, 12 phải có trước khi viết engine (H§7).
- **Tên test bất biến:** `test_t_inv_<k>_<hành vi>`, ví dụ `test_t_inv_4_node_order_does_not_change_result`, để `make check-inv` kiểm được đủ 12 ID.
- **Kiểm tra thống kê:** chỉ T-INV-1 (phần gossip) được so theo sai số lấy mẫu: hai phía, ±4 sai số chuẩn, t ≤ 3, seed cố định để không flaky.
- **Coverage:** ≥ 80% dòng cho `src/dfl` (trừ `cli.py`) và cho `analysis/` (nhóm chốt QS-5 ở P1.8); ngưỡng ghi trong `CONSTRAINTS.md` và `fail_under` của `pyproject.toml`, không được hạ để qua kiểm tra.
- **Kiểm tra kiến trúc:** import-linter (ENG-6, ENG-7, §9.4), cấu hình ở `[tool.importlinter]` trong `pyproject.toml`.

Mẫu test (dùng hàm ở §6):

```python
import networkx as nx
import numpy as np
import pytest

from dfl.network.mixing import metropolis_hastings


def test_metropolis_hastings_on_ring_is_doubly_stochastic_with_known_spectral_gap() -> None:
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
- Chỉ chạy `run`/`sweep` ghi vào `results/` khi cây làm việc sạch (`src/`, `configs/`, file khóa).
- (toàn vẹn) Commit thẻ tiền đăng ký `docs/experiments/E<k>.md` trước khi chạy nhóm thí nghiệm đó; điền kết quả sau khi chạy, kể cả kết quả trái giả thuyết.
- (toàn vẹn) Mỗi lượt có manifest đầy đủ (§5.3); giữ mọi attempt, kể cả attempt lỗi.
- (toàn vẹn) Chỉ chọn siêu tham số trên validation 5k; ghi η vào ADR-007 trước khi chạy E0b.
- (toàn vẹn) So sánh ghép cặp: cùng seed thì cùng partition, cùng hoán vị vị trí, cùng w⁰ cho mọi topology và thuật toán (G9); so D-PSGD với DFedAvg theo tổng số bước cục bộ và theo byte, không theo round (G10).
- (toàn vẹn) Báo cáo điểm của từng seed và "k/n seed đạt", không chỉ trung bình; báo cáo kích thước node và lệch mục tiêu kèm mọi kết quả Dirichlet (G11, G19); diễn giải FC kèm acc^loc và drift (G15).
- (toàn vẹn) Đạt SAN-1…3 trước khi chạy E1 trở đi (§9.3).

**Ask first**
- Thêm hoặc nâng phụ thuộc (kể cả dev).
- Sửa SPEC, ADR, `CONSTRAINTS.md`.
- Đổi siêu tham số hay ngưỡng đã chốt (KH §3.5, ADR-006/007, SAN-*): phải có ADR mới và chạy lại cả nhóm thí nghiệm bị ảnh hưởng.
- Di chuyển hoặc xóa bất cứ gì trong `results/`.
- Chạy lưới lớn hơn 10 lượt (báo kết quả `--dry-run` trước).
- Làm bất cứ mục nào ngoài cột "Trong phạm vi" ở §1.4, kể cả phần mở rộng trước M4, P2', P4 và T5.
- Cắt Tier 2, hoặc hạ thí nghiệm lõi từ 5 xuống 3 seed (KH §3.7; phải ghi rõ trong báo cáo).
- Đổi schema manifest/rounds/summary sau khi đã có kết quả Tier 1.

**Never**
- Sửa tay `results/` hoặc `figures/`.
- (toàn vẹn) Xóa lượt chạy lỗi; bỏ seed "xấu" hay chọn lọc lượt chạy để kết quả đẹp hơn.
- (toàn vẹn) Trộn kết quả của nhiều `code_version` trong một hình hay một so sánh.
- (toàn vẹn) Chọn siêu tham số trên tập test (G8, G25); sửa lr, τ hay μ sau khi xem kết quả mà không có ADR mới và chạy lại.
- Hạ ngưỡng trong `CONSTRAINTS.md` để qua kiểm tra; skip hay xóa test đang fail khi chưa được duyệt.
- Commit dữ liệu MNIST, `results/`, hay secret/token.
- `torch.load(..., weights_only=False)`; `yaml.load` không dùng SafeLoader.
- Đưa đại lượng toàn cục ($\bar w$, trung bình toàn mạng) vào huấn luyện (INV-1); đếm byte ngoài Transport (INV-2); cập nhật model tại chỗ khi tổng hợp (INV-3); để Observer sửa model hay tiêu thụ RNG của huấn luyện (INV-5); ghi thời gian vào `rounds.jsonl` (INV-6).
- Dùng BatchNorm (G3).
- Tự đánh dấu một gate là "đã duyệt" (H§7).

---

## 9. Success Criteria

Mỗi task còn phải đạt Definition of Done ở H§4.2.

### 9.1 Yêu cầu của đề (REQ-01…14, H§2.1)

| REQ | Tiêu chí chấp nhận (kiểm chứng được) | Kiểm chứng | Task |
|---|---|---|---|
| REQ-01 Mô phỏng DFL không server | Node chỉ đọc model của mình và inbox (INV-1); mọi message đi qua Transport, CommLedger là nơi duy nhất đếm (INV-2); lịch gossip do từng node tự tính từ (seed chung, round), không có điều phối viên (ADR-009) | T-INV-2, T-INV-4, T-INV-12; ENG-7 | T08, T10 |
| REQ-02 Non-IID nhiều mức | P0 IID; P1 Dirichlet α ∈ {100, 1, 0.5, 0.1, 0.05}; P2 #C = k, k ∈ {2, 1}; P3 quantity β = 0.5; đúng thuật toán NIID-Bench với min_size = 64. Trên MNIST thật (N = 20, 20 seed), JS trung bình của mỗi cách chia lệch không quá 0.02 so với bảng KH §3.2 | T-INV-8; `dfl describe`; hình P7 | T04, T05 |
| REQ-03 Accuracy / F1 | Ở mỗi lần đánh giá: $\mathrm{Acc}_i$, $\mathrm{Acc}^{loc}_i$, $\mathrm{Acc}^{eff}$, macro-F1 của từng node (mean, std, min), acc($\bar w$) trên tập con test 2,000 ảnh; đánh giá cuối trên 10,000 ảnh, lưu ở trường riêng | T-INV-10, T-INV-11; test schema e2e | T09–T11 |
| REQ-04 Tốc độ hội tụ | R@x (x ∈ {90, 95}; 97 chỉ là phụ) và AUC đúng định nghĩa KH §3.6: đạt bền vững qua 3 điểm, nội suy tuyến tính, quy ra round và byte, ghi "> ngân sách" khi không đạt, tổng hợp qua seed bằng trung vị kèm "k/n seed đạt" | Unit test trong `analysis`: chuỗi đạt, không đạt, chuỗi ngắn, đạt rồi tụt | T15 |
| REQ-05 Chi phí giao tiếp | Message/round, byte/round, byte tích lũy, tải lớn nhất trên một node, bytes@x; byte = message × 4 × 46,730 | T-INV-6; P6 | T08, T15 |
| REQ-06 Độ ổn định | Độ dao động và mức sụt tính trên từng node rồi mới lấy trung bình (G24); độ lệch giữa seed; drift trước tổng hợp; consensus distance sau tổng hợp | T-INV-1, T-INV-11; unit test `analysis`; P5 | T10, T15 |
| REQ-07 So sánh topology | T1 ring, T2 ER liên thông (p = 0.3), T3 gossip trên $K_N$, T4 FC; thêm T3b và T0 (tham chiếu); spectral gap của topology tĩnh ghi trong manifest | T-INV-9; unit test topology (liên thông, đúng bậc, matching hợp lệ, gọi lại không đổi kết quả); P3, P8 | T06, T07, T18 |
| REQ-08 Thuật toán tổng hợp | Một vòng ATC tham số (τ, μ, $c_i$) cho D-PSGD (τ = 1, R = 4,000), DFedAvg (τ = 20, R = 200), DFedProx (τ = 20, μ theo QS-2); τ·R = 4,000 ở mọi điều kiện | T-INV-3, T-INV-7; E3a; P9 | T09, T10, T19 |
| REQ-09 Accuracy theo round, volume, variance | `rounds.jsonl` đủ trường ở §5.3 (có `acc_per_node`, `acc_local_per_node`); phương sai giữa node trên $\mathrm{Acc}_i$ và $\mathrm{Acc}^{loc}_i$ ở round cuối (std, min, max − min, trung bình 10% node kém nhất) | Test schema e2e; T-INV-11; P1, P4, P6 | T11, T15 |
| REQ-10 Biểu đồ so sánh | P1–P9 và bảng T1 (KH §3.8) sinh bằng một lệnh; bảng màu an toàn cho người mù màu; trục có đơn vị | e2e vẽ hình trên kết quả smoke; INT-1, INT-2 | T16 |
| REQ-11 Kiến trúc tổng quan | Sơ đồ KH §4.1 và luồng một round §4.3 có trong báo cáo và khớp capability map §3 | Duyệt ở Gate G1 và khi review báo cáo | P1.5, T24 |
| REQ-12 Baseline IID, biến cố định | B1 tập trung, B2 chỉ cục bộ (W = I), B3 FedAvg có server (trọng số $n_k/n$); đạt SAN-1…4; biến cố định (N, model, η, B, τ·R, lịch đánh giá) trùng nhau trong mọi so sánh | §9.3; INT-2 | T11, T14 |
| REQ-13 Nhận xét và khuyến nghị | Mỗi khuyến nghị (topology, τ, chiến lược tổng hợp, trọng số trộn) dẫn chiếu ít nhất một hình/bảng và run ID; có mục hạn chế | INT-4, INT-5; P4.3 | T24 |
| REQ-14 Dataset | MNIST qua torchvision: 60k train chia thành 55k cho node và 5k validation phân tầng; 10k test. Fashion-MNIST và CIFAR-10 chỉ là mở rộng sau M4 | Unit test T03 trên dataset giả (shape, đếm lớp, phân tầng) | T03 (T21) |

### 9.2 Test bất biến T-INV-1…12 (KH §5.2)

Cả 12 test phải tồn tại và xanh trước khi chạy bất kỳ lượt nào của Tier 1.

| ID | Điều kiện đạt | Mức | Task |
|---|---|---|---|
| T-INV-1 | lr = 0, khởi tạo khác nhau. Topology tĩnh (ring, ER, FC): $\Xi_t\le\rho^t\,\Xi_0$ (sai số số học nhỏ) và trung bình model được bảo toàn với MH. Gossip: mỗi W đối xứng, ngẫu nhiên kép, $W^2=W$; trung bình của W qua $10^4$ matching khớp $\frac12 I+\frac{1}{2(N-1)}(J-I)$ (N chẵn) trong sai số lấy mẫu; nếu kiểm Monte Carlo cho $\Xi_t^2$ thì hai phía, ±4 sai số chuẩn, chỉ với t ≤ 3 | integration | T10 |
| T-INV-2 | W = I cho kết quả trùng khớp huấn luyện chỉ cục bộ | integration | T10 |
| T-INV-3 | FC + trọng số size-weighted tương đương FedAvg có server với trọng số $n_k/n$ (allclose); khi kích thước node bằng nhau, FC + MH cũng vậy | integration | T10 |
| T-INV-4 | Đảo thứ tự duyệt node không làm đổi kết quả, trùng từng bit | integration | T10 |
| T-INV-5 | Cùng seed + `code_version` thì `rounds.jsonl` giống hệt; khác seed thì khác cách chia | e2e (smoke) | T11 |
| T-INV-6 | N = 20: ring 40, FC 380, gossip trên $K_N$ 20 message/round; byte = message × 4 × số tham số | unit | T08 |
| T-INV-7 | μ = 0 tương đương SGD thường; gradient của số hạng prox bằng μ(w − anchor); anchor không có gradient | unit | T09 |
| T-INV-8 | Partition rời nhau, phủ đủ, mọi node ≥ min_size; #C = k: mỗi node đúng k nhãn, mọi nhãn có node sở hữu; α = 1000 thì JS < 0.01; JS (trung bình 3 seed) tăng đơn điệu khi α giảm; **golden:** cùng `Generator` thì trùng tuyệt đối với hàm tham chiếu trong `tests/reference/`; #C = k với N < 10 báo lỗi cấu hình | unit | T04 |
| T-INV-9 | Spectral gap của ring(20) = ⅔(1 − cos(2π/20)) ≈ 0.0326; của FC = 1 | unit | T07 |
| T-INV-10 | Macro-F1 khớp `sklearn.metrics.f1_score(average="macro")` trên nhãn ngẫu nhiên; $\mathrm{Acc}^{loc}_i$ và $\mathrm{Acc}^{eff}$ khớp accuracy có trọng số từng mẫu $P_i(y)/P_{test}(y)$ trên cùng tập test (sai số ≤ 1e-12) | unit | T09 |
| T-INV-11 | FC + MH: sau tổng hợp mọi node có model trùng nhau (allclose), nên std của $\mathrm{Acc}_i$ bằng 0 và $\mathrm{Acc}^{loc}_i$ của mọi node tính từ cùng một vector recall | integration | T10 |
| T-INV-12 | Tắt Observer, hoặc gọi `graph_at` thêm lần, không làm đổi quỹ đạo huấn luyện | integration | T10 |

### 9.3 Sanity, tái lập và toàn vẹn (KH §5.3)

Ngưỡng có số do nhóm chốt ở P1.8 (QS-3) và được ghi trong `CONSTRAINTS.md` (R7).

- **SAN-1** (E0b) B1 tập trung: accuracy trên đủ 10,000 ảnh test ≥ 98.5% (trung bình 3 seed).
- **SAN-2** (E0b) DFedAvg-FC-IID kém B1 không quá 1.0 điểm % (so trung bình qua seed).
- **SAN-3** (E0b) B2 chỉ cục bộ ở α = 0.1 thấp hơn B2 ở IID ít nhất 20 điểm % (accuracy trên test toàn cục).
- **SAN-4** (E1) Trên cùng topology, accuracy ở α = 100 và ở IID chênh nhau không quá 1.0 điểm %.
- Không đạt SAN-1…3 thì dừng và debug (skill `debugging-and-error-recovery`) trước khi chạy E1; không đạt SAN-4 thì dừng trước khi chạy E2.
- **Tín hiệu dừng** (không phải ngưỡng đạt/không đạt): accuracy tăng khi α giảm → nghi lỗi đảo chiều α (G1) trước khi tin; std giữa node bằng 0 trên FC là do cấu trúc (G15), không phải một phát hiện.
- **REP-1** Clone sạch → cài đặt theo §4 → chạy lại 2 `run_id` chọn ngẫu nhiên trên cùng máy chủ, cùng số luồng, cùng `code_version` → `rounds.jsonl` và `summary.json` trùng từng byte (`cmp`).
- **INT-1** Mọi hình và bảng trong `figures/` sinh bằng `python -m analysis.plots --results results/ --out figures/`; không sửa tay.
- **INT-2** `analysis` báo lỗi rõ ràng (có unit test) khi một hình hay một so sánh trộn nhiều `code_version` (trừ khi được cho phép tường minh, KH §4.7), hoặc khi các biến cố định (N, model, η, B, τ·R, lịch đánh giá, tập đánh giá) khác nhau giữa các lượt, trừ biến là trục của hình đó.
- **INT-3** Không attempt nào bị xóa; phụ lục báo cáo liệt kê số attempt `failed` và lý do.
- **INT-4** Mọi nhận định trong báo cáo dẫn chiếu hình/bảng và run ID; khuyến nghị có mục hạn chế (lr dùng chung cho τ ≠ 20, gossip trên $K_N$, lệch mục tiêu, khởi tạo chung…).
- **INT-5** Mỗi kết luận chính được một reviewer ngữ cảnh mới tìm cách bác bỏ (doubt-driven, KH §5.3) trước khi chốt báo cáo.

### 9.4 Kỹ thuật

- **ENG-1** `make check-task` xanh trước mỗi commit; `make check-full` xanh ở mỗi checkpoint C1–C5.
- **ENG-2** Coverage dòng ≥ 80% cho `src/dfl` (trừ `cli.py`) và `analysis/` (chốt ở P1.8).
- **ENG-3** `ruff check .` và `ruff format --check .` sạch.
- **ENG-4** `python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke` xong dưới 60 s trên máy chủ và sinh đủ 4 file đúng schema §5.3.
- **ENG-5** `make check-task` xong dưới 90 s.
- **ENG-6** Module thuần không nạp torch, kể cả gián tiếp: contract `pure-no-torch` của import-linter (`lint-imports`) cho `dfl.contracts` và `dfl.network`; `dfl.data.partition`, `dfl.data.heterogeneity` được thêm ở T04. Riêng `dfl.contracts` còn có test chạy trong tiến trình con (`tests/unit/test_contracts_smoke.py`).
- **ENG-7** Hướng phụ thuộc §3.2 được kiểm bằng các contract `layers`, `no-analysis-in-src`, `analysis-reads-only` của import-linter. Cấm RNG toàn cục, kiểm bằng ruff `NPY002`, `S311` và `make check-floor` (`torch.manual_seed` trong `src/`).
- **ENG-8** Unit và integration test không truy cập mạng và không cần MNIST thật.
- **ENG-9** Sweep: bỏ qua `run_id` đã có attempt `ok`; chạy lại được attempt lỗi; từ chối khi `src/`, `configs/` hay file khóa có thay đổi chưa commit; commit chỉ đổi `docs/`, `tasks/`, `analysis/` hay `configs/` không làm đổi `code_version`; `--dry-run` in đúng số lượt tổng / đã ok / sẽ chạy.
- **ENG-10** `run` và `sweep` chỉ ghi vào `results/` khi cây sạch; khi ghi ra `--out` ngoài `results/`, trường `dirty` trong manifest phản ánh đúng trạng thái cây.
- **ENG-11** Demo: `configs/experiments/demo.yaml` (2 lượt) và lệnh vẽ hình xong trong tổng cộng ≤ 5 phút trên laptop demo, không cần mạng (MNIST tải sẵn) (QS-8).

### 9.5 Hoàn thành thực nghiệm (KH §3.7)

- **EXP-1** Mỗi nhóm thí nghiệm có thẻ tiền đăng ký `docs/experiments/E<k>.md` được commit trước khi chạy.
- **EXP-2** Tier 1 — E0a 15, E0b 37, E1 70, E2 30, E3a 24 = **176** `run_id` — đều có attempt `ok`, cùng một `code_version`.
- **EXP-3** Tier 2 — E1c 15, E2b 9, E3b 42, E4 21 = **87** `run_id` (+12 lượt D-PSGD với η riêng nếu điều kiện ở KH §3.5 xảy ra) — đều có attempt `ok`; các ô dùng lại giữa các nhóm thí nghiệm có cùng `code_version`.
- **EXP-4** Mỗi giả thuyết H1–H7 có kết luận ủng hộ / không ủng hộ / chưa kết luận, kèm hình và run ID, ghi trong thẻ thí nghiệm và trong báo cáo.
- **EXP-5** Nếu cắt Tier 2 hay hạ số seed: có quyết định của nhóm (Ask first) và báo cáo ghi rõ.

---

## 10. Khác biệt so với KH v0.2

SPEC được ưu tiên hơn KH ở các điểm dưới đây. KH sẽ được cập nhật theo bảng này sau khi SPEC được duyệt.

| # | KH v0.2 | SPEC | Lý do |
|---|---|---|---|
| 1 | §7: phân vai cho 4 người | 3 người: TV1 `data` + phân tích; TV2 `network`, `learning`, `metrics-io`, engine; TV3 sweep, `analysis`, thí nghiệm, báo cáo | QĐ-1 |
| 2 | A2, §3.5, §3.7, §9: có nhánh GPU / Colab | Chỉ CPU: bỏ cờ tất định cho GPU, cấu hình đánh giá 10k trên GPU và ước tính thời gian GPU | QĐ-5, Q4 |
| 3 | §3.5: profile `smoke`, `fast`, `full` | Thêm profile `demo` và `configs/experiments/demo.yaml` | QĐ-3 |
| 4 | §4.7: chỉ có `f1_macro_mean` | Thêm `f1_macro_std`, `f1_macro_min`; macro-F1 từng node ở đánh giá cuối | Q3 |
| 5 | §4.6: `src/dfl/metrics/{…,convergence,communication}.py` | `convergence` chuyển thành `analysis/convergence.py`; bỏ `metrics/communication.py` (số liệu message/byte đọc từ CommLedger) | KH §4.7: sửa code phân tích không được làm đổi `code_version`; INV-2 (QS-4) |
| 6 | §4.6: chưa nêu chỗ đặt sweep và Observer | `sim/sweep.py`, `sim/observer.py` | `cli.py` bị loại khỏi coverage, nên logic sweep không được nằm ở đó |
| 7 | §4.6: `e0_sanity.yaml`, `e3_tau_mu.yaml` | Tách `e0a_lr` / `e0b_sanity` và `e3a_algorithms` / `e3b_tau_mu` | E0a phải xong trước E0b; E3a thuộc Tier 1, E3b thuộc Tier 2 |
| 8 | §4.7: chỉ sweep từ chối khi cây bẩn | `run` cũng vậy khi ghi vào `results/`; `--out` ra `build/` thì được, manifest ghi `dirty: true` | `code_version` lấy từ commit nên không phản ánh thay đổi chưa commit (QS-7) |
| 9 | §3.2: "lưu ma trận node × lớp" (chưa nói lưu ở đâu) | Lưu trong `manifest.json` | P7 sinh được từ `results/` |
| 10 | (chưa có) | CLI `describe`, `download`, `sweep --dry-run`, `analysis.aggregate` | M2 cần P7 và spectral gap trước khi có engine; demo không cần mạng; quy tắc "lưới > 10 lượt" (QS-7) |
| 11 | (chưa có) | Kiểm tra kiến trúc ENG-6, ENG-7 bằng import-linter (thêm `network/transport.py` vào nhóm không nạp torch, vì payload là đối tượng mờ); `analysis` kiểm tra biến cố định (INT-2); đối chiếu JS trên MNIST thật với bảng KH §3.2 (REQ-02) | Kiểm được hướng phụ thuộc, REQ-12 và cách chia trên nhãn thật |
| 12 | §3.4, ADR-006: μ ∈ {0.1, 1}, lập luận ở η = 0.05 | Nếu η chọn ở E0a khác 0.05 thì giữ cường độ ημτ ∈ {0.1, 1} ở τ = 20 (QS-2) | H5 và lý do loại μ = 0.01 đều dựa trên ημτ |

---

## 11. Open Questions

### 11.1 Chờ GVHD (OQ, tin nhắn đã soạn)

| ID | Câu hỏi | Ảnh hưởng |
|---|---|---|
| Q1 | Mô phỏng đơn tiến trình có message-passing có được coi là "hệ thống P2P" không? | §1.4 (T22 thành bắt buộc nếu không), ADR-002 |
| Q2 | Ngày nộp, ngày bảo vệ, định dạng báo cáo, có cần slide/demo không? | Lịch KH §7, T24, T25, QS-8 |

### 11.2 Cần nhóm quyết (SPEC đang dùng giá trị mặc định)

| ID | Câu hỏi | Mặc định đang dùng | Phương án khác | Ảnh hưởng |
|---|---|---|---|---|
| QS-1 | File nào là "file khóa phụ thuộc" trong `code_version`? | `requirements.lock` = `pip freeze --exclude-editable` trên máy chủ, commit vào repo, tạo lại mỗi khi đổi phụ thuộc | `pyproject.toml`: không thêm file, nhưng mọi sửa cấu hình ruff/pytest/coverage sẽ làm toàn bộ kết quả thành stale (Tier 1 + 2 khoảng 44–57 giờ-CPU) | ADR-008, T12 |
| QS-2 | Nếu E0a chọn η ≠ 0.05, DFedProx giữ gì? | Giữ cường độ ημτ ∈ {0.1, 1} ở τ = 20, tức μ = {0.1, 1} × 0.05/η; ghi vào ADR-007 cùng η | Giữ μ ∈ {0.1, 1} như ADR-006 (khi η = 0.01, μ = 0.1 gần như không có tác dụng — đúng lý do đã loại μ = 0.01) | ADR-006/007, E3a, E3b, H5 |
| QS-3 | Ngưỡng sanity có số | **Đã chốt (P1.8, 2026-09-30):** SAN-2 ≤ 1.0 điểm %; SAN-3 ≥ 20 điểm %; SAN-4 ≤ 1.0 điểm % | — | §9.3, `CONSTRAINTS.md` (R7), T14 |
| QS-4 | Độ đo suy ra đặt ở đâu? | `analysis/convergence.py`; bỏ `src/dfl/metrics/convergence.py` và `communication.py` | Giữ cây KH §4.6 (mỗi lần sửa code R@x/AUC làm toàn bộ kết quả thành stale) | §3, §5, T15 |
| QS-5 | Coverage có tính `analysis/` không? | **Đã chốt (P1.8, 2026-09-30):** có, ≥ 80% cho cả `src/dfl` (trừ `cli.py`) và `analysis/`; đã thêm `pythonpath = ["."]` vào cấu hình pytest | — | `CONSTRAINTS.md`, `pyproject.toml` |
| QS-6 | Ranh giới module | `contracts`/`config`/`seeding` là nền tảng dùng chung, không có id; `cli`, `__main__`, sweep, Observer thuộc `sim`; bốn module lá không import lẫn nhau | Thêm một id riêng (ví dụ `core`) | §3, P1.7 |
| QS-7 | CLI bổ sung | `describe`, `download`, `sweep --dry-run`, `analysis.aggregate`, `--out` (cho phép cây bẩn khi ghi ngoài `results/`, manifest ghi `dirty: true`) | Bỏ bớt (ví dụ vẽ P7 bằng một script riêng) | §4, T05, T11, T12 |
| QS-8 | Demo trên laptop | 2 lượt ring vs FC, α = 0.1, N = 20, ngân sách khoảng 400 bước, MNIST tải sẵn, tổng ≤ 5 phút; chốt thông số ở T13 sau khi đo trên laptop thật. Laptop chạy hệ điều hành gì? (Makefile không chạy trên Windows; demo chỉ cần lệnh `python -m`) | Chỉ trình chiếu hình có sẵn, không chạy trực tiếp | §5.2, ENG-11, T11, T13 |

Sau khi SPEC được duyệt: cập nhật KH theo §10, rồi làm P1.6 (ADR).
