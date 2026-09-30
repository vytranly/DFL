# Session log

Mỗi phiên thêm một mục theo mẫu docs/HANDOFF_GUIDELINE.md §5.2. Mục mới nhất ở cuối.

## 2026-09-30 — Phiên #1 — P1.1 / T01 Dựng repo
- Trạng thái: T01 ✅ (commit đầu tiên "chore: bootstrap repo, rules file and project skeleton")
- File đã đổi: .gitignore, pyproject.toml, CLAUDE.md, README.md, khung src/dfl, analysis, tests, configs, figures, docs/{decisions,experiments,report}, tasks/SESSION_LOG.md
- Kiểm chứng: xem báo cáo của phiên (pip install, import torch/torchvision/networkx, pytest, ruff)
- Quyết định mới: cài torch/torchvision bản CPU từ https://download.pytorch.org/whl/cpu trước khi `pip install -e ".[dev]"` (tránh bản CUDA nhiều GB); có một test smoke import để `pytest -q` thoát mã 0 khi chưa có test nào khác
- Câu hỏi mở: Q1–Q7 trong docs/KE_HOACH_DE_TAI_2.md §11 (chưa hỏi GVHD)
- Việc tiếp theo: P1.3 (chốt giả định/câu hỏi mở) hoặc P1.5 (SPEC.md)

## 2026-09-30 — Phiên #2 — P1.2 Tóm tắt tài liệu nghiên cứu
- Trạng thái: P1.2 ✅ — docs/research/DIGEST.md
- File đã đổi: docs/research/DIGEST.md, tasks/SESSION_LOG.md
- Kiểm chứng: 3 subagent Explore đọc 10 tài liệu (2 PDF có sẵn + 8 bản arXiv); phiên chính đối chiếu 46 trích dẫn/con số với văn bản gốc, tất cả khớp; mã NIID-Bench (utils.py) xác nhận bước balance, min_require_size, nhãn đầu i%K
- Quyết định mới: không có (DIGEST chỉ đề xuất, chưa sửa kế hoạch)
- Câu hỏi mở: có áp các đề xuất sửa §1.3 ở DIGEST §1 vào kế hoạch không; Hegedűs 2021 và ACM CSUR 2025 chưa đọc được toàn văn
- Việc tiếp theo: P1.3 (chốt giả định/câu hỏi mở) hoặc sửa §1.3 theo DIGEST

## 2026-09-30 — Phiên #3 — P1.3 Chốt giả định và câu hỏi mở
- Trạng thái: P1.3 ✅ — docs/decisions/OPEN_QUESTIONS.md (nhóm đã xác nhận "chốt")
- File đã đổi: docs/decisions/OPEN_QUESTIONS.md, tasks/SESSION_LOG.md
- Kiểm chứng: phỏng vấn từng câu theo skill interview-me (5 câu + bản tổng kết), nhóm xác nhận rõ ràng
- Quyết định mới: 3 người, 12 tuần, chỉ CPU (máy chủ hiện tại), laptop để demo; Q3/Q5/Q7 nhóm tự trả lời; QĐ-1…QĐ-5 trong OPEN_QUESTIONS.md
- Câu hỏi mở: Q1 (P2P thật?) và Q2 (ngày nộp/bảo vệ, định dạng báo cáo, slide/demo) — chờ GVHD; tin nhắn đã soạn sẵn
- Việc tiếp theo: gửi tin nhắn cho GVHD; P1.4 (sửa đath1.docx) hoặc P1.5 (SPEC.md)

## 2026-09-30 — Phiên #4 — P1.4 Sửa bản nháp đath1.docx
- Trạng thái: P1.4 ✅ — docs/đath1_revised.docx (tracked changes + 4 comment, tác giả "Claude"); file gốc docs/đath1.docx không đổi (sha256 f51ba177…, `git diff` rỗng)
- File đã đổi: docs/đath1_revised.docx (mới), tasks/SESSION_LOG.md
- Kiểm chứng: `validate.py docs/đath1_revised.docx --original docs/đath1.docx --author "Claude"` → All validations PASSED (186 → 243 đoạn); render PDF bản tracked (18 trang) và bản đã accept, xem lại từng trang đã sửa; bản accept: đánh số mục 1–8 đúng, không còn "ፀ", không còn câu α sai chiều, "server" chỉ còn ở ngữ cảnh đúng, đủ dấu vết D1–D17; số liệu chèn khớp kế hoạch §3.2–§3.7. Chưa có Makefile nên không chạy `make check-task` (thay đổi chỉ gồm tài liệu)
- Quyết định mới: không (nội dung chèn lấy từ kế hoạch §1.1, §3.2–§3.7, §4.1–§4.3, ADR-001/003/004/006); các chỗ phụ thuộc lựa chọn của nhóm để ở dạng comment, không sửa
- Câu hỏi mở: nhóm tự quyết — gộp khối độ đo lặp (D16), thứ tự ATC trong "Workflow tại bất kỳ node", heading "Phương pháp Pathological partition" rỗng, bỏ hay giữ link blog, ô trống trong công thức Σθᵢ = 1 (có từ bản gốc), giữ mô tả shards hay chỉ #C=k, hình kiến trúc (matplotlib hay xuất từ Mermaid §4.1), dấu thập phân
- Việc tiếp theo: nhóm mở bản sửa trong Word, Accept/Reject từng thay đổi và trả lời comment; P1.5 (SPEC.md); Q1–Q2 vẫn chờ GVHD

## 2026-09-30 — Phiên #5 — P1.5 SPEC.md
- Trạng thái: P1.5 ⏳ chờ nhóm duyệt — SPEC.md Draft 0.1. Được commit ở phiên #6 dưới dạng bản nháp vì ADR dẫn chiếu SPEC; chưa được duyệt, QS-1…QS-8 vẫn mở
- File đã đổi: SPEC.md (mới), tasks/SESSION_LOG.md
- Kiểm chứng: 2 đoạn code mẫu trong SPEC (§6, §7) đặt đúng đường dẫn trên một bản sao repo: `ruff check`/`ruff format --check` sạch, test pass, cho cùng ma trận MH với hàm trong spike trên ring/ER/FC; 12 bảng Markdown đúng số cột; đủ ID REQ-01…14, T-INV-1…12, SAN-1…4, INT-1…5, ENG-1…11, EXP-1…5, QS-1…8; `rounds.jsonl` giữ đủ 20 trường của KH §4.7, thêm 2 trường theo Q3; `ruff check .`, `ruff format --check .`, `pytest -q` (7 passed) vẫn xanh
- Quyết định mới: không. SPEC §10 ghi các điểm chuẩn hóa khác KH v0.2; SPEC §11.2 ghi QS-1…QS-8 đang dùng giá trị mặc định
- Ghi chú công cụ: `ruff check --stdin-filename …` báo I001 giả cho `from dfl…` (chế độ stdin không nhận `dfl` là first-party); kiểm trên file thật thì sạch
- Câu hỏi mở: QS-1…QS-8 (nhóm); Q1, Q2 (GVHD); có áp đề xuất sửa §1.3 theo DIGEST không (từ P1.2)
- Việc tiếp theo: nhóm duyệt SPEC → sửa theo quyết định → commit "docs: add SPEC.md (P1.5)" → cập nhật KH theo SPEC §10 → P1.6 (ADR)

## 2026-09-30 — Phiên #6 — P1.6 ADR
- Trạng thái: P1.6 ✅ — ADR-001…009 trong docs/decisions/ (Proposed, chờ G1) + docs/decisions/README.md (mục lục, quy tắc vòng đời). Commit riêng trước đó: SPEC.md Draft 0.1 (11aa794)
- File đã đổi: docs/decisions/ADR-001…009-*.md, docs/decisions/README.md, docs/decisions/OPEN_QUESTIONS.md (đánh dấu các mục 1B đã làm), tasks/SESSION_LOG.md
- Kiểm chứng: 9 file có đủ 6 mục của template (Status / Date / Context / Decision / Alternatives Considered / Consequences), đúng thứ tự; tiêu đề khớp số file; Status = Proposed; 11 link trong README đều trỏ tới file có thật; mọi con số trong ADR có trong KH/SPEC/DIGEST/spike (hai số tham số của McMahan 2017, 199,210 và 1,663,370, kiểm lại bằng tay); `ruff check .`, `ruff format --check .`, `pytest -q` (7 passed) xanh
- Quyết định mới: không có quyết định kỹ thuật mới. ADR ghi lại KH §10, cập nhật theo OQ (Q1 dự phòng T22, QĐ-4, A3, Q3, QĐ-2) và theo mặc định của SPEC (QS-1, QS-2, QS-4, QS-7). ADR-003 đổi thứ tự mở rộng (CIFAR-10 trước Fashion-MNIST) theo QĐ-4. Quy ước: ADR còn Proposed thì sửa trực tiếp; đã Accepted thì thay bằng ADR mới và đánh dấu "Superseded by"
- Câu hỏi mở: Q1 (ảnh hưởng ADR-002); QS-1…QS-8; η và μ điền vào ADR-007 sau E0a
- Việc tiếp theo: P1.7 (contracts.py), hoặc nhóm trả lời QS-1…QS-8 trước để sửa SPEC và ADR trong một lượt

## 2026-09-30 — Phiên #7 — P1.7 Hợp đồng giao diện
- Trạng thái: P1.7 ✅ — src/dfl/contracts.py (7 Protocol, 4 dataclass, 5 TypedDict cùng union SummaryRecord, alias StateDict) và tests/unit/test_contracts_smoke.py (25 test)
- File đã đổi: src/dfl/contracts.py, tests/unit/test_contracts_smoke.py, tasks/SESSION_LOG.md
- Kiểm chứng: test viết trước và FAIL (chưa có module), sau đó PASS; `ruff check .` và `ruff format --check .` sạch; `pytest -q` 32 passed; coverage src/dfl 100%; import dfl.contracts không nạp torch, numpy hay networkx. 20 URL trong docstring đều là trang tài liệu đúng phiên bản (torch 2.14, torchvision 0.29, networkx 3.7, numpy 2.5); subagent đã tải từng trang và đối chiếu với gói trong .venv (20/20 VERIFIED); các anchor state_dict, load_state_dict, default_rng có trong HTML đã lưu
- Quyết định mới (thiết kế contract, duyệt ở G1):
  - `Transport.send(dst, message)` lấy người gửi từ `Snapshot.sender`; thêm `end_round` và `counters()`. `CommCounters` chỉ gồm `messages_cum`, `bytes_cum`, `max_node_bytes_cum`; biến thể star của B3 tính cả server
  - Evaluator chỉ đánh giá trên tập test; drift và consensus distance là hàm thuần ở metrics/consensus.py
  - summary.json là `SummaryOk | SummaryFailed`, phân biệt bằng `status`, thay vì trường tùy chọn: với `from __future__ import annotations`, Python 3.12 báo trường NotRequired là required lúc chạy (đã thử)
  - std qua node dùng ddof = 0; timing.jsonl ghi cùng nhịp với rounds.jsonl
  - ManifestRecord có thêm `dirty`, `attempt`, `experiment`, `child_seeds` (tên luồng → số nguyên)
  - Mọi Protocol có `@runtime_checkable`; `StateDict` là `type` alias (PEP 695) để không nạp torch
- Ghi chú tài liệu:
  - Trang torch.Generator 2.14 không hiện chữ ký hàm khởi tạo; tham số `device` chỉ có trong ví dụ
  - Bảng backend của torch.distributed 2.14 chỉ có dòng send/recv, không có isend/irecv; T22 phải kiểm lại
  - `.data`/`.targets` của MNIST không có trong tài liệu 0.29; đã ghi UNVERIFIED trong NodeDataset
- Câu hỏi mở: QS-1…QS-8; Q1
- Việc tiếp theo: P1.8 (CONSTRAINTS.md + Makefile)

## 2026-09-30 — Phiên #8 — P1.8 CONSTRAINTS.md + Makefile
- Trạng thái: P1.8 ✅ — CONSTRAINTS.md (sàn F1–F9, ngưỡng có số, toàn vẹn nghiên cứu R1–R8, ratchet, lịch kích hoạt theo task, bảng ngoại lệ, canh gác khi review) và Makefile (check-fast, check-task, check-full, check-floor, check-inv)
- File đã đổi: CONSTRAINTS.md, Makefile (mới); pyproject.toml (import-linter 2.15 trong dev; luật ruff của sàn; pytest `pythonpath`; coverage cho src/dfl và analysis/ với `fail_under = 80`; 4 contract import-linter); .gitignore (`.import_linter_cache/`); CLAUDE.md (dòng "Đọc CONSTRAINTS.md…", import-linter, check-inv); SPEC.md lên Draft 0.2 (chốt QS-3, QS-5; ENG-6/7 dùng import-linter; quy ước tên test_t_inv_<k>_); ADR-007 và docs/decisions/README.md (QS-3 đã chốt); tasks/SESSION_LOG.md
- Kiểm chứng:
  - Trên nhánh hiện tại: `make check-fast`, `make check-task` (1 s / 90 s; coverage 100%), `make check-full` (có `pip check`) đều xanh; `make check-inv` báo thiếu cả 12 T-INV, đúng dự kiến vì chưa tới C3
  - Thử phá trên bản sao: 7 kiểu vi phạm sàn đều bị `check-floor` bắt (F2, F3, F4 ×2, F5, F6, F7), noqa có `lý do:` được chấp nhận; 7 luật ruff của sàn đều bắt lỗi (FIX002, PGH004, S110, NPY002, ERA001, S311, S506); 4 contract import-linter đều bắt được vi phạm (kể cả import torch gián tiếp) và không báo nhầm với import dưới TYPE_CHECKING
  - Cú pháp import-linter đối chiếu tài liệu chính thức (import-linter.readthedocs.io, bản stable, lấy ngày 2026-09-30)
- Quyết định mới (nhóm trả lời 4 câu hỏi của P1.8): coverage ≥ 80% cho src/dfl và analysis/ (QS-5); ngưỡng sanity 1.0 / 20 / 1.0 điểm % (QS-3); thêm import-linter; ngân sách thời gian chỉ cảnh báo
- Quyết định của agent (ghi trong CONSTRAINTS.md): smoke e2e là test pytest tạo ở T11 (thay cho bước riêng trong Makefile ở mẫu H§5.5), để C1/C2 vẫn chạy được check-full; check-inv vào check-full từ T11; `exclude_type_checking_imports = true`
- Ghi chú công cụ: import-linter báo lỗi khi `source_modules` có module chưa tồn tại (trong `forbidden_modules` và layer tùy chọn `(…)` thì được), nên partition/heterogeneity chỉ thêm vào contract `pure-no-torch` ở T04; ruff 0.16.9 không có luật cho `torch.load`, nên kiểm bằng grep
- Câu hỏi mở: QS-1, QS-2, QS-4, QS-6, QS-7, QS-8; Q1, Q2 (GVHD)
- Việc tiếp theo: P1.9 (phản biện thiết kế theo doubt-driven)

## 2026-09-30 — Phiên #9 — P1.9 Phản biện thiết kế (doubt-driven)
- Trạng thái: P1.9 ✅ — 3 vòng phản biện, reconcile đầy đủ trong docs/decisions/REVIEW-P1.9.md: vòng 1 (F1–F33 nội bộ + X1–X13 cross-model), vòng 2 (F1–F30), vòng 3 (F1–F26). Sau vòng 3 còn 3 phát hiện mức cao; cả ba đã sửa nhưng chưa được phản biện lại, và nhóm quyết commit (không chạy vòng 4). Cross-model: vòng 1 có làm; vòng 2, 3 skipped (nhóm không gửi kết quả)
- File đã đổi: SPEC.md (Draft 0.3); docs/decisions/ADR-002, 005, 006, 007, 008, 009; docs/decisions/REVIEW-P1.9.md (mới); CONSTRAINTS.md; Makefile; pyproject.toml (contract `no-dev-deps`); src/dfl/contracts.py; tests/unit/test_contracts_smoke.py; CLAUDE.md; README.md; tasks/SESSION_LOG.md
- Kiểm chứng:
  - `make check-full` xanh: 34 test, coverage 100%, 5 contract import-linter; `make check-inv` báo thiếu 12 T-INV, đúng dự kiến trước C3
  - Contract `no-dev-deps` thử phá trên repo (import scipy trong `dfl`) và trên bản sao (import sklearn trong `analysis`); cả hai đều bị bắt
  - Hai mẫu code ở SPEC §6/§7 chạy lại trên bản sao: ruff và format sạch, pytest pass
  - 30/32 URL trả 200; 2 trang pip trả 429 do bị chặn tần suất, nội dung đã đọc qua WebFetch
  - Tính lại trên 2,000 đồ thị ER(20, 0.3) liên thông: trung bình 115 message/round (72–154), spectral gap trung bình 0.20 (95%: 0.09–0.32); gossip 0.312 là tốc độ trộn, spectral gap của E[W] là 0.526
  - Đo ngân sách (vòng 1): 10.5 ms/bước với 1 luồng; đánh giá 2,000 ảnh mất 0.40 s
- Quyết định mới (của nhóm, 2026-09-30):
  - SAN-2 so với B1-4k (η chung, 4,000 bước), giữ ngưỡng 1.0 điểm % (QS-3, CONSTRAINTS R7)
  - Lưới đánh giá 100 bước chung cho mọi τ; E0a dùng seed 100
  - QS-10: tới hết tuần 5 GVHD chưa trả lời Q1 thì vẫn chạy lưới đơn tiến trình, và làm T22 ngay sau T13
  - QS-11: cài sẵn P2' (shards) ở T04b, chỉ dùng trong `describe`
  - Commit P1.9 như hiện tại
- Câu hỏi mở: QS-1, QS-2, QS-4, QS-6, QS-7, QS-8; Q1, Q2 (GVHD); Q8, Q9 (đề xuất hỏi thêm GVHD); có sửa HANDOFF cho khớp SPEC không (P3.2 seed, P3.6 chạy lại lượt lỗi)
- Việc tiếp theo: commit P2.1 (tasks/plan.md, tasks/todo.md); P2.2 phản biện kế hoạch

## 2026-09-30 — Phiên #10 — P2.1 Phân rã task
- Trạng thái: P2.1 ✅ — tasks/plan.md (quyết định K1–K13, đồ thị phụ thuộc, tách/gộp WBS, lịch tuần, checkpoint, T-INV → task, truy vết REQ, ngân sách, rủi ro, câu hỏi mở) và tasks/todo.md (42 task S/M, checkpoint C1–C5, mở rộng). Nhóm duyệt Gate G2 ngày 2026-09-30
- File đã đổi: tasks/plan.md (mới), tasks/todo.md (mới), SPEC.md (§3.1 phân vai và mã task, §5.1 cây thư mục, §7 TDD cho engine, §10 dòng 1, 38, 39, QS-9 đã chốt), tasks/SESSION_LOG.md
- Kiểm chứng: script kiểm todo.md cho 42/42 task có mô tả, ≤ 3 gạch chấp nhận, lệnh kiểm chứng, phụ thuộc, ≤ 5 file, cỡ S hoặc M; mỗi T-INV-1…12 thuộc đúng một task; bảng Markdown đúng số cột; `make check-task` xanh (34 test)
- Quyết định mới (nhóm duyệt): phân vai QS-9 (TV1 nhận T09a–b; TV3 nhận T02a–c, T11a); engine tách T10a–c, mỗi task viết T-INV của nó trước; thêm file ngoài cây KH §4.6 (SPEC §10 dòng 39)
- Câu hỏi mở: O4–O9 ở tasks/plan.md §12 (QS-1, hook mở rộng trước E0a, QS-8, Q1/Q2/Q9, sửa HANDOFF, QS-2)
- Việc tiếp theo: P2.2 phản biện kế hoạch (tối đa 2 vòng); rồi chờ G1 (GVHD) trước khi bắt đầu pha 3
