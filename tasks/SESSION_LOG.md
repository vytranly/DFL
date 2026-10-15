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

## 2026-09-30 — Phiên #11 — P2.2 Phản biện kế hoạch (vòng 1) + Gate G2
- Trạng thái: P2.2 vòng 1: cross-model thủ công (X1–X16) đã reconcile (tasks/plan.md §14); reviewer nội bộ vẫn chạy lúc commit, sẽ reconcile ở commit sau. Gate G2 ✅ (nhóm "duyệt")
- File đã đổi: tasks/todo.md (quy trình thí nghiệm; tách T16 thành T16a–c; T10c đánh giá từng node; T14a chọn η của B1; T11c biến thể uniform và size-corrected; T15b lượt phân kỳ; T05a `--check-req02`; C1 vẽ P7 vào build/; C4 đòi đủ Tier 1; gán lại cỡ S/M theo số file), tasks/plan.md, SPEC.md §8 và ADR-008 (code của lưới đóng băng ở một commit, sweep chạy từ worktree; nhánh chính vẫn nhận T22), tasks/SESSION_LOG.md
- Kiểm chứng: script kiểm todo.md: 43 task đủ trường, ≤ 3 gạch chấp nhận, ≤ 5 file, cỡ đúng; không có vòng phụ thuộc, không task song song với thứ nó phụ thuộc; T-INV-1…12 mỗi cái đúng một chủ; `make check-task` xanh
- Quyết định mới: sửa mâu thuẫn giữa SPEC §8 (đóng băng src/) và QS-10 (T22 ở tuần 8–10) theo hướng quyết định đã duyệt
- Câu hỏi mở: như phiên #10; G1 (GVHD) chưa có
- Việc tiếp theo: P3.0 — T02a (được làm trước G1, theo tasks/plan.md §11)

## 2026-09-30 — Phiên #12 — T02a Seeding (TV3)
- Trạng thái: T02a ✅ · T02b ⏳ chưa bắt đầu. Làm trước G1, vì T02a thuộc nhóm task ít phụ thuộc quyết định của GVHD (tasks/plan.md §11)
- File đã đổi: src/dfl/seeding.py, tests/unit/test_seeding.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: RED (ImportError, chưa có module) → GREEN; `make check-task` → 50 passed, coverage 100%, 5 contract import-linter giữ, sàn F7 đạt
- Quyết định mới (không cần ADR): `seeding.py` import torch để dựng `torch.Generator`, nên các module thuần (network, data.partition, data.heterogeneity) không import nó mà nhận Generator qua tham số do sim truyền (giữ contract pure-no-torch). ID luồng ghim bằng test. Seed của torch.Generator là `generate_state(1, uint64)` của SeedSequence, nằm trong dải của manual_seed. API đã đối chiếu tài liệu numpy 2.5 và torch 2.14 (URL trong docstring)
- Câu hỏi mở: không
- Việc tiếp theo: T02b (config + YAML base/smoke), T02c (cần QS-1), T07 (TV2), T04a (TV1)

## 2026-09-30 — Phiên #13 — P2.2 vòng 1: reconcile reviewer nội bộ
- Trạng thái: P2.2 vòng 1 xong (cross-model X1–X16, nội bộ F1–F29; tasks/plan.md §14). Vòng 2 chờ nhóm quyết
- File đã đổi: tasks/todo.md (46 task; tách T04b thành T04b và T04c, T11b thành T11b và T11d; T22 thành task đầy đủ; TV1 nhận T14a, T16b, T16c; task thí nghiệm có hai commit; SAN-5 nằm trong quy trình thí nghiệm; quy tắc đặt tên T-INV và quy tắc sửa `contracts.py`), tasks/plan.md, tasks/SESSION_LOG.md
- Kiểm chứng: script kiểm cấu trúc todo.md sạch (46 task, T-INV-1…12 mỗi cái đúng một chủ); `make check-task` xanh
- Quyết định mới: không đổi SPEC ở lượt này
- Câu hỏi mở: có chạy P2.2 vòng 2 không; QS-1 (chặn T02c → T11a)
- Việc tiếp theo: T02b

## 2026-10-01 — Phiên #14 — T04a Heterogeneity (TV1)
- Trạng thái: T04a ✅ · T04b đang làm trong cùng phiên
- File đã đổi: src/dfl/data/heterogeneity.py, tests/unit/test_heterogeneity.py, pyproject.toml (`dfl.data.heterogeneity` vào contract `pure-no-torch`), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: RED (ImportError) → GREEN, 12 test; cài thử `import torch` vào module thì `lint-imports` báo BROKEN, gỡ ra thì 5 contract đều giữ; `make check-task` → 62 passed, coverage 100%
- Quyết định mới (không cần ADR): công thức JS, TV, CV(n) với ddof = 0, và lệch mục tiêu giống hàm `heterogeneity` của spike, nên bảng KH §3.2 so được; P là phân bố của mọi mẫu được chia; π mặc định 1/N, nhận `objective_weights` theo ADR-005
- Câu hỏi mở: không
- Việc tiếp theo: T04b

## 2026-10-01 — Phiên #14 (tiếp) — T04b Partitioner P0–P3 (TV1)
- Trạng thái: T04a ✅ (bae729b) · T04b ✅ · T04c ⏳ chưa bắt đầu (hoán vị vị trí node + P2')
- File đã đổi: src/dfl/data/partition.py, tests/reference/niid_bench_reference.py, tests/unit/test_partition.py, pyproject.toml (`dfl.data.partition` vào contract `pure-no-torch`), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (ImportError) → GREEN; 33 test mới chạy trong khoảng 1.5 s
  - T-INV-8 đủ các phần: rời nhau, phủ đủ, ≥ 64, tất định, #C = k đúng k nhãn và mọi nhãn có chủ, N < 10 báo lỗi, golden trùng tuyệt đối với tham chiếu (P0, Dirichlet α = 0.1 và 0.05, #C = 2, quantity; seed 0–2; số `retries` khớp)
  - Thống kê lỏng của P3.1: α = 1000 cho JS < 0.01; JS tăng đơn điệu theo α; α = 0.1 có JS trong [0.45, 0.65]; quantity β = 0.5 có JS < 0.02 và CV > 0.5
  - Cài thử `import torch` thì contract `pure-no-torch` báo BROKEN; `make check-task` → 95 passed, coverage 99.7%
- Quyết định mới (không cần ADR):
  - `retries` = số lần thử − 1; tối đa 1,000 lần lấy mẫu lại rồi `ValueError`
  - P0 và #C = k không lấy mẫu lại được, nên node nhỏ hơn min_size thì báo lỗi cấu hình
  - Partitioner nhận Generator qua tham số, không import `seeding`
  - Bản tham chiếu sao từ spike, chỉ thêm `strict=True` cho `zip` (ruff B905), không đổi thứ tự rút số
- Ghi chú: diff khoảng 430 dòng, vượt mốc khoảng 300 dòng; khoảng 90 dòng là bản tham chiếu sao gần nguyên từ spike, giữ cùng commit để test golden có tham chiếu
- Câu hỏi mở: không
- Việc tiếp theo: T04c (placement + P2', QS-11) rồi T05a

## 2026-10-01 — Phiên #15 — QS-1 + T02c requirements.lock, .gitattributes (TV3)
- Trạng thái: QS-1 chốt theo mặc định (file khóa chỉ gồm gói chạy); T02c ✅
- File đã đổi: requirements.lock (mới, 24 gói), .gitattributes (mới), README.md, CLAUDE.md (lệnh cài đặt), SPEC.md (§4, QS-1 đã chốt), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - `requirements.lock` sinh bằng `pip freeze --exclude-editable` trong một venv chỉ cài torch/torchvision bản CPU và `pip install -e .`; không chứa pytest, ruff, import-linter hay scikit-learn. Có setuptools vì torch khai báo nó là phụ thuộc
  - Venv mới cài đúng SPEC §4 (`-r requirements.lock`, rồi `-e ".[dev]" -c requirements.lock`): `pip check` sạch, 95 test pass
  - `.venv` hiện tại khớp file khóa ở mọi gói
  - `git ls-files --eol`: mọi file văn bản là LF; `git add --renormalize .` không đổi nội dung file nào
- Quyết định mới: QS-1 (nhóm, 2026-10-01). Sửa CLAUDE.md chỉ ở dòng lệnh cài đặt; nhóm xem diff
- Câu hỏi mở: không
- Việc tiếp theo: T04c, rồi T07, T06, T08 (P3.2)

## 2026-10-01 — Phiên #15 (tiếp) — T04c Hoán vị vị trí node + P2' (TV1)
- Trạng thái: T04c ✅ — nhóm T04 xong (T04a, T04b, T04c)
- File đã đổi: src/dfl/data/partition.py (`assign_positions`, `ShardsPartitioner`), tests/unit/test_partition_extras.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: RED (AttributeError) → GREEN, 6 test; `make check-task` → 101 passed, coverage 99.7%
- Quyết định mới (không cần ADR): mỗi hàm chỉ rút đúng một `rng.permutation` (placement: hoán vị N vị trí; P2': hoán vị 2N mảnh), và thứ tự rút được ghi trong docstring để test golden so được; P2' sắp chỉ số bằng `argsort(kind="stable")`
- Câu hỏi mở: không
- Việc tiếp theo: T07, T06, T08 (P3.2)

## 2026-10-01 — Phiên #15 (tiếp) — T07 Mixing (TV2)
- Trạng thái: T07 ✅
- File đã đổi: src/dfl/network/mixing.py, tests/unit/test_mixing.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: RED (ImportError) → GREEN, 23 test; `make check-task` → 124 passed, coverage 99.2%, contract `pure-no-torch` (gồm `dfl.network`) giữ
  - T-INV-9: gap của ring(20) = ⅔(1 − cos(2π/20)) (sai số 1e-12) và 0.033 ± 0.001; FC = 1
  - MH đối xứng, ngẫu nhiên kép, không âm, bằng 0 ngoài cạnh (ring, FC, 2 ER, star); uniform chỉ ngẫu nhiên theo hàng trên ER và trùng MH trên ring; size-weighted khớp n_j/n và từ chối đồ thị không đầy đủ; matching cho W đối xứng, W² = W, node lẻ có W_ii = 1
  - π theo ADR-005; điểm đồng thuận của uniform (W^2000) khớp π ∝ d_j + 1
- Quyết định mới (không cần ADR): `spectral_gap` dùng eigvalsh khi W đối xứng, eigvals khi không; mixing không kiểm liên thông (việc của topology)
- Câu hỏi mở: không
- Việc tiếp theo: reconcile P2.2 vòng 2, rồi T06, T08

## 2026-10-01 — Phiên #15 (tiếp) — P2.2 vòng 2
- Trạng thái: P2.2 xong 2/2 vòng; reconcile vòng 2 ở tasks/plan.md §14. Cross-model vòng 2: chưa có kết quả
- File đã đổi: tasks/todo.md (49 task: thêm T11e, T14e, T22a/T22b; nhánh `grid` và PYTHONPATH cho worktree; T-INV-6 chuyển sang T10b; pilot ở C3), tasks/plan.md, tasks/SESSION_LOG.md
- Kiểm chứng: đã kiểm thực tế F1 (worktree nạp `dfl` từ repo chính nếu không đặt PYTHONPATH); script kiểm cấu trúc todo.md sạch; `make check-task` xanh
- Quyết định mới: nhánh `grid` cố định code của lưới từ E0a (khớp SPEC §5.3, §8: một commit code cho cả lưới)
- Câu hỏi mở: không
- Việc tiếp theo: T06, T08 (P3.2)

## 2026-10-01 — Phiên #15 (tiếp) — T06 Topology (TV2)
- Trạng thái: T06 ✅
- File đã đổi: src/dfl/network/topology.py, tests/unit/test_topology.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: RED (ImportError) → GREEN, 25 test; `make check-task` → 149 passed, coverage 99.4% (topology.py 100%), contract `pure-no-torch` giữ
  - Ring bậc 2, FC bậc N − 1, star một hub; ER liên thông, tất định theo seed, khác nhau giữa các seed; đồ thị tĩnh bị freeze
  - T3: mỗi node tối đa một đối tác, N = 21 thì đúng một node đứng ngoài; T3b chỉ dùng cạnh ER và là maximal matching
  - `graph_at` thuần: gọi lại, gọi xen, và hai lịch độc lập gọi theo thứ tự khác cho cùng đồ thị
  - E[W] của T3 qua 10⁴ round: lệch lớn nhất 3.19 SE (ngưỡng 4); bản cài sai (matching cố định) lệch 60 SE, nên test phân biệt được
  - Hợp 200 round của T3 và T3b liên thông
- Quyết định mới (không cần ADR): T3b dùng luồng `Stream.GOSSIP_ER` riêng (đúng hợp đồng `TopologySchedule`); cạnh ER sắp theo thứ tự chuẩn trước khi xáo; ER thử tối đa 1,000 lần, mỗi lần lấy seed nguyên mới từ luồng GRAPH; ring cần N ≥ 3. Lệch so với H P3.2: không tạo Generator từ [seed, r] (plan K10, ADR-009)
- Sự cố: floor F7 bắt chuỗi tạo Generator trong docstring; sửa câu chữ, không sửa luật
- Câu hỏi mở: không
- Việc tiếp theo: T08 (P3.2)

## 2026-10-01 — Phiên #15 (tiếp) — T08 Transport + CommLedger + snapshot (TV2)
- Trạng thái: T08 ✅ — P3.2 xong (T06, T07, T08)
- File đã đổi: src/dfl/network/transport.py, src/dfl/sim/node.py, tests/unit/test_transport.py, tests/unit/test_snapshot.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: RED (ImportError) → GREEN, 26 test; `make check-task` → 175 passed, coverage 99.4%; `lint-imports` 5/5 contract giữ (transport.py không import torch)
  - CommLedger: byte = Σ `nbytes`; tải = byte gửi + byte nhận; số đếm giữ qua `end_round`; message tự gửi bị từ chối và không bị tính
  - Snapshot SmallCNN (shape theo ADR-004) có `nbytes` = 4 × 46,730; dtype khác tính đúng theo element_size; gửi 4 láng giềng = 4 × 186,920 byte
  - Không copy payload (`is`); inbox chỉ chứa message của đúng node và round, chỉ đọc (MappingProxyType); `end_round` giải phóng tham chiếu (kiểm bằng weakref)
  - Server ảo id N (B3/E1c): 2N message, tải lớn nhất là của server
  - Theo H P3.2: gửi một message mỗi cạnh có hướng cho ring 40, FC 380, gossip 20. Đây là test thành phần; T-INV-6 qua engine vẫn thuộc T10b
- Quyết định mới (không cần ADR): mỗi (người gửi, người nhận, round) chỉ một message, gửi lần hai thì ValueError (tránh ghi đè mà vẫn tính phí); gửi hoặc đọc inbox của round đã `end_round` thì RuntimeError (gọi sai thứ tự); id N chỉ hợp lệ khi `with_server=True`; thêm `node_bytes()` cho tải theo từng node
- Câu hỏi mở: không
- Việc tiếp theo: theo tasks/todo.md (T05a/T05b, T03, T08b…); cross-model P2.2 vòng 2 vẫn chờ kết quả của nhóm

## 2026-10-01 — Phiên #15 (tiếp) — P3.3 bước 1 + T10a Engine ATC lõi (TV2)
- Trạng thái: P3.3 bước 1 xong (subagent `test-engineer` thiết kế 22 test cho T10a–c, nhóm duyệt danh sách và thứ tự T10a → T10b → T10c); T10a ✅
- File đã đổi: src/dfl/sim/node.py (`Node`, `combine`), src/dfl/sim/engine.py (mới), tests/conftest.py (mới), tests/integration/conftest.py (mới), tests/integration/test_engine_core.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED: conftest không import được `Node` (engine chưa có) → GREEN, 14 test tích hợp trong 1.7 s (chậm nhất 0.5 s); `make check-task` → 189 passed, coverage 99.5% (`sim/engine.py`, `sim/node.py` 100%)
  - Đột biến trên bản sao `src/` trong scratchpad (repo không bị sửa), cả 7 đều bị bắt: snapshot là view (G22) và ghi ngay sau từng node (G2) → T-INV-4; cộng theo thứ tự inbox → T-INV-4 (ER); cộng "self trước" → test FC trùng từng bit; trộn buffer int → test buffer (G3); trainer học trên bản sao → T-INV-2; W của FC bất kể đồ thị → T-INV-2
  - Fixture RNG (D6, cách mặc định): test thăm dò gọi `torch.rand` trong `tests/integration/` bị ERROR lúc teardown với thông báo T-INV-5; cùng test đó trong `tests/unit/` không bị kiểm. Đã xóa file thăm dò
- Quyết định mới (không cần ADR):
  - API: `Node(node_id, model, data, generator, loss_scale)` thay cho các dãy song song; `combine(own, inbox, weights_row)` là hàm thuần chỉ nhận snapshot của chính node và inbox (INV-1 theo chữ ký); engine duyệt theo thứ tự của dãy `nodes` được giao, không có tham số chỉ để test; `RoundResult(round_idx, weights chỉ đọc, train_losses, snapshots)`; `states()` trả bản sao
  - Combine cộng theo id tăng dần trên các j có W_ij ≠ 0 (kể cả chính node); W khác 0 mà thiếu snapshot thì RuntimeError (có test với transport làm rơi message); tensor không phải float giữ giá trị của chính node
  - Mọi test chạy torch một luồng (`pytest_configure` trong tests/conftest.py): máy chủ đang tải ~9/8 lõi, 8 luồng làm test tích hợp chậm ~5 lần (9 s so với 1.6 s); tài liệu torch yêu cầu gọi `set_num_threads` trước mọi phép tính, nên đặt ở hook cấu hình chứ không ở fixture
  - Thêm ngoài danh sách đã duyệt: test transport làm rơi message và test kiểm `node_id`/kích thước lịch (phủ nhánh lỗi)
- Lệch quy trình: diff ≈ 750 dòng (166 dòng `src/`, 580 dòng test và fixture dùng chung cho T10b, T10c), vượt mức ~300 dòng của P3.0; giữ một commit vì một task = một commit, như T04b
- Câu hỏi mở: D5 (SPEC §9.2 ghi T-INV-6 "unit · T08", plan/todo ghi T10b) chờ nhóm cho sửa SPEC
- Việc tiếp theo: T10b (test 9–16)

## 2026-10-01 — Phiên #15 (tiếp) — T10b Gossip + đường server (TV2)
- Trạng thái: T10b ✅
- File đã đổi: src/dfl/sim/engine.py (`weights_at`, kiểm INV-4, đường server), src/dfl/sim/node.py (tách `weighted_sum` khỏi `combine`), tests/integration/test_engine_consensus.py, tests/integration/test_engine_fedavg.py, tests/integration/test_engine_messages.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED: 12/26 fail đúng lý do (chưa có `weights_at`, chưa có `server_path`, `schedule`/`mixing` còn bắt buộc). 14 test qua ngay vì engine của T10a đã gọi `graph_at` mỗi round. Sức phân biệt của chúng kiểm bằng đột biến ở dưới
  - GREEN: 27 test mới; `make check-task` → 216 passed, coverage 99.5% (`sim/engine.py`, `sim/node.py` 100%); test chậm nhất là E[W] qua 10⁴ round, 2.1 s
  - T-INV-1 (float64, lr = 0, khởi tạo khác nhau, 20 round): ring(4), ER(4) và FC(4) đạt Ξ_t ≤ ρ^t Ξ_0 (1 + 1e-6) + 1e-12, với ρ lấy từ MH do test tự dựng; trung bình được bảo toàn (1e-12); quỹ đạo X_t = W^t X_0 (1e-12). Gossip: mỗi W đối xứng, ngẫu nhiên kép, W² = W, giá ngoài đường chéo trùng đúng cạnh của `graph_at(r)`; E[W] (N = 4) trong ±4 SE
  - T-INV-3: FC + size-weighted (kích thước 64/96/128/200), đường server, và FC + MH với kích thước bằng nhau đều khớp FedAvg viết trong test (atol 1e-6); đối chứng dương: FC + MH với kích thước lệch thì lệch FedAvg > 1e-3
  - T-INV-6 qua engine, N = 20: ring 40, FC 380, gossip 20, ER 2·|E|, server 2N message/round; byte = message × nbytes; cặp gửi–nhận trùng đúng cạnh có hướng; không tự gửi; server gánh tải lớn nhất
  - Đột biến mức engine trên bản sao `src/`, cả 7 bị bắt: `graph_at(0)` mọi round; dùng cột W thay hàng; trộn lười 0.5(I + W); server tổng hợp đều 1/N; chỉ gửi một chiều; node gossip đứng ngoài bỏ Adapt; bỏ kiểm INV-4
- Quyết định mới (không cần ADR):
  - D1 (nhóm duyệt cùng danh sách test): dung sai T-INV-1 theo todo, `(1 + 1e-6)·ρ^t·Ξ_0 + 1e-12`, với model float64, kèm phép kiểm hai phía X_t = W^t X_0. Dạng "+1e-6" của H P3.3 lệch với todo; thêm vào O8
  - D2: bỏ phần Monte Carlo tùy chọn cho Ξ_t². D3: E[W] lấy qua `weights_at(r)`, cùng hàm mà `run_round` dùng
  - Engine kiểm INV-4 cho mọi W trước khi dùng: shape N × N, hữu hạn, không âm, tổng hàng bằng 1 (1e-12), khác 0 chỉ trên cạnh của round; sai thì ValueError
  - Đường server: `Engine(..., server_path=True)` không nhận schedule/mixing; server ảo id N tổng hợp Σ_k (n_k/n) w_k theo id tăng dần rồi gửi một snapshot xuống; node lấy phần float từ snapshot đó và giữ tensor nguyên của mình; `RoundResult.weights` là ma trận hiệu dụng (mọi hàng bằng n_k/n); server thiếu snapshot thì RuntimeError
  - Không sửa conftest: bộ ghi cặp gửi–nhận viết trong test_engine_messages.py để T10b giữ 5 file
- Lệch quy trình: diff ≈ 630 dòng (147 dòng `src/`, 485 dòng test), vượt mức ~300 dòng của P3.0; giữ một commit như T10a
- Câu hỏi mở: D5 (sửa SPEC §9.2: T-INV-6 là "integration · T10b" thay vì "unit · T08", và câu "T10b có T-INV-1, 3" ở §7) vẫn chờ nhóm cho phép; chưa sửa SPEC
- Việc tiếp theo: T10c cần T09b (Evaluator, của TV1, chưa làm)

## 2026-10-01 — Phiên #15 (tiếp) — D5: SPEC khớp chủ sở hữu T-INV-6 (TV2)
- Trạng thái: nhóm cho phép sửa SPEC (D5, 2026-10-01)
- File đã đổi: SPEC.md (§7: "T10b có T-INV-1, 3, 6"; §9.2: T-INV-6 "integration · T10b" thay vì "unit · T08"), tasks/SESSION_LOG.md
- Kiểm chứng: diff chỉ gồm đúng hai chỗ trên; `make check-task` xanh
- Quyết định mới: SPEC khớp plan §8 và todo (P2.2 vòng 2 đã chuyển T-INV-6 sang T10b). T08 giữ test thành phần của CommLedger, không mang tên T-INV
- Câu hỏi mở: không
- Việc tiếp theo: T09b (nhóm giao cho phiên này vì T10c cần Evaluator), rồi T10c

## 2026-10-01 — Phiên #15 (tiếp) — T09b Độ đo phân loại + consensus + Evaluator (phiên này làm thay TV1, nhóm giao)
- Trạng thái: T09b ✅ (nhóm chọn để phiên này làm T09b, vì T10c cần Evaluator)
- File đã đổi: src/dfl/metrics/classification.py, src/dfl/metrics/consensus.py, tests/unit/test_classification.py, tests/unit/test_consensus.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (ImportError: chưa có hai module) → GREEN, 23 test; `make check-task` → 239 passed, coverage 99.6% (hai module mới 100%)
  - T-INV-10: macro-F1 khớp `f1_score(average="macro", labels=range(10), zero_division=0)` của scikit-learn 1.9 (sai số 1e-12) trên bốn ca: nhãn ngẫu nhiên, dự đoán chỉ một lớp, không bao giờ đoán lớp 7, lớp 9 vắng ở cả nhãn lẫn dự đoán; không có NaN. Acc^loc và Acc^eff khớp `accuracy_score` với trọng số từng mẫu q(y)/P_eval(y) (1e-12)
  - Evaluator khớp sklearn trên dự đoán thật của từng model; acc(w̄) khớp model trung bình; chia batch không đổi kết quả; state và `torch.get_rng_state()` không đổi; state được nạp vào đúng model cấp phát sẵn (cùng tham số, cùng bộ nhớ)
  - Ξ khớp ví dụ tính tay sqrt(14/3); bằng đúng 0 khi mọi model trùng nhau; bỏ qua tensor nguyên; tính bằng float64 (ca 10⁴ ± 2⁻¹¹ mà float32 tính sai)
  - Đột biến trên bản sao `src/`, cả 6 bị bắt: macro-F1 chỉ trên lớp có mặt; Acc^eff dùng P_i; recall chia cho số lần đoán; Ξ bằng float32; trả Ξ²; tạo model mới mỗi lần đánh giá. Đột biến cuối lúc đầu không bị bắt, nên test model cấp phát sẵn được thêm điều kiện "model được giao đã nhận state"
- Quyết định mới (không cần ADR):
  - API cho T10c: `ClassificationEvaluator(model, x, y, *, label_distributions (N × 10), objective_weights (N), batch_size=1000)`, nhận model bất kỳ kiến trúc, ép x về dtype của model; `consensus_distance(states)` dùng cho cả drift (trên snapshot) và consensus distance (sau tổng hợp)
  - Tập đánh giá phải có đủ 10 lớp, nếu không thì ValueError, để recall của mọi lớp xác định và Acc^loc bằng đúng accuracy có trọng số
  - acc(w̄): trung bình từng tensor float qua các node (torch.mean, dtype của tensor); tensor nguyên lấy của node 0, vì không có trung bình có nghĩa
- Câu hỏi mở: không
- Việc tiếp theo: T10c (Observer, T-INV-11, T-INV-12, kiểm hữu hạn) rồi doubt-driven cho engine.py + node.py

## 2026-10-01 — Phiên #15 (tiếp) — Sửa kế hoạch theo trả lời Q1 của GVHD (đề xuất, chờ nhóm duyệt)
- Trạng thái: GVHD trả lời Q1 lúc 17:27: "Cô cần các process chạy song song, mỗi node chạy độc lập. Các em có thể tạo dockers, số lượng ít cũng được. Chạy mấy mô hình đơn giản, sau mình sẽ tăng độ phức tạp lên xem communication cost sẽ thay đổi như thế nào." Kế hoạch đã sửa theo phương án ghi sẵn ở ADR-002 (mức bổ sung), cộng phần mới: Docker và thang model
- File đã đổi: docs/decisions/ADR-010-multiprocess-node-runtime.md (mới), docs/decisions/ADR-011-model-complexity-ladder.md (mới), docs/decisions/ADR-002 (Status và mục Q1, theo quy tắc sửa ADR còn Proposed), docs/decisions/README.md, docs/decisions/OPEN_QUESTIONS.md (Q1 đã trả lời; Q10–Q13 mới; tin nhắn hỏi lại GVHD soạn sẵn), tasks/plan.md (K14, đồ thị, lịch tuần 5–9, ngân sách, rủi ro, câu hỏi mở), tasks/todo.md (T22a–c bắt buộc trước E0a; T09c; E8 = T27a–c; T13, T14b, T26 cập nhật), tasks/SESSION_LOG.md
- Kiểm chứng:
  - Máy chủ có Docker 29.4.0, Docker Compose v2.36.0; torch có gloo
  - Bảng backend của torch 2.14 ghi gloo trên CPU có send/recv/barrier/gather; isend/irecv không có trong bảng, nên T22a kiểm lại
  - Số tham số của M0–M3 kiểm bằng torch: 7,850 / 46,730 / 199,210 / 1,663,370
  - Script kiểm cấu trúc todo: 54 task, không lỗi (≤ 3 gạch chấp nhận, ≤ 5 file, nhãn S/M theo số file, phụ thuộc có thật, không vừa ∥ vừa phụ thuộc)
- Quyết định đề xuất (chờ nhóm duyệt, ADR ở trạng thái Proposed):
  - ADR-010: mỗi node một tiến trình (tiến trình cục bộ cho test, container Docker 4–8 cho demo và E8), gloo, giao thức đồng bộ không có điều phối viên; bộ mô phỏng một tiến trình giữ lại cho lưới N = 20; T-INV-13 buộc hai runtime cho rounds.jsonl trùng từng byte
  - ADR-011: thang M0 (logistic) → M1 SmallCNN → M2 2NN → M3 CNN McMahan; E8 = 4 model × {ring, gossip, FC} × {IID, α = 0.1} × 3 seed = 72 lượt, N = 8, trong Docker
- Câu hỏi mở: Q10–Q13 gửi GVHD (OPEN_QUESTIONS.md); O13: các chỗ sửa SPEC, KH, Makefile, CONSTRAINTS (Ask first) chưa làm, chờ nhóm duyệt K14
- Ghi chú: `docs/đath1.docx` đổi lúc 17:46 (55 KB → 297 KB) không do phiên này; không sửa, không commit
- Việc tiếp theo: nhóm duyệt K14; gửi tin nhắn Q10–Q13; sửa SPEC/KH sau khi duyệt; hoàn tất T10c (reconcile phản biện vòng 2, xem build/review/P3.3-doubt-cycle2-result.md), rồi T22a

## 2026-10-01 — Phiên #15 (tiếp) — Nhóm duyệt K14; SPEC, CONSTRAINTS, Makefile theo ADR-010/011
- Trạng thái: nhóm duyệt K14 (2026-10-01). Q12: Docker là tùy chọn. Q13: đo cả byte lẫn thời gian. Q10, Q11: nhóm quyết sau, trong lúc chờ dùng mặc định
- File đã đổi:
  - SPEC.md (ASSUMPTIONS 1; §1.4; §3.1; §4; §5.1–§5.3; §7; REQ-05, REQ-10, REQ-15, REQ-16; T-INV-13; EXP-2 = 269 với E8; EXP-4 H1–H8; §10 dòng 40; Q1, Q10–Q13, QS-10)
  - CONSTRAINTS.md (T-INV-1…13; phép kiểm Docker vào nhóm "ý kiến bên ngoài"; kích hoạt ở T22b, T22c), Makefile (`check-inv` đếm tới 13)
  - ADR-010, ADR-011 (Status: nhóm duyệt; Docker tùy chọn), ADR-007 (ngân sách tách E8)
  - OPEN_QUESTIONS.md (Q12, Q13 đã chốt; Q10, Q11 để sau), tasks/plan.md, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng: script kiểm cấu trúc todo: 54 task, không lỗi; số cột của mọi dòng bảng mới trong SPEC đúng; `make check-inv` báo thiếu đúng T-INV-5, 7, 13 (của T11b, T09a, T22b, chưa làm); `make check-task` xanh
- Quyết định mới:
  - Docker tùy chọn (Q12): T22c ra khỏi đường găng, không chặn E0a. File Docker nằm ngoài `src/`; container gọi cùng lệnh `dfl node` (T22b) như launcher cục bộ
  - E8 chạy bằng tiến trình cục bộ (`deploy: local`), thêm 4 ô trong Docker nếu có T22c
  - E8 thuộc Tier 1 (yêu cầu của GVHD): Tier 1 = 197 lượt lưới lõi + 72 lượt E8
  - KH chưa sửa: theo SPEC §10 (dòng 40 mới), KH được cập nhật sau Gate G1
- Câu hỏi mở: Q10, Q11 (nhóm); G1 (GVHD duyệt SPEC, ADR-001…011)
- Việc tiếp theo: hoàn tất T10c (reconcile phản biện vòng 2), rồi T22a

## 2026-10-01 — Phiên #15 (tiếp) — T10c Observer + phản biện engine, kèm T10d (TV2)
- Trạng thái: T10c ✅; T10d ✅ (bản sửa sau phản biện, gộp cùng commit)
- File đã đổi: src/dfl/sim/observer.py (mới), src/dfl/sim/engine.py, tests/integration/conftest.py (`eval_set`, `make_evaluator`), tests/integration/test_observer.py (mới), tests/integration/test_engine_review.py (mới), tests/integration/test_engine_consensus.py và tests/integration/test_engine_messages.py (thêm `check_schedule`, 1–4 dòng), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - T10c: RED (chưa có `RunDiverged`, `observer`) → GREEN. T-INV-11: FC + MH cho node trùng từng bit, `acc_std` = 0, acc^loc = P_i · cùng một vector recall (sklearn). T-INV-12: tắt Observer, hay gọi `graph_at` thêm ≥ 12 lần, thì quỹ đạo vẫn trùng từng bit. Lịch đánh giá 100 bước + round cuối; `RoundRecord` đủ trường, không NaN; drift và consensus là RMS (oracle độc lập); loss NaN hay tham số inf thì dừng đúng round
  - `make check-full` → 299 passed, coverage 99.7% (`sim/` 100%), `pip check` sạch. 22 đột biến của T10a–c đều bị bắt với engine bản cuối
  - Test E[W] (10⁴ round) mất 2.5 s khi chạy riêng, 3.4–4.6 s trong cả bộ lúc máy tải nặng; ma trận kề đã vector hóa (90 → 71 µs/round). Theo dõi ở T13
  - Thăm dò gloo (scratchpad, 4 tiến trình trên ring): `isend`/`irecv` và `send`/`recv` theo thứ tự cố định đều đúng từng bit, 7–11 ms/lần, cho T22a
- Reconcile doubt-driven (H P3.3 bước 4; ARTIFACT = engine.py + node.py; CONTRACT = KH §3.4, §4.5, H§2.4 G2–G7, G15, G18, G21, G22; reviewer ngữ cảnh mới, không nhận lời khẳng định của tác giả):
  - **Vòng 1:** code-reviewer: 0 critical, 3 major, 13 minor. Cross-model thủ công do nhóm chạy: 15 phát hiện, tự nhận 4 critical. Gộp 31 phát hiện:
    - Sửa (13 nhóm): node độc lập (Generator, bộ nhớ, nhãn dữ liệu, BatchNorm); dừng engine sau lỗi giữa round; `RunDiverged` pickle được; lý do phân kỳ theo id; w⁰ không hữu hạn là lỗi lúc dựng; INV-4 hai chiều, đồ thị tĩnh liên thông, cấu trúc đồ thị; kích thước node và c_i; cấu trúc state đồng nhất; `RoundResult` so sánh theo danh tính; W tĩnh dựng một lần; docstring
    - Đánh đổi (4): float32(W_ij), tensor nguyên trên đường server, traffic của round phân kỳ, tham chiếu inbox sau khi dừng
    - Không phải lỗi (4): `counters()` sau khi dừng là cần thiết; float8; kiểm W trong `combine` (đã kiểm ở biên engine); snapshot round cũ (tới vòng 3 thì thành lỗi thật)
  - **Vòng 2:** 0 critical, 2 major, 11 minor. Đã đề nghị cross-model, nhóm không yêu cầu, nên vòng này không có.
    - Sửa (8): hợp các round của gossip liên thông; Adapt dùng RNG toàn cục; Generator trùng seed; W chỉ đọc; thuộc tính "weight" của cạnh; `states()` sau khi dừng; lớp giữ thống kê chạy; len(x) = len(y)
    - Đánh đổi (3): lỗi lắp ráp đường server chỉ lộ ra ở round 0; `nbytes` theo hợp đồng; FC + MH float64 lệch bit cuối
    - Không phải lỗi (2)
  - **Vòng 3** (vòng cuối): 0 critical, 2 major, 13 minor. Cross-model không chạy; file đề build/review/P3.3-doubt-cycle3.md có sẵn nếu nhóm muốn. Làm theo H P3.4: viết 13 test hồi quy, xác nhận FAIL, rồi mới sửa. Nguyên nhân gốc mỗi lỗi, một câu:
    - #1: MT19937 chỉ dùng 32 bit thấp của seed, mà phép kiểm so cả 64 bit → so 32 bit thấp
    - #2: INV-4 của gossip chỉ được kiểm khi người gọi dùng `run_rounds` → engine từ chối round ngoài phạm vi đã `check_schedule`
    - #3: Combine tin khóa của inbox → kiểm người gửi và round
    - #4: `.base` của view chỉ đọc vẫn bật lại quyền ghi được, và cache dùng chung đồ thị của lịch → W bất biến (dựng trên bytes), cache bản sao đóng băng của đồ thị
    - #6b, #6c: chỉ có phép kiểm RNG lúc chạy → từ chối `default_generator` và Dropout lúc dựng
    - #9: tính thuần của lịch chỉ được giả định → mỗi round gọi `graph_at` hai lần và so
    - #11: `total_rounds` < 1 không bị chặn → phải ≥ 1
    - Hai test của T10b (`test_engine_messages`) và một test của T10d fail sau khi sửa. Khoanh vùng cho thấy lỗi ở test: hợp hai matching trên K_N chưa liên thông. Đã sửa thành khai báo phạm vi của một lượt thật (50 hay 10 round); không hạ ngưỡng nào
    - Đánh đổi, ghi vào docstring: `batch_norm` dạng hàm và lớp con riêng của `_BatchNorm`; RNG của numpy và `random` (đã bị lint chặn); thăm dò transport lúc dựng; trainer phải không có trạng thái; tham số buộc chung tính hai lần; dấu của tổng hàng float32; dữ liệu chỉ kiểm theo nhãn
    - Không phải lỗi: w⁰ chung, vì T-INV-1 cần khởi tạo khác nhau
  - Sau 3 vòng vẫn còn phát hiện thực chất ở vòng cuối (đã sửa). Theo doubt-driven, đây là thông tin về artifact: đề nghị review lại engine ở C3 (P3.7, code-reviewer)
- Quyết định mới (không cần ADR): API Observer và `run_rounds` (generator; kiểm lịch trước round đầu); `RunDiverged(round_idx, reason)` với `diverged_at_round` = round_idx + 1; std bằng `statistics.pstdev` (D4); `check_union_connected` là hàm dùng chung cho T22b; T22b phải gọi nó và kiểm inbox như engine
- Lệch quy trình:
  - Chạm 7 file, trong đó 2 file test của T10b chỉ thêm 1–4 dòng
  - Diff khoảng 1,450 dòng; bản sửa sau phản biện vượt khoảng 300 dòng, nên được ghi thành T10d nhưng gộp cùng commit, vì sửa đan xen trong `engine.py` và trạng thái trung gian chưa từng được test
- Câu hỏi mở: không
- Việc tiếp theo: T22a (Transport gloo), cùng T11a, T09a, T02b, T03 theo thứ tự phụ thuộc

## 2026-10-02 — Phiên #16 — T22a TorchDistTransport (gloo) (TV2)
- Trạng thái: T22a ✅
- File đã đổi: src/dfl/sim/torchdist.py, tests/unit/test_torchdist_codec.py, tests/integration/test_torchdist.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có module) → GREEN, 16 test; `make check-task` → 315 passed, coverage tổng 95.8%
  - Unit: mã hóa rồi giải mã trùng từng bit với float32, float64, int64, bool, tensor rỗng, tensor 0 chiều và tensor không liền bộ nhớ; số byte = `nbytes` của snapshot; tensor giải mã có bộ nhớ riêng; layout lệch thì ValueError; tự gửi, người gửi lạ, dst ngoài 0..N-1, và đọc inbox của node khác (INV-1) đều bị chặn trước khi chạm mạng
  - Integration: 4 tiến trình `spawn`, rendezvous `file://` (không tranh cổng), ring và FC, 2 round mỗi đồ thị, có hạn giờ 90 s. Mỗi node nhận đúng snapshot của láng giềng, trùng từng bit và đúng dtype. Message, byte và tải của từng node bằng InMemoryTransport; tổng qua node bằng số đếm của InMemoryTransport. Chạy 3.3 s
- Quyết định mới (không cần ADR):
  - Primitive: `isend`/`irecv` có tag = round; `irecv` đăng ký trước bằng `expect(round, senders)`. Theo tài liệu thì UNVERIFIED cho gloo (bảng backend không có dòng `isend`/`irecv`); đã kiểm bằng thực nghiệm (thăm dò ở scratchpad và test tích hợp)
  - Message là một tensor uint8 liền theo `StateLayout`, giải mã bằng `view(dtype)` trên bản clone (tránh điều kiện căn lề của offset)
  - CommCounters của một tiến trình là của riêng node: message và byte nó gửi, `max_node_bytes_cum` = tải gửi + nhận của nó; observer của T22b cộng lại
  - Buffer đã gửi được giữ tới `end_round`, theo cảnh báo của tài liệu `isend`; `end_round` báo lỗi nếu còn snapshot đã expect mà chưa đọc inbox
- Câu hỏi mở: coverage của tiến trình con không được đo (torchdist.py 66% theo báo cáo, dù các dòng còn lại chạy trong test tích hợp). T22b chạy phần lớn code trong tiến trình con, nên giải quyết ở đó: bật đo coverage cho tiến trình con. Nếu cần sửa cấu hình coverage trong pyproject (thắt chặt, không nới) thì hỏi nhóm
- Việc tiếp theo: T02b

## 2026-10-02 — Phiên #16 (tiếp) — T02b Cấu hình + YAML base/smoke (TV3)
- Trạng thái: T02b ✅
- File đã đổi: src/dfl/config.py, tests/unit/test_config.py, configs/base.yaml, configs/profiles/smoke.yaml (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có `dfl.config`) → GREEN, 42 test; `make check-task` → 357 passed, coverage 97% (`config.py` 100%); import-linter 5/5
  - Đã kiểm PyYAML 6.0.3: `5e-2` thành chuỗi, `5.0e-2` thành 0.05, `yes` thành True
  - base.yaml ghi đúng mặc định của `RunConfig` (test so JSON chuẩn hóa); smoke.yaml theo SPEC §5.2 (synthetic, N = 4, τ = 2, R = 6, đánh giá mỗi round, 500 ảnh)
  - Hai YAML tương đương (khác thứ tự khóa, 0.05 so với 5.0e-2, 0 so với 0.0) cho cùng JSON chuẩn hóa; JSON không chứa `extends`, tên nhóm, `--out`, `--workers`
  - 29 ca cấu hình sai đều ra ValueError nêu ràng buộc, gồm: `5e-2` (có gợi ý viết 5.0e-2), #C = k với N < 10 (G23), lưới 100 bước không chia hết cho τ, khóa lạ, bool hay số thực cho trường số nguyên, `yes` cho trường enum, docker không đi với processes, size_weighted ngoài FC, server không dùng size_weighted, tham số scheme thiếu hay thừa, `extends` vòng
  - Đọc cấu hình không nạp torch (kiểm trong tiến trình sạch)
- Quyết định mới (không cần ADR; schema có thể còn thêm trường ở T11c cho B1):
  - Schema: `dataset`, `num_nodes`, `seed`, `local_steps` (τ·R), `model`, `mixing`, `threads`, `runtime`, `deploy`, và các nhóm con `partition` (scheme, alpha, k, beta, min_size), `topology` (kind, p), `algorithm` (tau, lr, mu, batch_size), `eval` (split, subset_size, every_steps); `rounds` = local_steps / τ
  - B3/E1c là `topology.kind: server` (đường server của engine), khác star T0; bắt buộc đi với `mixing: size_weighted`
  - Tham số scheme hay topology không dùng phải là null; ER không ghi p thì p = 0.3; trường số thực luôn là float; mọi `RunConfig` đều được kiểm trong `__post_init__`
  - Profile kế thừa bằng `extends` (đường dẫn tính từ file đang đọc; trộn mapping đệ quy)
- Câu hỏi mở: có nên thêm `dfl.config` vào contract `pure-no-torch` của import-linter không (thắt chặt; sửa pyproject, hỏi trước)? Hiện đã có test bằng tiến trình con
- Việc tiếp theo: T09a

## 2026-10-02 — Phiên #16 (tiếp) — T09a SmallCNN + huấn luyện cục bộ (TV1)
- Trạng thái: T09a ✅
- File đã đổi: src/dfl/learning/models.py, src/dfl/learning/local_train.py, tests/unit/test_models.py, tests/unit/test_local_train.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có module) → GREEN, 24 test; `make check-task` → 381 passed, coverage 96.6% (hai module mới 100%)
  - T-INV-7: với μ = 0, `SgdTrainer` trùng từng bit với `torch.optim.SGD` (momentum 0, weight decay 0, `foreach=False`) trên cùng minibatch, ở c_i = 1 và c_i = 2.5; Generator batch kết thúc cùng trạng thái; mất mát trả về là trung bình τ bước. Gradient của `prox_term` bằng μ(w − anchor); anchor là bản clone + detach, không gradient, không chung bộ nhớ. Với μ = 5, kết quả khớp oracle có anchor cố định từ đầu lời gọi và lệch SGD không prox 0.017 (ngưỡng test 1e-3)
  - n_i = 8 < B = 32 vẫn chạy và khớp oracle (lấy có hoàn lại)
  - Mất mát inf hay NaN ở bước 3: trả về đúng giá trị đó, chỉ 3 lần forward, và tham số bằng model sinh đôi chạy 2 bước (bước hỏng không cập nhật)
  - SmallCNN: đúng 46,730 tham số, không BatchNorm/Dropout (theo danh sách của engine), không buffer; dựng model và huấn luyện không đổi `torch.get_rng_state()`; cùng seed cùng w⁰, khác seed khác ở mọi tensor; w⁰ nằm trong U(−1/√fan_in, 1/√fan_in)
  - Mutation: 16/16 mutant bị bắt (anchor lấy lại mỗi bước, anchor không clone, sai dấu, cập nhật trước khi kiểm hữu hạn, chạy tiếp sau NaN, trả mất mát cuối, RNG toàn cục, lấy không hoàn lại, μ thay vì μ/2, bỏ prox, c_i chỉ vào gradient, cận khởi tạo sai hai kiểu, khởi tạo bằng RNG toàn cục, bỏ khởi tạo bias, khởi tạo mặc định)
- Quyết định mới (không cần ADR):
  - Cập nhật bằng `torch.autograd.grad` rồi `param.add_(grad, alpha=-lr)`, đúng phép cập nhật vòng for của `torch.optim.SGD` (đã đọc mã nguồn torch 2.14 đã cài; trên CPU mặc định không dùng foreach); model không giữ `.grad`
  - Mất mát trả về gồm cả c_i và số hạng prox, đúng định nghĩa trong `contracts.LocalTrainer`; với μ = 0 không tính prox
  - w⁰: `skip_init(SmallCNN)` rồi `uniform_(−b, b, generator=…)` với b = 1/√fan_in, theo thứ tự lớp, weight trước bias. `init_from_stream` raise ValueError nếu còn tham số hay buffer không có quy tắc khởi tạo (sau skip_init chúng là bộ nhớ rác); T09c dùng lại hàm này cho M0, M2, M3
- Câu hỏi mở: không
- Việc tiếp theo: T03

## 2026-10-02 — Phiên #16 (tiếp) — T03 Datasets + lệnh `download` (TV1)
- Trạng thái: T03 ✅
- File đã đổi: src/dfl/data/datasets.py, src/dfl/cli.py, src/dfl/__main__.py, tests/unit/test_datasets.py, tests/unit/test_cli.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có module) → GREEN, 38 test; `make check-task` → 419 passed, coverage 96.8% (`datasets.py` 100%; `__main__.py` chỉ chạy trong tiến trình con nên không được đo)
  - Phân tầng: trên nhãn giả cùng tỉ lệ lớp với MNIST, validation 5,000/60,000 và tập con 2,000/10,000 lệch tỉ lệ dưới 1 ảnh mỗi lớp; phương pháp số dư lớn nhất kiểm bằng ví dụ tính tay; cùng luồng thì cùng chỉ số, khác khóa thì khác
  - `prepare`: validation lấy đúng từ luồng VAL_SPLIT; phần còn lại giữ thứ tự gốc; mean/std khớp phép tính độc lập trên phần chia cho node (rel 1e-12), cả khi đếm qua 158 khối; test được chuẩn hóa bằng thống kê của train; float32 (n, 1, 28, 28) và nhãn int64 (cả khi nhãn đầu vào là uint8)
  - File MNIST giả theo định dạng IDX: bộ nạp (đọc `.data`/`.targets`, UNVERIFIED) cho đúng mảng đã ghi và khớp `__getitem__` có tài liệu; `load("mnist")` tách 55,000/5,000/10,000; thiếu file thì FileNotFoundError kèm lệnh tải
  - Dữ liệu tổng hợp: cố định, cân bằng lớp, 1,800/200/500 như SPEC §5.2; phân loại theo ảnh trung bình của lớp đúng hơn 90% ảnh test
  - CLI: `import dfl.cli` và `python -m dfl --help` không nạp torch (tiến trình sạch); `download mnist` gọi đúng thư mục (mặc định `data`); dòng lệnh sai thoát mã 2
  - Mutation: 20/20 mutant bị bắt (một mutant viết sai lúc đầu, tương đương code gốc, đã viết lại)
  - Kiểm tay trên máy chủ: `python -m dfl download mnist` tải 4 file vào `data/MNIST/raw` trong 14.6 s (torchvision kiểm md5), chạy lại thì không tải; `data/` không hiện trong `git status`. MNIST thật: 55,000/5,000/10,000, validation lớp 0 có 494 ảnh (≈ 5923/12), mean 0.130613, std 0.308059, nạp 0.46 s; tập con test 2,000 ảnh phân tầng
- Quyết định mới (không cần ADR):
  - Đọc MNIST bằng `.data`/`.targets` (UNVERIFIED theo tài liệu 0.29): đo trên 60k ảnh giả, `__getitem__` có tài liệu mất 7.1 s, `.data` mất 0.06 s. Test giữ hai đường khớp nhau
  - Chuẩn hóa bằng mean/std của 55k ảnh chia cho node (không gồm validation), tính từ histogram 256 mức theo khối 4,096 ảnh (tránh mảng tạm int64 345 MB), rồi tra bảng 256 giá trị float32
  - Khóa luồng EVAL_SUBSET: test 0, validation 1. Phân tầng chọn trong từng lớp bằng `Generator.choice(replace=False)`, lớp 0 trước; lớp có định mức 0 không gọi `choice`
  - `Datasets.split(name)` trả đủ tập cho đánh giá cuối; `eval_subset` cho learning curve
- Lệch quy trình: thêm `tests/unit/test_cli.py`, không có trong danh sách file của T03 (test của `cli.py`, vốn có trong danh sách)
- Câu hỏi mở: không
- Việc tiếp theo: T11a

## 2026-10-02 — Phiên #16 (tiếp) — T09a bổ sung test guard khởi tạo (TV1)
- Trạng thái: T09a ✅ (bổ sung test)
- File đã đổi: tests/unit/test_models.py, tasks/SESSION_LOG.md
- Kiểm chứng: báo cáo coverage của T03 lộ ra nhánh `raise ValueError` của `init_from_stream` (models.py:67) chưa có test. Thêm test: model có tham số hay buffer không thuộc Conv2d/Linear thì raise ValueError nêu tên tensor. Test FAIL khi bỏ guard (bản sao src ở scratchpad), PASS với code hiện tại; `make check-task` xanh
- Câu hỏi mở: không
- Việc tiếp theo: T11a

## 2026-10-02 — Phiên #16 (tiếp) — T11a Manifest, `code_version`, RunLogger (TV3)
- Trạng thái: T11a ✅
- File đã đổi: src/dfl/io/manifest.py, src/dfl/io/run_logger.py, tests/unit/test_manifest.py, tests/unit/test_run_logger.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có module) → GREEN, 57 test; `make check-task` → 478 passed, coverage 97.2% (hai module mới 100%); import-linter 5/5
  - `code_version` (trên gói giả trong thư mục tạm): không đổi khi đổi CRLF ↔ LF, sửa file ngoài gói (docs, configs), sửa file không phải `.py` trong gói, hay đổi thứ tự, khoảng trắng, dòng trống của file khóa; đổi khi sửa, đổi tên hay thêm file `.py`, hay đổi phiên bản trong file khóa. `current_code_version()` hash đúng gói đang nạp (`dfl.__file__`) cùng `requirements.lock` của repo
  - `run_id` đổi khi cấu hình, `code_version` hay máy đổi. Máy = tên CPU kèm hash cờ CPU đã sắp xếp (đổi thứ tự cờ không đổi máy, thêm cờ thì đổi)
  - Cây sạch, trên repo git tạm: sửa file đã track, file chưa track trong `src/` hay `configs/`, sửa file khóa, và file `*.py`/`*.yaml`/`*.yml` bị ignore (kể cả trong thư mục bị ignore) đều làm bẩn; `__pycache__`, `*.egg-info` và thay đổi ngoài `src/`, `configs/` thì không. Không có git, repo chưa có commit, máy không có lệnh git, hay gọi từ thư mục con của repo đều cho ("none", bẩn)
  - Môi trường: gói lệch phiên bản hay chưa cài thì `env_match: false` và có mô tả từng gói; công cụ dev không có trong file khóa bị bỏ qua; file khóa của repo khớp venv hiện tại; dòng không có dạng `tên==phiên bản` thì ValueError
  - RunLogger: ghi đủ 4 file; cùng bản ghi (khác thứ tự khóa) cho `rounds.jsonl` trùng từng byte; attempt đã có thì FileExistsError và không đụng tới file cũ; gọi sai thứ tự thì RuntimeError; NaN hay vô cực thì ValueError mà không ghi dòng dở; summary bị từ chối thì attempt vẫn mở để ghi SummaryFailed; khóa lệch schema của contracts.py, status lạ, manifest thiếu trường hay `attempt` không phải số nguyên dương thì ValueError trước khi tạo thư mục
  - Đọc provenance và import hai module không nạp torch (tiến trình sạch), để sweep tính run_id rẻ
  - Mutation: 24/24 mutant bị bắt, mỗi mutant bởi đúng test dự định. Lần chạy đầu cho kết quả giả (bản sao `src/` thiếu `requirements.lock`, nên mọi mutant chết ở cùng một test vì môi trường); đã chép file khóa vào bản sao và in tên test fail để kiểm
  - Trên máy chủ: `env_match: true`; máy "Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz (flags 395ae1858272)"
- Quyết định mới (không cần ADR):
  - `run_id` = 16 ký tự hex đầu của sha256(JSON của [cấu hình chuẩn hóa, code_version, máy]); `code_version` là đủ 64 ký tự hex. Mỗi phần của hash `code_version` đi kèm độ dài, đường dẫn tính từ thư mục chứa gói (`dfl/...`)
  - Thư mục bị ignore được kiểm từng file bằng `git status --porcelain -z --ignored --untracked-files=all` (tài liệu git: chế độ traditional cùng `-uall` hiện từng file); đuôi `.yml` cũng tính, chặt hơn SPEC một chút
  - `Provenance` (code_version, commit, dirty, env_match, versions, máy, các gói lệch) tính một lần mỗi tiến trình; `build_manifest` nhận thêm các giá trị của lượt chạy (T11b) và kiểm các dãy theo node có đúng N phần tử
  - RunLogger mở file theo từng lần ghi (không giữ file mở giữa các lời gọi); manifest và summary ghi qua file tạm rồi `os.replace`
- Câu hỏi mở: có thêm `dfl.config`, `dfl.io.manifest`, `dfl.io.run_logger` vào contract `pure-no-torch` không (thắt chặt; sửa pyproject, hỏi trước)? Hiện có test bằng tiến trình con
- Việc tiếp theo: lô T22a, T02b, T09a, T03, T11a đã xong. P3.5 (T13) áp dụng khi T11c, T11d, T12, T22b, T09c xong

## 2026-10-02 — Phiên #17 — T09c Thang model M0–M3 (TV1)
- Trạng thái: T09c ✅
- File đã đổi: src/dfl/learning/models.py, tests/unit/test_models.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có `MODEL_CLASSES`, `build_model`) → GREEN; 23 test trong `test_models.py`; `make check-task` xanh
  - Số tham số 7,850 / 46,730 / 199,210 / 1,663,370 và byte state = 4 × số tham số (bảng ADR-011); không BatchNorm/Dropout, không buffer; đầu ra (B, 10); tạo model không đổi `torch.get_rng_state()`; cùng seed cùng w⁰, khác seed khác ở mọi tensor; `build_model("smallcnn")` cho đúng w⁰ của `make_smallcnn` (lõi không đổi); tên model khớp `config.MODELS`
  - Oracle kiến trúc: test tự viết forward bằng `torch.nn.functional` theo mô tả ở ADR-004/ADR-011 và so với model. Test này được thêm sau khi mutant "bỏ ReLU giữa fc2 và fc3 của M2" sống sót (số tham số và shape không đổi); giờ mutant đó bị bắt. Mutation: 6/6 bị bắt
- Quyết định mới (không cần ADR): tên lớp `LogisticRegression`, `TwoLayerMLP`, `McMahanCNN`; `MODEL_CLASSES` ánh xạ tên của cấu hình sang lớp; `make_smallcnn` giữ lại cho code và test cũ
- Lệch quy trình: `src/dfl/config.py` có trong danh sách file nhưng không phải sửa: trường `model` và danh sách `MODELS` đã có từ T02b
- Câu hỏi mở: không
- Việc tiếp theo: T11b

## 2026-10-02 — Phiên #17 (tiếp) — T11b `python -m dfl run` + smoke e2e (TV2)
- Trạng thái: T11b ✅ (trừ ý "make check-inv xanh": còn thiếu T-INV-13, thuộc T22b của cùng lô)
- File đã đổi: src/dfl/sim/run.py, src/dfl/cli.py, tests/e2e/test_smoke_run.py, tests/integration/test_seeds.py (mới), tests/unit/test_cli.py, Makefile, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có `dfl.sim.run`) → GREEN; `make check-task` → 516 passed, coverage 97.5% (`sim/run.py` 99%, dòng còn lại là guard runtime processes, thay ở T22b)
  - ENG-4: `python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke` xong trong 8.2 s (ngân sách 60 s); đủ 4 file đúng schema của contracts.py; `rounds.jsonl` 6 dòng (round 1…6, local_steps 2…12); manifest có `cpu_threads` = 1 = `threads`, cấu hình chuẩn hóa, run_id trùng tên thư mục, spectral gap 2/3 (ring N = 4, MH), π = 1/4; message 8/round và byte = message × 4 × 46,730; các pha của timing.jsonl cộng lại đúng khoảng thời gian giữa hai lần ghi. Accuracy tăng từ 0.15 lên khoảng 0.3, mất mát huấn luyện giảm đều qua 6 round
  - T-INV-5: cùng seed thì `rounds.jsonl` trùng từng byte; RNG toàn cục của torch không đổi sau một lượt; khác seed thì khác cách chia và khác w⁰ ở mọi tensor; cùng seed thì cùng cách chia, vị trí, w⁰, luồng batch và dữ liệu node ở 11 biến thể (FC, ER, gossip, ER-gossip, star, local_only, server, uniform, size-corrected, prox, τ = 1); mọi cách chia (IID, Dirichlet, quantity, #C = 2 với N = 10) cố định theo seed và phủ đủ dữ liệu; vị trí node đúng bằng hoán vị của luồng placement; mọi node cùng w⁰ nhưng bộ nhớ riêng, luồng batch đúng (BATCH, i)
  - CLI: trong tiến trình sạch, `run` đọc cấu hình, đặt OMP/OPENBLAS/MKL_NUM_THREADS = `threads`, rồi mới nạp sim.run, lúc torch chưa được nạp; thiếu `--out` thì thoát mã 2
  - Mutation trên `sim/run.py`: 6/6 mutant bị bắt đúng test (bỏ gán vị trí, w⁰ riêng từng node, sai luồng batch, sai luồng partition, combine_s tính cả train và eval, spectral gap luôn None)
  - `make check-inv`: chỉ còn thiếu T-INV-13 (T22b); 68 test T-INV hiện có đều xanh. `make check-full` vì vậy đỏ ở đúng bước này cho tới T22b
- Quyết định mới (không cần ADR):
  - `sim/run.py` lắp được mọi giá trị hợp lệ của schema cấu hình hiện tại (mọi topology kể cả server và local_only; MH, uniform, size_weighted, size_corrected), không chỉ ring/MH. T11c còn thêm B1 (cần trường cấu hình mới), profile fast và test của baseline
  - π: MH 1/N; uniform (d_j + 1) trên đồ thị tĩnh; gossip 1/N (mỗi W^(t) là ma trận matching, đối xứng); size-weighted, size-corrected, server n_j/n. Spectral gap: W tĩnh và ma trận hiệu dụng của server; None với gossip và local_only
  - Đánh giá cuối trên đủ tập của `eval.split` (REQ-03) làm luôn ở đây, vì SummaryOk cần các trường final_*. T11e vẫn còn: không nạp ảnh test khi split là validation, và `test_final_eval.py`
  - Mỗi node dựng model từ một Generator mới của luồng init (cùng w⁰, bộ nhớ riêng), không dùng deepcopy
  - Ghi ra `<out>/runs/<run_id>/attempt-<k>/` với k = số lớn nhất đã có + 1; `--out` bắt buộc cho tới T11d (ghi vào results/)
  - timing.jsonl: train_s và eval_s đo quanh trainer và evaluator; combine_s là phần còn lại của khoảng giữa hai lần ghi
  - Makefile: `check-full: check-static check-inv` theo bảng kích hoạt của CONSTRAINTS.md (dòng T11)
- Ghi chú cho T13: trên smoke, đánh giá chiếm khoảng 0.5 s mỗi round, nhiều gấp 7 lần huấn luyện (4 node + w̄ trên 500 ảnh). Với MNIST, 21 model × 2,000 ảnh × 40 mốc có thể chiếm phần lớn thời gian của một lượt; cần đo ở T13
- Câu hỏi mở: không
- Việc tiếp theo: T05a

## 2026-10-02 — Phiên #17 (tiếp) — T05a `dfl describe`: dữ liệu (TV1)
- Trạng thái: T05a ✅
- File đã đổi: src/dfl/sim/describe.py, src/dfl/cli.py, configs/reference/kh_js_table.yaml, tests/integration/test_describe.py, tests/unit/test_req02_check.py (mới), tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có `dfl.sim.describe`) → GREEN, 17 test; `make check-task` xanh, `describe.py` 100%
  - Tích hợp trên dữ liệu tổng hợp (N = 10, 3 seed): đủ 10 cách chia × 3 seed, ma trận node × lớp 10 × 10, JSON đọc lại không có NaN; mỗi seed dùng đúng luồng partition rồi placement như `sim.run` (ma trận trùng lượt chạy cùng seed); CSV một hàng trung bình mỗi cách chia, theo thứ tự hàng của bảng KH §3.2; N < 10 thì #C = k bị bỏ qua và ghi lý do (G23); `--check-req02` trên dữ liệu tổng hợp trả mã 1 và vẫn ghi file
  - Unit (số giả): khớp bảng trong sai số thì đạt; lệch 12% (> 10%), lệch tuyệt đối 0.0025 (> 0.002), sai thứ tự Dirichlet, sai thứ tự #C, thiếu một cách chia đều báo đúng tên; bảng YAML hỏng (`1e-3` là chuỗi với PyYAML, list, NaN) bị từ chối
  - **REQ-02 trên MNIST thật** (máy chủ, N = 20, 20 seed, 55k ảnh): `python -m dfl describe --config configs/base.yaml --seeds 20 --out build/describe --check-req02` → mã 0 ("REQ-02: đạt"), 6.9 s. JS (bit) đo / bảng KH §3.2: IID 0.0006 / 0.001; α100 0.0018 / 0.002; α1 0.176 / 0.185; α0.5 0.276 / 0.283; α0.1 0.562 / 0.553; α0.05 0.636 / 0.631; #C2 0.614 / 0.613; #C1 0.758 / 0.758; quantity β = 0.5 0.0032 / 0.003. Số lần lấy mẫu lại trung bình: α0.1 0.35, α0.05 2.95, quantity 12.9 (KH ghi 1.1, 4.0, 11). P2' (shards): JS 0.594, CV 0 (kích thước đều)
- Quyết định mới (không cần ADR):
  - Một `describe.json` (mọi cách chia × seed, kèm cấu hình chuẩn hóa) và một `describe.csv` (trung bình qua seed); lệch mục tiêu với π = 1/N như bảng KH §3.2; seed 0 … S − 1
  - Khóa cách chia: `iid`, `dirichlet_<α>`, `label_<k>`, `quantity_0.5`, `shards`; `--table` mặc định `configs/reference/kh_js_table.yaml`
- Lệch quy trình: thêm `tests/unit/test_req02_check.py` cho hàm so sánh ("hàm so sánh có unit test với số giả"); đặt tên khác `test_describe.py` vì pytest không cho hai module test trùng tên gốc
- Câu hỏi mở: không
- Việc tiếp theo: T11c

## 2026-10-02 — Phiên #17 (tiếp) — T11c Baseline, biến thể trộn, profile fast (TV2)
- Trạng thái: T11c ✅
- File đã đổi: src/dfl/config.py, src/dfl/sim/run.py, configs/base.yaml (chỉ comment), configs/profiles/fast.yaml (mới), tests/integration/test_baselines.py (mới), tests/unit/test_config.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED: 3 test của B1 và 7 test cấu hình fail trước khi sửa. Các test của B2, B3, uniform, size-corrected xanh ngay, vì T11b đã lắp sẵn các đường này; ở đây chúng là test đặc tả hành vi đã có
  - GREEN: `make check-task` → 547 passed, coverage 97.6%. Lần chạy này mất 109 s, vượt ngân sách 90 s (chỉ cảnh báo), vì chạy cùng lúc hai lượt demo trên máy chủ đang tải 16.7/8 lõi; lần trước là 64 s. Test tích hợp chạy lượt thật (test_seeds, test_baselines) chiếm khoảng 50 s; xem lại ở T13
  - B1, B1-4k (`topology.kind: central`, N = 1): mốc đánh giá là các bội của 100 tới hết ngân sách của chính lượt đó (200 và 300 trên dữ liệu tổng hợp); 0 message, 0 byte; π = [1], spectral gap None; node giữ đủ 1,800 ảnh train theo đúng thứ tự gốc; minibatch từ luồng B1_BATCH
  - B2 (local_only): 0 message, consensus distance > 0 ở mọi mốc (W = I), π = 1/N, spectral gap None. B3 (server): 2N = 8 message/round, byte = message × 4 × 46,730, consensus distance = 0, π = n_j/n, spectral gap 1. Uniform trên star N = 4: π = (0.4, 0.2, 0.2, 0.2), 6 message/round. Size-corrected với quantity β = 2: c_i = n_i/n̄ và π = n_j/n
  - Cấu hình: `central` ⇔ N = 1; central đòi partition iid và mixing MH, để hai cấu hình tương đương không cho hai JSON khác nhau; size_corrected đòi μ = 0 (ADR-005); profile fast đúng SPEC §5.2 (MNIST, N = 20, τ·R = 1,000, τ = 10, đánh giá mỗi 100 bước)
  - **Demo C3 trên MNIST thật** (máy chủ, `--out build/fast`, IID, profile fast; hai lượt chạy song song khi máy đang tải 16.7 trên 8 lõi):
    - ring: accuracy cuối trên 10,000 ảnh test trung bình 0.9797, node kém nhất 0.9755, acc(w̄) 0.9811; spectral gap 0.0326; 4,000 message (40/round); 7 phút 23 giây; RAM tối đa 858 MB
    - FC: 0.9816 ở mọi node (FC + MH cho cùng một model); spectral gap 1.0; 38,000 message (380/round); 7 phút 41 giây
    - Learning curve (tập con 2,000 ảnh) tăng từ 0.90–0.92 ở 100 bước lên 0.984–0.986 ở 1,000 bước. Mỗi 100 bước: huấn luyện khoảng 19 s (khoảng 9.5 ms một bước), đánh giá khoảng 10 s. Tổng mỗi lượt: huấn luyện 236 s, đánh giá 118–123 s
- Quyết định mới (không cần ADR): B1 và B1-4k là `topology.kind: central` với `num_nodes: 1`; ngân sách 17,200 và 4,000 bước, cùng η riêng của B1, ghi trong YAML thí nghiệm (T14b, T14c), không trong code
- Câu hỏi mở: không
- Việc tiếp theo: T11d

## 2026-10-02 — Phiên #17 (tiếp) — T11d Quy tắc ghi results/, phân kỳ, lỗi (TV3)
- Trạng thái: T11d ✅
- File đã đổi: src/dfl/sim/run.py, src/dfl/io/manifest.py, src/dfl/cli.py, tests/integration/test_results_rules.py (mới), tests/unit/test_manifest.py, tests/unit/test_cli.py, tests/integration/test_baselines.py, tests/integration/test_seeds.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (`run` chưa nhận `results_root`, `provenance`; chưa có `RunRefused`, `run_id_of`, `unassembled_manifest`) → GREEN, 12 test quy tắc; `make check-task` → 562 passed, coverage 97.6%, 66 s
  - ENG-10: vào results/ thì từ chối khi cây bẩn, môi trường lệch (nêu tên gói), dữ liệu tổng hợp (profile smoke), hay run_id đã có attempt ok hoặc diverged; không tạo attempt nào khi từ chối. Ra `--out` thì chạy được, và manifest ghi đúng `dirty`, `env_match`. Sau attempt failed, `run` tay tạo được attempt-2
  - Phân kỳ (η = 1e4 trên dữ liệu tổng hợp): SummaryDiverged có round và lý do; rounds.jsonl chỉ có các round trước round phân kỳ, không có NaN hay Infinity. Ngoại lệ giữa lượt: SummaryFailed có tên lỗi, số message tới lúc lỗi, rồi raise lại. Vượt 1,000 lần lấy mẫu lại: attempt failed trước khi huấn luyện, manifest đủ khóa với các trường của bước lắp để trống, rounds.jsonl rỗng
  - Hai tiến trình: một tiến trình con giữ flock trên `results/runs/<run_id>/.lock`; lượt trong tiến trình test raise RunLocked và không tạo attempt
  - CLI: không có `--out` thì ghi vào results/; từ chối → mã 3, khóa đang bị giữ → mã 4 (sweep dựa vào các mã này)
  - Mutation: 10/10 mutant bị bắt đúng test (bỏ kiểm cây bẩn, bỏ kiểm môi trường, cho smoke vào results/, diverged không tính là xong, failed tính là xong, bỏ flock, không bắt RunDiverged, nuốt lỗi, không ghi lỗi lúc lắp, kiểm "đã xong" ngoài khóa)
- Quyết định mới (không cần ADR):
  - Lượt hỏng trước khi lắp xong vẫn có attempt: `unassembled_manifest` để trống các trường của bước lắp (0, None, [], 0.0, []); lý do nằm ở SummaryFailed. `build_manifest` vẫn kiểm chặt N phần tử
  - `run` tay được chạy lại sau attempt failed; sweep thì chỉ khi có `--retry-failed` (T12)
  - KeyboardInterrupt (BaseException) không bị bắt, nên attempt bị ngắt không có summary và sẽ được chạy lại
  - Kiểm "đã xong" và chọn số attempt nằm trong khóa của run_id
- Bảo trì test: `check-task` đã lên 101 s khi máy tải 14–16. Các test chạy lượt thật (test_results_rules, test_baselines, test_seeds) chuyển sang cấu hình smoke rút gọn (2–3 round, tập con 100 ảnh, B1 100/200 bước); test e2e vẫn chạy đúng profile smoke. `check-task` còn 66 s
- Lệch quy trình: chạm 10 file (gồm `io/manifest.py` và hai file test của T11b, T11c để rút gọn thời gian test)
- Câu hỏi mở: không
- Việc tiếp theo: T12

## 2026-10-02 — Phiên #17 (tiếp) — T12 Sweep (TV3)
- Trạng thái: T12 ✅
- File đã đổi: src/dfl/sim/sweep.py, tests/integration/test_sweep.py (mới), src/dfl/cli.py, src/dfl/contracts.py, tests/unit/test_contracts_smoke.py, src/dfl/config.py, src/dfl/io/manifest.py, src/dfl/io/run_logger.py, src/dfl/sim/run.py, src/dfl/data/datasets.py, tests/unit/test_cli.py, tests/unit/test_datasets.py, tests/unit/test_manifest.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có `ExperimentIndex`, `dfl.sim.sweep`) → GREEN, 33 test của sweep; `make check-task` → 597 passed, coverage 97.7% (`sweep.py`, `manifest.py` 100%)
  - Lưới: khối = tích Descartes các trục × seeds, seed là vòng ngoài; `set` và giá trị trục trộn như `extends`; `--only khóa=giá trị`; 13 dạng YAML sai đều raise ValueError nêu chỗ sai (khóa lạ, không có khối, seeds thiếu hay trùng, trục không phải danh sách hay rỗng, trục không phải khóa cấu hình, seed trong set, ô trùng, ô sai cấu hình, tên nhóm không hợp lệ, khối không phải mapping, profile không phải đường dẫn); cấu hình của ô qua YAML cho tiến trình con đọc lại đúng (cả 1.0e-05)
  - Kế hoạch: attempt ok hay diverged là xong; nếu không thì attempt mới nhất quyết định (chưa có summary là bị ngắt, chạy lại; failed chỉ chạy lại khi có --retry-failed); `--dry-run` in "tổng / đã xong / failed / sẽ chạy" và không ghi gì
  - Chạy thật bằng tiến trình con `python -m dfl run` (ENG-9): 2 worker và 1 worker cho `rounds.jsonl` trùng từng byte; chạy lại thì không chạy gì; chỉ mục có run_id của code_version hiện tại và giữ nguyên mục của code_version khác; lượt bị ngắt chạy lại thành attempt-2, lượt failed chỉ chạy lại với --retry-failed; tiến trình con có code_version khác ghi attempt failed và sweep dừng xếp lượt mới; sweep từ chối cây bẩn kể cả với --out; --out chỉ cảnh báo khi môi trường lệch
  - results/: từ chối khi không chạy từ worktree liên kết, môi trường lệch, ô dữ liệu tổng hợp, hay file thí nghiệm ngoài worktree. Trên một repo tạm (bản sao code hiện tại, đã commit) cùng worktree `grid` có MNIST giả 5,400/1,000 ảnh: hai sweep chạy đồng thời dùng chung ô ring chỉ chạy ô đó một lần (nơi kia nhận `locked` hay `done`), mọi attempt ok; commit chỉ đổi docs/ không đổi code_version, nên chạy lại không chạy gì. Tiến trình con nạp dfl ngoài `--expect-root` thì thoát mã 5
  - Mutation: 16 mutant, đều bị bắt đúng test. Harness lần đầu cho kết quả giả: test truyền `root=ROOT` là repo thật, trong khi code mutant nằm ở bản sao, nên code_version lệch và tiến trình con lượt nào cũng chết vì môi trường; lần thứ hai thì `ignore_patterns("data")` bỏ luôn `src/dfl/data`. Sau khi sửa (chép cả repo, chỉ bỏ thư mục cấp gốc), bản sao không mutant xanh. Mutant "bỏ kiểm worktree" sống sót vì thông báo của phép kiểm vị trí file cũng chứa chữ "worktree"; đã tách thành hai test với thông báo riêng
- Quyết định mới (không cần ADR; định dạng YAML ghi trong docstring của `sim/sweep.py`):
  - YAML thí nghiệm: `experiment` (tên chỉ mục), `profile`, `blocks`; mỗi khối có `seeds`, `set`, `grid` (trục là khóa cấu hình cấp đầu)
  - Tiến trình con nhận `--expect-code-version`, `--expect-root`, `--results`; mã thoát: 0 xong, 1 lỗi, 3 từ chối, 4 đang bị khóa, 5 code khác của sweep. Sweep dừng xếp lượt mới khi gặp 5, hay 3 với run_id chưa xong
  - Chỉ mục ghi lúc bắt đầu (sau các phép kiểm), liệt kê run_id của cả lưới, kể cả khi có --only; mục của code_version khác giữ nguyên
  - Ghi vào results/ thì sweep phải chạy với `PYTHONPATH=<worktree>/src` từ một worktree liên kết, và file thí nghiệm cùng profile phải nằm trong worktree đó (đã commit)
  - Đọc cấu trúc results/ (trạng thái attempt, số attempt kế tiếp, khóa flock) chuyển vào `io/run_logger.py`, không nạp torch; tiến trình cha của sweep không nạp torch
  - Hiệu năng: `dfl.data.datasets` chỉ import torchvision trong hàm đọc hay tải MNIST, vì import torchvision nạp mọi model (2.4 s). Import `dfl.sim.run` giảm từ 5.7 s xuống 2.7 s; một tiến trình con smoke tí hon từ 6.2 s xuống khoảng 3.5 s
- Ngân sách: `make check-task` 104 s, vượt 90 s (chỉ cảnh báo), khi máy tải 17/8 lõi; test sweep chạy khoảng 11 tiến trình con thật (khoảng 30 s). Xem lại ở T13
- Lệch quy trình: chạm 15 file, vượt 5 file của K8: các module nền được mở rộng thêm (config.load_mapping, merge; manifest.provenance_of, is_linked_worktree; run_logger chuyển helper sang; run có --expect-*). Diff khoảng 900 dòng
- Câu hỏi mở: không
- Việc tiếp theo: T22b

## 2026-10-02 — Phiên #17 (tiếp) — T22b Process runtime: mỗi node một tiến trình, cộng observer (TV2)
- Trạng thái: T22b ✅
- File đã đổi: src/dfl/sim/process_runtime.py, tests/integration/test_process_runtime.py (mới), src/dfl/sim/run.py, src/dfl/sim/observer.py, src/dfl/contracts.py (TimingRecord.comm_s), src/dfl/config.py, src/dfl/cli.py, src/dfl/sim/sweep.py, tests/unit/test_contracts_smoke.py, tests/unit/test_config.py, tests/unit/test_run_logger.py, tests/unit/test_cli.py, tests/integration/test_sweep.py, tests/e2e/test_smoke_run.py, tasks/todo.md, tasks/SESSION_LOG.md
- Kiểm chứng:
  - RED (chưa có `dfl.sim.process_runtime`, `comm_s`) → GREEN; `make check-task` → 606 passed, coverage 95.2%; **`make check-inv` xanh đủ 13 T-INV** (72 test); **`make check-full` xanh** (610 test, `pip check` sạch), lần đầu kể từ T11b
  - **T-INV-13**: với smoke rút gọn N = 4 (2 round) trên ring, gossip và FC, `rounds.jsonl` của runtime processes trùng từng byte với runtime inprocess; `summary.json` trùng toàn bộ; manifest trùng ở node_class_counts, objective_weights, spectral_gap, partition_retries. Chạy tay `python -m dfl run --config configs/profiles/smoke.yaml --runtime processes --out build/smoke-proc` (profile smoke đầy đủ, 6 round): `cmp` với bản inprocess trùng từng byte; mất 12.0 s so với 8.0 s
  - T-INV-6 ở N = 4: ring 8 message/round, byte = message × 4 × 46,730, nên kênh tới observer không bị đếm; timing có `comm_s` (5–12 ms mỗi khoảng trên smoke; 0 ở inprocess)
  - Phân kỳ (η = 1e4): mọi tiến trình dừng gọn ở cùng `diverged_at_round` với bản inprocess, cùng message và byte; rounds.jsonl trùng
  - Node không khởi động được (code_version khác) → lượt dừng ngay với NodeFailed và SummaryFailed, không đợi hết thời gian chờ của gloo
  - Kiểm tay (không thành test vì dễ chập chờn): giết node 1 bằng SIGKILL giữa lượt → observer nhận "Connection closed by peer" ngay, ghi SummaryFailed (24 message tới lúc lỗi), không còn tiến trình con nào. Lần thử đầu giết nhầm shell bash vì `pgrep -f` khớp cả chuỗi lệnh của chính shell; đã chuyển sang đọc /proc/<pid>/task/*/children
  - Mutation (chép cả repo để tiến trình con cũng nạp code mutant): 4/4 bị bắt (cộng byte bằng max, drift tính trên state sau tổng hợp, node không dừng khi phân kỳ, lệch round phân kỳ). Mutant "node không dừng khi phân kỳ" làm lượt treo tới hết thời gian chờ, lộ ra rằng observer đợi tiến trình con tới hết `timeout` (600 s); nay chỉ đợi tối đa 60 s sau lần gom cuối rồi kill (NodeFailed)
- Quyết định mới (không cần ADR, khớp ADR-010):
  - Tiến trình chạy lượt kiêm observer (rank N) và là nơi duy nhất ghi kết quả; nó khởi chạy N tiến trình con `python -m dfl node --rank i --world-size N+1 --init-method file://… --config …`, tức N + 1 tiến trình. Rendezvous qua file:// trong thư mục tạm (không tranh cổng TCP); node báo sẵn sàng bằng file `ready-<rank>` trước khi vào process group
  - Mỗi round có hai lần `all_gather` [mã, message, byte, tải], ứng đúng hai chỗ engine kiểm hữu hạn; lý do phân kỳ lấy node id nhỏ nhất theo thứ tự ưu tiên của engine (mất mát, rồi tham số sau Adapt, rồi sau tổng hợp)
  - Ở mốc đánh giá, node gửi observer state sau tổng hợp, snapshot trước tổng hợp, mất mát và thời gian các pha bằng send/recv. Timing của processes: train_s, comm_s, combine_s lấy lớn nhất qua node
  - `run.assemble` / `Assembly` là phần lắp dùng chung cho hai runtime; `run.build` dựng trên nó; `_attempt` chạy qua một driver (RunSetup hay ProcessRun), nên quy tắc results/, khóa và summary dùng chung. `observer.is_due` là hàm dùng chung cho mốc đánh giá
  - Config từ chối `runtime: processes` với topology server và central; `run --runtime` ghi đè runtime của cấu hình trước khi tính run_id (SPEC §4); `config.to_yaml` dùng chung cho sweep và process runtime
- Coverage: `process_runtime.py` báo 68%, vì thân node chạy trong tiến trình con và không được đo (như `torchdist.py`). Muốn đo thì bật đo coverage cho tiến trình con, tức sửa cấu hình coverage trong pyproject (thắt chặt; hỏi trước)
- Ngân sách: `make check-task` 108 s khi máy tải 18/8 lõi (chỉ cảnh báo); test process runtime khoảng 35 s
- Lệch quy trình: chạm 16 file (run.py tái cấu trúc, observer.py, config.py, sweep.py dùng `to_yaml`); diff lớn, khoảng 800 dòng
- Câu hỏi mở: bật coverage cho tiến trình con không? Có tách các test chạy tiến trình con (sweep, process runtime; khoảng 65 s) ra khỏi check-task không (sửa CONSTRAINTS/Makefile, hỏi trước)?
- Việc tiếp theo: lô T09c, T11b, T05a, T11c, T11d, T12, T22b đã xong. E0a (T14b, P3.6) còn chờ T11e, T13, T14a, T14e, checkpoint C3 và Gate G1

## 2026-10-02 — Phiên #17 (tiếp) — P3.7 Rà soát kiểm thử ở C3 (subagent test-engineer)
- Trạng thái: báo cáo xong (`docs/decisions/REVIEW-C3-P3.7.md`); không sửa code
- File đã đổi: docs/decisions/REVIEW-C3-P3.7.md (mới), tasks/SESSION_LOG.md
- Cách làm: đo coverage dòng và nhánh trên 611 test (tổng 94.6%). Subagent test-engineer rà soát ở chế độ chỉ đọc, đối chiếu T-INV-1…12 (KH §5.2, SPEC §9.2) và CONSTRAINTS.md theo `testing-patterns.md`, cài mutant trong bộ nhớ bằng plugin pytest. Tiến trình chính tự chạy lại các khẳng định quan trọng nhất: probe P0-1, probe P2-1, hai mutant ghim seed (28 passed mỗi mutant), grep các trường tổng hợp, Makefile, SPEC REQ-04
- Kết quả: P0 = 4, P1 = 8, P2 = 14, cùng 15 test RED cho T15b. 18 mutant mới sống sót qua test hiện có; mỗi mutant bị một test đề xuất giết (chạy dạng script)
  - **Lỗi thật, đã tái hiện (P0-1):** process runtime không gọi `check_union_connected`. Cùng cấu hình gossip có hợp các round không liên thông: inprocess raise ValueError (INV-4), còn processes ghi `ok`, nên T-INV-13 lệch ở các cấu hình như vậy. Lưới N = 20 (inprocess) không bị ảnh hưởng; E8 và demo thì có. Lỗi nằm trong code T22b
  - Khoảng trống P0: dây seed của 5 luồng trong `sim/run.py` không được test với seed khác 0, và test "khác seed thì khác cách chia" qua được chỉ nhờ hoán vị vị trí node; các trường F1, acc^loc, acc^eff tổng hợp của RoundRecord không được kiểm giá trị; thứ tự `node_class_counts` theo id node không được kiểm
  - Lỗi tiềm ẩn (P2): `mixing.uniform` đọc thuộc tính cạnh `weight`; `size_weighted` nhận sizes NaN; thông báo khi cạn số lần lấy mẫu lại nêu sai nguyên nhân; `confusion_matrix` broadcast khi độ dài lệch
  - Quy trình: `MIN_TESTS` vẫn là 34 trong khi có 611 test (P1-8, việc của C3)
- Lệch quy trình: không
- Câu hỏi mở: sửa P0-1 theo P3.4 ngay không (P3.7 không cho sửa code); config có nên từ chối ring với N < 3 không (P2-6)
- Việc tiếp theo: chờ nhóm duyệt báo cáo; đề xuất sửa P0-1, rồi thêm test P0-2…P0-4 và P1

## 2026-10-02 — Phiên #17 (tiếp) — P3.8 Checkpoint C3 (M3)
- Trạng thái: các bước của checkpoint đã chạy; **chờ nhóm duyệt** (chưa đạt, xem dưới)
- File đã đổi: Makefile (`MIN_TESTS` 34 → 611), CONSTRAINTS.md (giá trị đo ở bảng "Đo và giữ", đúng quy tắc "cập nhật ở mỗi checkpoint"), tasks/plan.md (§13, dòng C3), docs/decisions/REVIEW-C3-P4.1.md (mới), tasks/SESSION_LOG.md
- Bước 1: `make check-full` ở 2adb226 xanh: sàn, ruff, format, import-linter 5/5, `check-inv` 13/13 T-INV (72 test), 611 test, coverage 95.16% dòng, `pip check`; 3 phút 12 giây
- Bước 2: C1 và C2 chưa từng chạy (T05b, T08b chưa xong), nên danh sách tính từ Gate G2 (59ea6a7). 25 commit task: T02a, T04a, T04b, T02c, T04c, T07, T06, T08, T10a, T10b, T09b, T10c/T10d, T22a, T02b, T09a (2 commit), T03, T11a, T09c, T11b, T05a, T11c, T11d, T12, T22b, P3.7
- Bước 3 (MNIST thật, máy chủ, `dirty: false`, `env_match: true`; cấu hình ở build/c3/):
  - demo fast IID trên test: ring trung bình 0.9797 (thấp nhất 0.9755, acc(w̄) 0.9811), gap 0.0326, 4,000 message, 7.2 phút; FC 0.9816 ở mọi node, gap 1.0, 38,000 message, 7.3 phút. Trùng đúng con số của demo ở T11c
  - pilot chỉ đánh giá trên validation (5,000 ảnh), ngoài lưới: B1 (central, 1,000 bước, η 0.05) 0.9626 trong 31 s; DFedAvg-FC-IID (fast) 0.9794. Không thấp bất thường. B1 ở đây chỉ chạy 1,000 bước (khoảng 0.58 epoch) nên chưa so được với SAN-1 hay SAN-2
- Bước 4: subagent code-reviewer (chỉ đọc) review 319a604..HEAD và canh gác CONSTRAINTS trên 59ea6a7..HEAD: Critical 0, Important 6, Suggestion 12. Tiến trình chính đã kiểm lại trên code I1, I4, S1, S5, S8. Không có gì bị nới trong CONSTRAINTS hay Makefile
- Bước 5, đối chiếu DoD (H§4.2):
  - Đạt: lint/format; không regression; chạy thật được (smoke, demo, pilot, cả hai runtime); manifest đầy đủ; không đổi siêu tham số hay ngưỡng; bảo mật ở phạm vi hẹp; log có cấu trúc; rollback bằng `git revert`
  - Chưa đạt:
    - ca biên và đường lỗi: P0-1 và các khoảng trống của P3.7;
    - logic lặp: vòng ATC viết lại ở process runtime (I6);
    - phạm vi commit: 5 commit vượt 300 dòng hay 5 file;
    - tài liệu: README vẫn ghi "chưa có code chạy được"; ADR-010 mô tả launcher và Docker khác với code; contracts ghi `objective_shift_tv` so với phân bố test (S8)
  - Mục C3 trong todo còn thiếu: T11e chưa làm; smoke trên laptop demo (QS-8) chưa chạy (cần laptop của nhóm)
- Câu hỏi mở (cần nhóm): duyệt C3 có điều kiện hay đợi sửa xong P0-1, I1, I6, S1 và thêm test P0 của P3.7? Các quyết định I3 (chuỗi máy và chỉ mục, ADR-008), I4 (Docker), I5 (tải máy trong manifest; workers của E8), S2 (CodeMismatch), S8 (định nghĩa lệch mục tiêu), S9 (fast đánh giá trên validation), S12 (định nghĩa train_loss)
- Việc tiếp theo: dừng chờ nhóm duyệt C3

## 2026-10-02 — Phiên #16–#17 (kết thúc) — Ranh giới phiên (P3.9)
- Trạng thái:
  - Xong (mỗi task một commit):
    - phiên #16: T22a ✅ (48fbff4), T02b ✅ (191290d), T09a ✅ (e2c59f3, bổ sung test 0e02a97), T03 ✅ (c13686f), T11a ✅ (99fc7d5);
    - phiên #17: T09c ✅ (7a42ee6), T11b ✅ (a973a08), T05a ✅ (06965ce), T11c ✅ (26ab64a), T11d ✅ (692a2a8), T12 ✅ (7e611a6), T22b ✅ (8e807f4);
    - rà soát: P3.7 ✅ (2adb226, `docs/decisions/REVIEW-C3-P3.7.md`).
  - Đang chờ: checkpoint C3 ⏳ **chờ nhóm duyệt** (bcde002; `docs/decisions/REVIEW-C3-P4.1.md`, `tasks/plan.md` §13).
  - Không task nào đang làm dở. Lỗi P0-1 (process runtime không kiểm INV-4 hợp các round) mới được ghi nhận, chưa sửa: P3.7 không cho sửa code, và C3 đang chờ nhóm quyết.
- File đã đổi trong phiên:
  - `src/dfl`: sim/{torchdist, run, describe, sweep, process_runtime}.py (mới); config.py, cli.py, `__main__.py`, data/datasets.py, io/{manifest, run_logger}.py, learning/{models, local_train}.py (mới); contracts.py (`ExperimentIndex`, `TimingRecord.comm_s`), sim/observer.py (`is_due` dùng chung).
  - Test: unit (config, datasets, cli, manifest, run_logger, models, local_train, req02_check, torchdist_codec, contracts_smoke); integration (torchdist, seeds, baselines, results_rules, describe, sweep, process_runtime); e2e (smoke_run).
  - configs: base.yaml, profiles/{smoke, fast}.yaml, reference/kh_js_table.yaml.
  - Khác: Makefile (`check-full` gồm `check-inv`; `MIN_TESTS` 611), CONSTRAINTS.md (chỉ cập nhật giá trị đo ở bảng "Đo và giữ" lúc checkpoint), tasks/{todo, plan, SESSION_LOG}.md, docs/decisions/REVIEW-C3-{P3.7, P4.1}.md.
- Trạng thái git:
  - Mọi thay đổi của phiên đã được commit trên `main`, HEAD = bcde002. Không có stash.
  - Cây làm việc chỉ còn `docs/đath1.docx` bị sửa. Người khác sửa file này từ 2026-10-01 17:46 (55 KB → 297 KB), trước phiên này; cố ý không commit, chờ người sửa hay nhóm quyết. `docs/đath1_revised.docx` đã commit từ P1.4 và không đổi.
  - Thư mục bị ignore, không commit: `data/` (MNIST, 64 MB, tải bằng `python -m dfl download mnist`); `build/` (1.9 MB: describe, fast, c3, smoke, smoke-proc, review). Không có `results/` nào: chưa chạy lượt nào vào results/.
  - Không còn tiến trình `dfl` nào chạy nền.
- Kiểm chứng:
  - `make check-full` ở bcde002 → mã 0: "Sàn: đạt (611 test)", ruff, format, import-linter 5/5, `check-inv` 13/13 T-INV (72 test), 611 passed, coverage 95.16%, `pip check` sạch; 3 phút 25 giây.
  - Kiểm tay trên máy chủ:
    - `describe --check-req02` trên MNIST (N = 20, 20 seed) → mã 0;
    - demo fast ring/FC IID cho 0.9797/0.9816 trên test;
    - pilot B1 và FC-IID chỉ trên validation cho 0.9626/0.9794;
    - smoke processes trùng từng byte với inprocess;
    - giết một node giữa lượt thì lượt ghi SummaryFailed, không treo.
- Quyết định mới, không cần ADR (chi tiết ở mục của từng task):
  - B1 là `topology.kind: central`;
  - định dạng YAML thí nghiệm (`experiment`, `profile`, `blocks`, `seeds`, `set`, `grid`);
  - mã thoát của `run`: 0/1/3/4/5;
  - quy tắc results/ (khóa flock, `unassembled_manifest`);
  - process runtime: tiến trình chạy lượt kiêm observer, hai lần gom mỗi round, `comm_s`;
  - torchvision chỉ được import khi đọc MNIST;
  - `MIN_TESTS` = 611.
- Quyết định cần ADR hay sửa SPEC (Ask first, chưa làm):
  - I3: chuỗi máy chỉ hash cờ ISA; khóa chỉ mục theo (code_version, máy) — sửa ADR-008;
  - S2: CodeMismatch không ghi vào run_id của code mới — ADR-008;
  - I4: Docker chưa có, và ADR-010 mô tả launcher khác code — ADR-010;
  - I5: tải máy chủ trong manifest như ADR-011 đòi — sửa schema trước Tier 1;
  - S8: định nghĩa `objective_shift_tv` (so với phân bố test hay train gộp) — contracts/SPEC.
- Câu hỏi mở:
  1. Duyệt C3 có điều kiện, hay đợi sửa P0-1, I1, I6, S1 và thêm test P0 của P3.7?
  2. Các mục I3, I4, I5, S2, S8, S9 (fast đánh giá trên validation), S12 (định nghĩa `train_loss_mean`), P2-6 (config từ chối ring N < 3).
  3. Tách các test chạy tiến trình con (khoảng 65 s) khỏi `check-task`? Sửa CONSTRAINTS/Makefile, hỏi trước.
  4. Bật coverage cho tiến trình con (sửa pyproject)?
  5. Thêm `dfl.config`, `dfl.io.manifest`, `dfl.io.run_logger`, `dfl.sim.sweep` vào contract `pure-no-torch`?
  6. Q10, Q11 (người dùng quyết sau); Gate G1 (GVHD duyệt SPEC và ADR-001…011).
- Việc tiếp theo:
  1. Nhóm duyệt C3 (cả smoke trên laptop demo, QS-8).
  2. Nếu được duyệt hướng "sửa trước": P0-1 theo P3.4 (preflight dùng chung với engine, I6), rồi I1, S1, các test P0 và P1 của P3.7, README.
  3. T11e (tập đánh giá + đánh giá cuối), rồi T15a, T15b (R@x: dùng 15 test RED ở REVIEW-C3-P3.7 §5), T14a, T14e, T13 (P3.5).
  4. Sau đó G1, rồi E0a theo P3.6 (24 lượt, lớn hơn 10, nên trình `--dry-run` cho nhóm trước khi chạy).

## 2026-10-02 — Phiên #18 — Nhóm duyệt C3
- Trạng thái: Checkpoint C3 ✅ nhóm duyệt (người dùng trả lời "duyệt" cho câu hỏi duyệt C3 có điều kiện hay đợi sửa xong)
- File đã đổi: tasks/plan.md (§13, dòng C3), tasks/todo.md (đánh dấu C3, ghi các mục còn mở), tasks/SESSION_LOG.md
- Các mục còn mở, làm trước E0a: T11e; smoke trên laptop demo (QS-8); sửa P0-1 (process runtime thiếu kiểm INV-4 hợp các round), I1 (`--out` vào results/), I2 (results/ trong worktree), I6 (preflight dùng chung với engine), S1 (thứ tự ghi summary ở process runtime); thêm test P0 và P1 của P3.7; cập nhật README
- Chưa đụng tới (cần nhóm quyết): I3, I4, I5, S2, S8, S9, S12, P2-6
- Việc tiếp theo: sửa P0-1 theo P3.4 (Prove-It FAIL trước)

## 2026-10-02 — Phiên #18 (tiếp) — Sửa P0-1: process runtime thiếu các phép kiểm trước round đầu (P3.4)
- Trạng thái: P0-1 ✅ (REVIEW-C3-P3.7 P0-1, REVIEW-C3-P4.1 I6 phần preflight)
- File đã đổi: src/dfl/sim/engine.py, src/dfl/sim/process_runtime.py, tests/integration/test_process_runtime.py, tasks/SESSION_LOG.md
- **Nguyên nhân gốc (một câu):** process runtime tự viết lại vòng ATC mà không gọi các phép kiểm trước round đầu của engine (`_check_nodes` lúc dựng Engine và `check_union_connected` trong `check_schedule`), nên lượt gossip có hợp các round không liên thông vẫn chạy ở runtime processes trong khi engine từ chối
- Tái hiện → khoanh vùng → rút gọn: smoke rút gọn (2 round), gossip, seed 1 (round 0 và 1 cùng một matching trên K₄): inprocess raise ValueError, processes ghi `ok`
- Canh gác: `test_t_inv_13_processes_refuse_a_gossip_schedule_whose_union_never_connects`. Trước khi sửa: FAIL ("DID NOT RAISE ValueError" ở runtime processes). Sau khi sửa: PASS. Hai runtime ghi SummaryFailed trùng nhau, cùng lý do
- Sửa: thêm `engine.preflight(nodes, schedule, total_rounds)` (`_check_nodes` cộng INV-4 theo hợp các round) và `engine.check_schedule_union`, dùng chung cho `Engine.check_schedule`. `ProcessRun.execute` chạy preflight trên các node mà tiến trình chạy lượt tự lắp, trước khi spawn; mỗi tiến trình node cũng tự kiểm INV-4 trước khi báo sẵn sàng (cho các lần khởi chạy từ bên ngoài, như Docker ở T22c)
- Kiểm chứng: `make check-full` → mã 0: 612 passed, `check-inv` 73 test (13 T-INV), coverage 95.14%, `pip check` sạch
- Còn lại của I6 (chưa làm): vòng round của node vẫn là bản viết riêng; lý do phân kỳ của processes vẫn ở dạng chung, không có tên tensor hay giá trị mất mát như engine
- Việc tiếp theo: S1 (thứ tự ghi summary ở process runtime)

## 2026-10-02 — Phiên #18 (tiếp) — Sửa S1: process runtime ghi summary trước khi node thoát (P3.4)
- Trạng thái: S1 ✅ (REVIEW-C3-P4.1)
- File đã đổi: src/dfl/sim/process_runtime.py, tests/integration/test_process_runtime.py, tasks/SESSION_LOG.md
- **Nguyên nhân gốc (một câu):** `_observe` ghi SummaryOk rồi `execute` mới đợi node thoát. Vì vậy một node chậm thoát làm `_attempt` ghi summary lần hai ("finish sau finish"), và ở đường phân kỳ NodeFailed thay chỗ RunDiverged
- Canh gác: `test_a_node_lingering_after_the_last_round_does_not_fail_a_finished_run` và `test_a_node_lingering_after_divergence_keeps_the_run_diverged`, giả lập node không thoát bằng cách thay `ProcessRun._wait_exit`. Trước khi sửa: cả hai FAIL ("RuntimeError: finish sau finish"). Sau khi sửa: PASS
- Sửa: `_observe` chỉ trả SummaryOk về; `execute` đợi node thoát (`_settle`) rồi mới `finish`. Sau khi mọi round đã được quan sát, node chậm thoát hay thoát lỗi chỉ in cảnh báo; node còn sống bị kill
- Kiểm chứng: `make check-full` → mã 0 (614 passed)
- Việc tiếp theo: I1 và I2 (quy tắc ghi results/ khi dùng `--out`, và results/ trong worktree)

## 2026-10-02 — Phiên #18 (tiếp) — Sửa I1: `--out` dẫn vào results/ bỏ qua mọi quy tắc (P3.4)
- Trạng thái: I1 ✅ (REVIEW-C3-P4.1)
- File đã đổi: src/dfl/sim/run.py, src/dfl/sim/sweep.py, src/dfl/cli.py (chữ trợ giúp của `run --results`), tests/integration/test_results_rules.py, tests/integration/test_sweep.py, tasks/todo.md, tasks/SESSION_LOG.md
- **Nguyên nhân gốc (một câu):** nhánh `--out` của `run` và của `sweep` không đối chiếu đích ghi với thư mục kết quả, nên `--out results` (hay một symlink, hay `..` dẫn tới đó) ghi thẳng vào results/ mà bỏ qua mọi quy tắc của nó: dữ liệu smoke, cây bẩn, môi trường, khóa, run_id đã xong
- Tái hiện: `python -m dfl run --config configs/profiles/smoke.yaml --out results`, chạy trong thư mục tạm để không đụng results/ thật. Trước khi sửa, các test Prove-It dưới đây báo "DID NOT RAISE"; sau khi sửa, lệnh thoát mã 3, in đường dẫn đã resolve, không ghi attempt nào
- Canh gác:
  - `test_out_inside_results_is_refused_so_the_results_rules_cannot_be_bypassed[same-directory, inside]` (run) và `test_sweep_out_inside_results_is_refused` (sweep): FAIL trước khi sửa, PASS sau khi sửa
  - `test_out_whose_runs_directory_links_into_results_is_refused`: viết sau phần sửa, cho nhánh xét `out/runs/<run_id>`. Nó giết mutant "chỉ xét `out`" (plugin pytest thay hàm trong bộ nhớ, không sửa `src/`)
- Sửa:
  - `run`: `_check_out` từ chối (RunRefused, mã thoát 3) khi `out` hay `out/runs/<run_id>`, sau khi resolve, nằm trong `results_root`. Phép kiểm chạy trước `_check_code`, nên lượt bị từ chối không ghi gì
  - `sweep._check` nhận `out` và `results` đã resolve, và từ chối `--out` nằm trong thư mục kết quả trước mọi phép kiểm khác. `--dry-run` vẫn chỉ lập kế hoạch như trước. Dòng lệnh của tiến trình con giữ nguyên: nó tự so `--out` với `results/` mặc định trong cùng thư mục làm việc, nên chỉ có thể từ chối nhiều hơn sweep, không bao giờ ít hơn
  - SPEC §5.3 vốn ghi "`run --out` ra ngoài `results/`", nên code giờ khớp SPEC; không sửa SPEC
- Kích thước: I1 và I2 tách thành hai commit để mỗi commit giữ trong giới hạn của K5 (tối đa 5 file); gộp lại thì chạm 7 file. Trước khi tách, `make check-full` đã chạy trên trạng thái có cả I1 lẫn I2 (xem mục I2)
- Kiểm chứng: `make check-task` trên trạng thái chỉ có I1 → mã 0: "Sàn: đạt (618 test)", 613 passed, coverage 95.16%. Mất 155 s, vượt ngân sách 90 s (T1 chỉ cảnh báo), khi máy chủ có load average 17–20 trên 8 lõi
- Việc tiếp theo: I2 (results/ nằm trong worktree liên kết)

## 2026-10-02 — Phiên #18 (tiếp) — Sửa I2: results/ nằm trong worktree liên kết (P3.4)
- Trạng thái: I2 ✅ (REVIEW-C3-P4.1)
- File đã đổi: src/dfl/io/manifest.py, src/dfl/sim/run.py, src/dfl/sim/sweep.py, tests/unit/test_manifest.py, tests/integration/test_results_rules.py, tests/integration/test_sweep.py, tasks/todo.md, tasks/SESSION_LOG.md. Là 6 file ngoài tasks/, quá giới hạn của K5 một file: hàm dùng chung ở manifest, quy tắc ở cả nơi ghi (`run`) lẫn nơi kiểm sớm (`sweep`), mỗi nơi một test
- **Nguyên nhân gốc (một câu):** không chỗ nào kiểm thư mục kết quả, sau khi resolve, nằm ở đâu, nên một worktree quên symlink `results/` giữ kết quả ngay trong nó, và `git worktree remove` xóa luôn mà không hỏi (file bị ignore không làm worktree "bẩn")
- Tái hiện (git 2.43, repo tạm): worktree có `results/runs/abc/summary.json` bị ignore, `git status` sạch, `git worktree remove` thoát mã 0 và xóa cả results/. Tài liệu chỉ chặn "untracked files" và "modification in tracked files": https://git-scm.com/docs/git-worktree#Documentation/git-worktree.txt-remove
- Canh gác (FAIL trước khi sửa, PASS sau khi sửa):
  - `test_a_path_resolving_into_a_linked_worktree_names_that_worktree` (unit, manifest)
  - `test_results_inside_a_linked_worktree_are_refused` (run)
  - `test_results_inside_the_worktree_are_refused` (sweep)
  - Thêm `test_results_symlinked_from_a_linked_worktree_to_the_main_repo_are_accepted`, xanh cả trước lẫn sau khi sửa: canh để quy tắc không chặn quy trình đúng ở tasks/todo.md (results/ của worktree là symlink tới results/ của repo chính)
- Sửa:
  - `manifest.linked_worktree_of(path)` trả gốc của worktree liên kết chứa `path` sau khi resolve, hoặc None. Symlink tính theo đích; đường dẫn chưa có thì xét thư mục tổ tiên gần nhất đang có
  - `run` (khi ghi vào results/) và `sweep._check` từ chối khi thư mục kết quả nằm trong một worktree liên kết bất kỳ, và in đường dẫn đã resolve. Sweep kiểm trước khi ghi chỉ mục. Cây chính không bị ảnh hưởng: results/ của repo chính vẫn ghi được như trước
  - `test_two_sweeps_sharing_a_run_id_in_results_run_it_once` đổi sang results/ là symlink tới thư mục ngoài worktree, đúng quy trình. Trước đây test này ghi results/ ngay trong worktree, tức chính ca I2. Không test nào bị nới
- Kiểm chứng:
  - `make check-full` trên trạng thái này (sha256 của 7 file code và test trùng với bản đã kiểm) → mã 0: "Sàn: đạt (622 test)", ruff, format, import-linter 5/5, `check-inv` 73 test, 622 passed, coverage 95.19%, `pip check` sạch
  - `make check-task` trước commit → mã 0: 617 passed, coverage 95.19%; mất 140 s, vượt ngân sách 90 s (T1 chỉ cảnh báo), khi load average 20–23 trên 8 lõi
- Câu hỏi mở: SPEC §5.3 có nên ghi thành quy tắc "thư mục kết quả không nằm trong worktree liên kết" không? Hiện đó mới là quy trình ở tasks/todo.md. Việc này là sửa SPEC, nên phải hỏi trước
- Việc tiếp theo: các test P0 của P3.7 (P0-2 dây seed, P0-3 trường tổng hợp của RoundRecord, P0-4 thứ tự P_i), rồi P1-1…P1-7, rồi README

## 2026-10-02 — Phiên #18 (tiếp) — Test P0 của REVIEW-C3-P3.7 (P0-2, P0-3, P0-4)
- Trạng thái: test P0-2, P0-3, P0-4 ✅. Đây là test Prove-It cho code đúng (kỳ vọng PASS), giết các mutant mà P3.7 thấy sống sót
- File đã đổi: tests/integration/test_seeds.py, tests/integration/test_observer.py, tasks/todo.md, tasks/SESSION_LOG.md
- Test mới:
  - P0-2: `test_t_inv_5_every_seeded_stream_of_a_run_follows_config_seed[seed=0, seed=3]`. Nó so những gì `sim.run.assemble` rút ra (cách chia và vị trí node, w⁰, batch của từng node và của B1, cạnh ER, matching của gossip và ER-gossip ở 8 round đầu) với giá trị dựng thẳng từ luồng của seed theo SPEC §6
  - Kèm `test_seed_3_tells_apart_a_run_that_pins_any_one_stream_to_seed_0[8 luồng]`: ghim riêng một luồng vào seed 0 phải đổi giá trị rút ra, để ca seed 3 không trở nên vô nghĩa nếu profile smoke đổi sau này
  - Sửa `test_t_inv_5_another_seed_changes_the_partition_and_every_w0_tensor` (test_seeds.py:58 cũ) để so tập các phần thay vì so theo vị trí. Đây là thắt chặt: riêng hoán vị vị trí node không còn đủ cho test qua
  - P0-3: `test_round_record_aggregates_are_population_statistics_of_the_matching_per_node_values`: Evaluator giả, 12 trường tổng hợp so với giá trị tính tay (ddof = 0), sai số tương đối 1e-12
  - P0-4: `test_node_class_counts_follow_node_ids_so_acc_local_uses_each_nodes_own_labels`: Dirichlet α = 0.1, min_size = 8 (smoke chia được, 0 lần lấy mẫu lại, 4 hàng P_i khác nhau); `node_class_counts[i]` bằng bincount nhãn của node i, ở cả setup lẫn manifest.json
- Kiểm bằng mutant (plugin pytest thay hàm trong bộ nhớ, không sửa `src/`; chạy test_seeds, test_observer, test_baselines, test_describe) — 13/13 bị giết:
  - `sim.run` ghim seed 0 cho từng luồng PARTITION, PLACEMENT, GRAPH, INIT, BATCH, B1_BATCH, GOSSIP, GOSSIP_ER: cả 8 bị ca `[seed=3]` bắt. PARTITION bị thêm test so tập các phần đã sửa bắt; trước đó nó qua test này
  - 4 mutant của trường tổng hợp (`acc_local_min` dùng max, `f1_macro_mean` lấy acc, `acc_eff_mean` lấy acc_local, `f1_macro_std` tính trên acc): bị test P0-3 bắt
  - Thống kê P_i theo các phần trước khi gán vị trí: bị test P0-4 bắt. Vài test của B1 cũng đỏ theo, nhưng đó là do plugin giữ trạng thái cũ ở đường central, không phải hiệu ứng thật
- Kiểm chứng: `make check-task` → mã 0: "Sàn: đạt (634 test)", 629 passed, coverage 95.19%. Mất 139 s, vượt ngân sách 90 s (T1 chỉ cảnh báo), khi load average 16–18 trên 8 lõi
- Việc tiếp theo: test P1-1…P1-7 của P3.7 (P1-8 đã xong ở C3), rồi README, rồi phần còn lại của I6

## 2026-10-02 — Phiên #18 (tiếp) — README sau C3
- Trạng thái: README ✅ (mục còn mở của C3)
- File đã đổi: README.md, tasks/todo.md, tasks/SESSION_LOG.md
- Nội dung:
  - trạng thái sau C3: cái đã chạy được, cái chưa có (`analysis/`, thẻ và kết quả thí nghiệm);
  - lệnh tải MNIST;
  - bảng lệnh `make check-*`, `run` (smoke, fast, `--runtime processes`), `describe --check-req02`, `sweep --dry-run`;
  - quy tắc ghi kết quả, gồm hai quy tắc mới I1 và I2;
  - liên kết tới SPEC, CONSTRAINTS, docs/decisions/, tasks/
- Kiểm: lệnh smoke đúng như README (`--out build/smoke`) ghi đủ 4 file trong 10 s; `download`, `run`, `describe`, `sweep` đều có `--help`; lệnh `describe` là lệnh đã cho "REQ-02: đạt" trên MNIST thật (mục T05a)
- Kiểm chứng: `make check-task` → mã 0: "Sàn: đạt (634 test)", 629 passed, coverage 95.19%; mất 131 s (vượt ngân sách 90 s, chỉ cảnh báo)
- Việc tiếp theo: test P1-1…P1-7 của P3.7, phần còn lại của I6 (vòng round dùng chung, lý do phân kỳ trùng giữa hai runtime), T11e, smoke trên laptop demo (QS-8)

## 2026-10-15 — Phiên #19 — Quy ước Conventional Commits và đẩy lên GitHub
- Trạng thái: hook commit-msg ✅; đẩy nhánh `binh` lên https://github.com/vytranly/DFL (remote `origin`)
- File đã đổi: scripts/git-hooks/commit-msg, .gitmessage, .gitignore (bỏ qua docs/report/*.pptx và slides/), Makefile (`install-hooks`), README.md, tests/unit/test_commit_msg_hook.py, tasks/SESSION_LOG.md
- Nội dung: hook kiểm dòng đầu `<type>(<scope>)!: <subject>`, ASCII (tiếng Anh), ≤ 100 ký tự; cho qua thông điệp git tự sinh (Merge, Revert, fixup!/squash!/amend!). Cài bằng chép vào `.git/hooks` (không đặt `core.hooksPath`, để giữ hook post-commit/post-rewrite có sẵn)
- Test mới: 17 ca trong tests/unit/test_commit_msg_hook.py (8 nhận, 9 từ chối)
- Việc tiếp theo: thành viên khác chạy `make install-hooks` sau khi clone

## 2026-10-15 — Phiên #19 (tiếp) — CIFAR-10 (ADR-012, T28a–f)
- Trạng thái: T28a–f ✅; review T28 xong (APPROVE, R1–R5 đã sửa). ADR-012 còn Proposed, chờ nhóm duyệt. Nhánh `binh` chưa đẩy lên GitHub vì máy chủ không có credential GitHub
- Quyết định (hỏi theo Ask first, 2026-10-15): làm code CIFAR-10 trước M4; model là CNN của NIID-Bench; có augmentation crop + flip; phạm vi gồm code, ADR và đo trên dữ liệu thật, không chạy lưới. Sau review (R1): augmentation do dataset quyết định, ghi giá trị khác thì cấu hình báo lỗi
- Commit: 296d0aa (T28a), d20a314 (T28b), 5aa6d55 (T28c), 1916eb4 (T28d), 8aa7f3b (T28e), d18fd39 (R2), f11172d (T28f, R5), d3d8f7c (R1), e90393f (R3, R4), cộng commit này
- File chính: `src/dfl/data/datasets.py`, `src/dfl/learning/{models,augment,local_train}.py`, `src/dfl/config.py`, `src/dfl/cli.py`, `src/dfl/sim/{run,sweep}.py`, `configs/base.yaml`, `configs/profiles/{smoke_cifar10,cifar10_fast}.yaml`, ADR-012, SPEC, README
- Nội dung:
  - Dữ liệu: bản binary của CIFAR-10 đọc bằng numpy, không pickle (F6); tải bằng `torchvision.datasets.utils.download_and_extract_archive` có kiểm md5; chia 45k/5k/10k; chuẩn hóa theo kênh; dữ liệu tổng hợp `synthetic_cifar10`
  - Model `cnn_niidbench` (62,006 tham số); cấu hình từ chối model không khớp shape ảnh của dataset
  - Augmentation `crop_flip`, lấy ngẫu nhiên từ Generator batch của node; không đổi contract `LocalTrainer`
  - CLI `download cifar10`; `run` và `sweep` chặn mọi dữ liệu tổng hợp khỏi `results/`
- Test mới: 703 test thu thập được (611 ở C3, 646 đầu phiên).
  - Oracle `torchvision.transforms.v2.functional` cho crop + flip.
  - Định dạng bản ghi binary dựng tay theo trang chính thức.
  - T-INV-13 trên đường CIFAR-10 có augmentation.
  - Golden sha256 cho dữ liệu smoke và tensor MNIST đã chuẩn hóa.
  - Mutation: đảo thứ tự file train và tính bảng chuẩn hóa bằng float32 đều bị bắt.
- Kiểm chứng:
  - `make check-task` mã 0 trước mọi commit; lần cuối 697 passed, coverage 95.4%, 143 s (vượt ngân sách 90 s, chỉ cảnh báo; load 14–18). `pytest tests/e2e` 6 passed.
  - MNIST trước/sau (1c10f2b → 8aa7f3b): `rounds.jsonl` và `summary.json` của smoke và fast trùng từng byte.
  - Số đo trên CIFAR-10 thật ở ADR-012, mục "Số đo".
- Việc tiếp theo:
  - Nhóm duyệt ADR-012, gồm cả việc có đổi cách đệm sang reflect như NIID-Bench không (R5).
  - Chọn η của CIFAR-10 trên validation; quyết lưới thu gọn, ngân sách và ngưỡng sanity sau T13/M4.
  - Process runtime với CIFAR-10 ở N = 20 cần khoảng 27 GB RAM, không vừa máy chủ dùng chung: giảm N, hoặc điền tensor theo khối để hạ RAM đỉnh.
  - Đẩy nhánh `binh` (`git push -u origin binh`, cần token GitHub).

## 2026-10-15 — Phiên #19 (tiếp) — Lối tắt make không cần activate venv
- Trạng thái: ✅ (yêu cầu của thành viên nhóm: gõ `make` mà không phải activate venv)
- File đã đổi: Makefile, README.md, tasks/SESSION_LOG.md
- Nội dung:
  - Thêm `make help`, `make venv` (đúng lệnh cài đặt của SPEC §4, gọi pip của `.venv`), `make shell` (shell con bash đọc ~/.bashrc rồi activate `.venv`; gõ exit để quay lại), `make download-mnist`, `make download-cifar10`, `make smoke`, `make smoke-cifar10`, `make fast`, `make cifar10-fast`.
  - Các lệnh chạy nhận thêm cờ qua `ARGS`, và ghi ra `build/<tên>`.
  - Lý do có `make shell`: make không đổi được môi trường của terminal đang gõ, vì mỗi dòng lệnh chạy trong một tiến trình con.
  - Target kiểm tra giữ nguyên; `make` không kèm tên vẫn chạy `check-fast`.
- Kiểm chứng:
  - Với PATH không có python nào, `make download-cifar10` và `make smoke-cifar10` chạy bằng `.venv/bin/python`.
  - `make shell` cho `VIRTUAL_ENV=.../.venv`, `python` của `.venv`, dấu nhắc `(.venv)`.
  - Thiếu venv thì `make shell` báo "chạy: make venv".
  - `make -n venv` in đúng bốn lệnh cài đặt; không chạy `make venv` thật vì `.venv` đã có.

## 2026-10-15 — Phiên #19 (tiếp) — Test P1 của REVIEW-C3-P3.7 (P1-1 … P1-7)
- Trạng thái: test P1-1…P1-7 ✅. Đây là test Prove-It cho code đúng, kỳ vọng PASS. P1-8 đã làm ở C3. Mục còn mở của C3: phần còn lại của I6
- File đã đổi: tests/integration/{conftest,test_engine_core,test_observer,test_process_runtime}.py, tests/unit/test_partition.py, tasks/todo.md, tasks/SESSION_LOG.md
- Test mới (12 ca):
  - P1-1 `test_t_inv_4_server_path_is_bitwise_independent_of_node_order[server, gossip, fully_connected]`: kích thước (64, 96, 128, 200), ba thứ tự duyệt, 3 round; snapshot đã gửi, kể cả snapshot server phát xuống, không đổi. `RecordingTransport` có thêm `with_server`
  - P1-2: tham số hóa `test_t_inv_11_…` với n ∈ {4, 5, 20}, và siết điều kiện consensus từ ≤ 1e-6·drift thành == 0.0 (SAN-5)
  - P1-3 `test_t_inv_8_dirichlet_resamples_when_every_remaining_probability_is_zero`: 2 lần thử bị bỏ vì xác suất còn lại bằng 0, trùng hàm tham chiếu; nhánh này trước đây chưa từng chạy
  - P1-4 `test_t_inv_8_a_node_with_exactly_min_size_samples_is_accepted`: IID 4 × 64 mẫu được nhận, 255 mẫu bị từ chối. Dirichlet α = 0.1, seed 2: min_size 72 cho retries 0, 73 cho retries 1
  - P1-5 `test_t_inv_8_quantity_retries_count_the_resamples_of_the_reference`: số lần thử đếm trên hàm tham chiếu bằng một Generator bọc (19 lần lấy mẫu lại)
  - P1-6 `test_exhausting_retries_makes_one_try_plus_exactly_1000_resamples[dirichlet, quantity]`: sau ValueError, trạng thái Generator bằng một Generator sinh đôi đã rút đúng 1,001 lần thử
  - P1-7 `test_t_inv_13_processes_match_the_simulator_on_a_sparse_evaluation_grid[ring, gossip]`: mốc ở round 2, 4, 5; `process_timeout` 120 s để lệch lịch thì fail nhanh
- Kiểm bằng mutant: 9/9 bị giết
  - Chạy trong một tiến trình, trên bản sao `src/` qua PYTHONPATH:
    - server cộng theo thứ tự duyệt;
    - consensus tính bằng float32;
    - nhánh xác suất 0 đổi `break` thành `continue`;
    - `>=` thành `>` khi nhận phần Dirichlet;
    - `<` thành `<=` ở `_result`;
    - retries của quantity cộng thêm một;
    - `range(MAX_RETRIES)` và `range(MAX_RETRIES + 2)`.
  - Chạy với tiến trình con: phía node xét `is_due(t + 2)`. Sửa tạm trong `src/` rồi khôi phục bằng `git checkout`; `git diff -- src` rỗng sau đó.
  - Đối chứng: mutant consensus float32 vẫn qua test T-INV-11 cũ (N = 4), và mutant `is_due` phía node vẫn qua T-INV-13 cũ (lịch mỗi round). Đó đúng là hai khoảng trống review nêu.
- Kiểm chứng: `make check-task` mã 0, 709 passed, coverage 95.47% (partition.py đủ 100%); mất 333 s, vượt ngân sách 90 s (chỉ cảnh báo; máy chủ đang tải nặng)
- Việc tiếp theo:
  - Phần còn lại của I6: round dùng chung, lý do phân kỳ trùng giữa hai runtime.
  - T11e.
  - Các lỗi P2 có test FAIL: P2-1, P2-2 ca NaN, P2-3, P2-12.
  - Tăng `MIN_TESTS` ở checkpoint sau.

