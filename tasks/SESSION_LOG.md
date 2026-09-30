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
