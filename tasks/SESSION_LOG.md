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
