# Session log

Mỗi phiên thêm một mục theo mẫu docs/HANDOFF_GUIDELINE.md §5.2. Mục mới nhất ở cuối.

## 2026-09-30 — Phiên #1 — P1.1 / T01 Dựng repo
- Trạng thái: T01 ✅ (commit đầu tiên "chore: bootstrap repo, rules file and project skeleton")
- File đã đổi: .gitignore, pyproject.toml, CLAUDE.md, README.md, khung src/dfl, analysis, tests, configs, figures, docs/{decisions,experiments,report}, tasks/SESSION_LOG.md
- Kiểm chứng: xem báo cáo của phiên (pip install, import torch/torchvision/networkx, pytest, ruff)
- Quyết định mới: cài torch/torchvision bản CPU từ https://download.pytorch.org/whl/cpu trước khi `pip install -e ".[dev]"` (tránh bản CUDA nhiều GB); có một test smoke import để `pytest -q` thoát mã 0 khi chưa có test nào khác
- Câu hỏi mở: Q1–Q7 trong docs/KE_HOACH_DE_TAI_2.md §11 (chưa hỏi GVHD)
- Việc tiếp theo: P1.3 (chốt giả định/câu hỏi mở) hoặc P1.5 (SPEC.md)
