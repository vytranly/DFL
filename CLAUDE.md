# Project: Non-IID × Decentralized Federated Learning (Đề 2 — ĐATH HK261)

## Mục tiêu
Mô phỏng DFL không server để đo ảnh hưởng của Non-IID lên accuracy/F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định, phương sai giữa node; so sánh topology và thuật toán.
Chi tiết: docs/KE_HOACH_DE_TAI_2.md · SPEC.md · docs/decisions/ · CONSTRAINTS.md · tasks/todo.md

## Tech stack
Python 3.12 · torch 2.14.0 · torchvision 0.29.0 · networkx 3.x · numpy 2.x · pandas · matplotlib · pyyaml · pytest · ruff · import-linter

## Commands
- Cài đặt (CPU): `python3.12 -m venv .venv && . .venv/bin/activate && pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]"`
- Kiểm tra nhanh / theo task / đầy đủ: `make check-fast` · `make check-task` · `make check-full` (test bất biến: `make check-inv`, bắt buộc từ C3)
- Chạy một lượt: `python -m dfl run --config configs/profiles/smoke.yaml`
- Chạy lưới: `python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --workers 4`
- Vẽ hình: `python -m analysis.plots --results results/ --out figures/`

## Quy ước code
- Layout src/: `src/dfl/...`; ưu tiên hàm thuần + dataclass; type hint đầy đủ; docstring ngắn giải thích *vì sao*.
- Truyền `numpy.random.Generator` / `torch.Generator` tường minh; không dùng `np.random.*` toàn cục.
- `data/partition.py`, `data/heterogeneity.py`, `network/topology.py`, `network/mixing.py` chỉ phụ thuộc numpy/networkx (không import torch); `network/transport.py` coi payload là đối tượng mờ.
- Không BatchNorm. Tổng hợp model dùng snapshot, không cập nhật tại chỗ.
- Test: pytest, Arrange-Act-Assert, tên test mô tả hành vi; `tests/unit|integration|e2e`.

## Bất biến P2P (không được vi phạm) — chi tiết docs/KE_HOACH_DE_TAI_2.md §4.5
INV-1 node chỉ đọc model của mình + inbox · INV-2 chỉ Transport đếm byte · INV-3 tổng hợp dùng snapshot ·
INV-4 W không âm, tổng hàng = 1, khớp cạnh; đồ thị tĩnh liên thông, gossip liên thông theo hợp các round · INV-5 observer chỉ đọc · INV-6 cùng cấu hình + seed + code_version + máy + số luồng → cùng rounds.jsonl

## Boundaries
Đọc CONSTRAINTS.md; không bao giờ làm yếu nó để qua kiểm tra.
- Always: viết test trước; `make check-task` trước khi commit; một task = một commit; cập nhật tasks/todo.md và tasks/SESSION_LOG.md; dẫn URL tài liệu chính thức cho API framework; chỉ chạy sweep khi cây làm việc sạch.
- Ask first: thêm phụ thuộc; sửa SPEC/ADR/CONSTRAINTS; đổi siêu tham số đã chốt; di chuyển hoặc xóa bất cứ gì trong results/; chạy lưới lớn hơn 10 lượt.
- Never: sửa tay results/ hoặc figures/; xóa lượt chạy lỗi; trộn kết quả của nhiều code_version trong một hình; chọn siêu tham số trên tập test; hạ ngưỡng trong CONSTRAINTS.md để qua kiểm tra; skip hay xóa test đang fail khi chưa được duyệt; commit dữ liệu MNIST hoặc results/; `torch.load(..., weights_only=False)`.
