# Non-IID × Decentralized Federated Learning (Đề 2 — ĐATH HK261)

Mô phỏng Decentralized Federated Learning (không server) trên MNIST để đo ảnh hưởng của mức Non-IID lên accuracy/F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định và phương sai giữa các node, và so sánh các topology (ring, Erdős–Rényi, random gossip, fully-connected) cùng các thuật toán (D-PSGD, DFedAvg, DFedProx).

> Trạng thái: khung dự án (P1.1). Chưa có code chạy được.

## Cài đặt (CPU)

```bash
python3.12 -m venv .venv
. .venv/bin/activate
# Cài torch bản CPU trước; nếu không, pip sẽ kéo bản CUDA nặng vài GB từ PyPI.
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
# Đúng phiên bản mọi gói chạy (requirements.lock chỉ gồm gói chạy, không có công cụ dev; SPEC §4).
pip install -r requirements.lock
# Công cụ dev; -c giữ nguyên phiên bản gói chạy.
pip install -e ".[dev]" -c requirements.lock
```

## Lệnh

| Lệnh | Mô tả |
|---|---|
| `pytest -q` | Chạy test |
| `ruff check .` | Lint |
| `ruff format .` | Định dạng code |

Các lệnh `make check-*`, `python -m dfl run|sweep` và `python -m analysis.plots` sẽ có sau các task tiếp theo.

## Tài liệu

- Kế hoạch, kiến trúc, thiết kế thí nghiệm: [docs/KE_HOACH_DE_TAI_2.md](docs/KE_HOACH_DE_TAI_2.md)
- Quy trình làm việc với agent: [docs/HANDOFF_GUIDELINE.md](docs/HANDOFF_GUIDELINE.md)
- Quy tắc cho agent: [CLAUDE.md](CLAUDE.md)
