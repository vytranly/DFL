# Non-IID × Decentralized Federated Learning (Đề 2 — ĐATH HK261)

Mô phỏng Decentralized Federated Learning (không server) trên MNIST để đo ảnh hưởng của mức Non-IID lên accuracy/F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định và phương sai giữa các node, và so sánh các topology (ring, Erdős–Rényi, random gossip, fully-connected) cùng các thuật toán (D-PSGD, DFedAvg, DFedProx).

> **Trạng thái (2026-10-02):** xong lát chạy trọn một lượt (checkpoint C3, nhóm duyệt ngày 2026-10-02).
> - Chạy được: `python -m dfl run` (một lượt, ghi đủ 4 file kết quả, mọi node trong một tiến trình hay mỗi node một tiến trình), `describe` (thống kê Non-IID của các cách chia, kiểm REQ-02) và `sweep` (một nhóm thí nghiệm).
> - Chưa có: `analysis/` (R@x, AUC, hình, bảng; T15, T16), thẻ và kết quả thí nghiệm (từ E0a).
> - Tiến độ từng task: [tasks/todo.md](tasks/todo.md).

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
# MNIST, một lần (cần mạng), vào data/.
python -m dfl download mnist
```

## Lệnh

| Lệnh | Mô tả |
|---|---|
| `make check-fast` | Sàn của CONSTRAINTS.md, ruff, format, import-linter và test unit; sau mỗi lần sửa |
| `make check-task` | Như `check-fast`, cộng test integration và coverage; trước mỗi commit (ngân sách 90 s, chỉ cảnh báo) |
| `make check-full` | Toàn bộ test (cả e2e), `check-inv` và `pip check`; ở mỗi checkpoint |
| `make check-inv` | Có đủ test bất biến T-INV-1…13 và tất cả đều xanh |
| `python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke` | Một lượt smoke trên dữ liệu tổng hợp, không cần MNIST |
| `python -m dfl run --config configs/profiles/fast.yaml --out build/fast` | Một lượt fast trên MNIST (N = 20). Thêm `--runtime processes` để mỗi node chạy trong một tiến trình riêng (ADR-010) |
| `python -m dfl describe --config configs/base.yaml --seeds 20 --out build/describe --check-req02` | Thống kê Non-IID của mọi cách chia qua 20 seed, so JS với bảng KH §3.2 (REQ-02) |
| `python -m dfl sweep --experiment configs/experiments/<nhóm>.yaml --dry-run` | Lập kế hoạch một nhóm thí nghiệm; bỏ `--dry-run` (thêm `--workers <n>` nếu cần) để chạy |

## Commit

Dòng đầu commit theo [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), bằng tiếng Anh, tối đa 100 ký tự: `<type>(<scope>)!: <subject>`, ví dụ `fix(sim): refuse an --out that leads into results/ (I1)`. `type` là một trong `feat fix docs style refactor perf test build ci chore revert`; thân commit viết tự do. Sau khi clone, chạy `make install-hooks` một lần để cài hook `commit-msg` (chặn header sai dạng) và mẫu thông điệp `.gitmessage`.

## Kết quả

- Mỗi lượt ghi `runs/<run_id>/attempt-<k>/` gồm `manifest.json`, `rounds.jsonl`, `timing.jsonl`, `summary.json` (SPEC §5.3).
- Có `--out` thì ghi vào thư mục đó ở mọi trạng thái cây. Thư mục này phải nằm ngoài `results/`, ví dụ `build/`.
- Không có `--out` thì ghi vào `results/`, và chỉ khi:
  - cây sạch;
  - môi trường khớp `requirements.lock`;
  - dữ liệu là MNIST thật, không phải dữ liệu smoke;
  - run_id chưa có attempt nào xong.
- Lưới cho báo cáo chạy bằng `sweep` từ một worktree riêng (`git worktree add`). Trong worktree đó, `data/` và `results/` là symlink tới thư mục của repo chính (quy trình ở [tasks/todo.md](tasks/todo.md), E0a).
- Không sửa tay `results/` hay `figures/` (CLAUDE.md).

## Tài liệu

- Kế hoạch, kiến trúc, thiết kế thí nghiệm: [docs/KE_HOACH_DE_TAI_2.md](docs/KE_HOACH_DE_TAI_2.md)
- Đặc tả: [SPEC.md](SPEC.md). Ràng buộc chất lượng và các ngưỡng: [CONSTRAINTS.md](CONSTRAINTS.md)
- Quyết định kiến trúc (ADR) và các bản review ở checkpoint: [docs/decisions/](docs/decisions/)
- Kế hoạch task, tiến độ, nhật ký phiên: [tasks/plan.md](tasks/plan.md), [tasks/todo.md](tasks/todo.md), [tasks/SESSION_LOG.md](tasks/SESSION_LOG.md)
- Quy trình làm việc với agent: [docs/HANDOFF_GUIDELINE.md](docs/HANDOFF_GUIDELINE.md)
- Quy tắc cho agent: [CLAUDE.md](CLAUDE.md)
