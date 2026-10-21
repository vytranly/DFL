# E3a — Ba thuật toán đề yêu cầu (tiền đăng ký)

- **Mục đích:** REQ-08 (đủ D-PSGD, DFedAvg, DFedProx) và kiểm H5 ở τ = 20; khuyến nghị tạm về chiến lược tổng hợp (3 seed, REQ-13). Định nghĩa dùng chung ở `docs/experiments/E1.md`.
- **Lưới** (`configs/experiments/e3a_algorithms.yaml`, 24 lượt, SPEC EXP-2): {D-PSGD (τ = 1, μ = 0), DFedProx (τ = 20, μ = 0.5)} × {T1 ring, T4 FC} × {IID, α = 0.1} × seed {0, 1, 2}. DFedAvg (τ = 20, μ = 0) lấy từ E0b (IID) và E1 (α = 0.1), seed {0, 1, 2}.
- **μ của DFedProx:** SPEC ghi "μ = 1" là giá trị danh nghĩa ở η = 0.05; theo QS-2 và ADR-007, cường độ ημτ = 1 ở τ = 20 được giữ, nên với η = 0.1 thì μ = 1 × 0.05/0.1 = **0.5**.
- **Cố định:** η = 0.1 chung cho mọi τ (ADR-007; τ = 1 không kích hoạt T19c ở E0a lần đầu), τ·R = 4,000, N = 20, MH, profile full.

## H5 — phép so sánh chính

- **Phát biểu (KH §2.3):** DFedProx mạnh (ημτ = 1 ở τ = 20) giảm drift và cải thiện accuracy ở α = 0.1, rõ hơn khi τ lớn; ở IID nó làm chậm hội tụ; μ yếu cho khác biệt nhỏ.
- **Độ đo:** với mỗi seed s ∈ {0, 1, 2}: $d_s$ = trung bình qua {ring, FC} của [accuracy cuối của DFedProx μ = 0.5 − accuracy cuối của DFedAvg], ở α = 0.1, τ = 20.
- **Chiều kỳ vọng:** $d_s > 0$. **Quy tắc:** 3 seed, tối đa "gợi ý"; chỉ cho khuyến nghị tạm.
- **Khám phá:** drift theo thời gian của DFedProx so với DFedAvg; R@95 ở IID (chậm hơn?); phần "rõ hơn khi τ lớn" và μ yếu ở E3b.

## Khám phá: D-PSGD và so sánh theo byte (G10)

- D-PSGD (τ = 1) giao tiếp mỗi bước: 4,000 round, gấp 20 lần số byte của DFedAvg ở cùng tổng số bước cục bộ. So accuracy theo cả trục bước cục bộ và trục byte; bytes@95; hình P9.
- Khuyến nghị tạm (REQ-13) về chiến lược tổng hợp chỉ dựa trên kết luận "gợi ý" của H5 và mô tả này, ghi rõ 3 seed.

## Chạy

- **code_version, máy:** như E1. Chạy sau khi SAN-1…3 và SAN-5 của E0b đạt (ADR-007, đường găng).
- **Lệnh** (từ `/home/tun/ecaep/dfl-run`, `PYTHONPATH=src`): `python -m dfl sweep --experiment configs/experiments/e3a_algorithms.yaml --workers 4`; `python -m analysis.sanity --results results/ --san5 e3a_algorithms`; `python -m analysis.hypotheses --results results/ --only H5`.
- **Ngân sách:** khoảng 8.7 giờ-lõi, tức khoảng 2.2 giờ với 4 worker.
- **Tiêu chí hợp lệ:** 24 run_id có attempt `ok` hoặc `diverged`, cùng `code_version` với E0b, E1.
- **Kết quả** (điền sau khi chạy).
