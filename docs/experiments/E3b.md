# E3b — Tần suất giao tiếp và prox (tiền đăng ký, Tier 2)

- **Mục đích:** RQ3; kiểm H7; phần "rõ hơn khi τ lớn" và "μ yếu cho khác biệt nhỏ" của H5 (phép so sánh chính của H5 ở E3a). Định nghĩa dùng chung ở `docs/experiments/E1.md`.
- **Lưới** (`configs/experiments/e3b_tau_mu.yaml`, 42 lượt, SPEC EXP-3), seed {0, 1, 2}, τ·R = 4,000 ở mọi ô:
  - μ = 0: τ ∈ {5, 50} × {T1, T4} × {IID, α = 0.1} (24);
  - μ = 0.05 (danh nghĩa 0.1, QS-2), τ = 20 × {T1, T4} × {IID, α = 0.1} (12);
  - μ = 0.5 (danh nghĩa 1, QS-2), τ = 50 × {T1, T4} × α = 0.1 (6).
  Ô τ = 20, μ = 0 lấy từ E0b (IID) và E1 (α = 0.1), seed {0, 1, 2}.
- **Cố định:** η = 0.1 chung cho mọi τ (ADR-007), N = 20, MH, profile full. Nếu E0a lần chạy lại chọn η riêng cho τ = 50 khác η chung, 12 lượt có điều kiện (T19c) chạy thêm với η riêng cho các ô μ = 0, τ = 50.

## H7 — phép so sánh chính

- **Phát biểu (KH §2.3):** ở cùng tổng số bước cục bộ, τ nhỏ cho accuracy tốt hơn dưới Non-IID nhưng tốn nhiều byte hơn; điểm cân bằng phụ thuộc topology.
- **Độ đo:** với mỗi seed s ∈ {0, 1, 2}: $h_s$ = trung bình qua {ring, FC} của [accuracy cuối ở τ = 5 − accuracy cuối ở τ = 50], μ = 0, α = 0.1.
- **Chiều kỳ vọng:** $h_s > 0$. **Quy tắc:** 3 seed, tối đa "gợi ý".
- Phần "tốn nhiều byte hơn" đúng theo cấu trúc (τ = 5 gửi gấp 10 lần số message của τ = 50 ở cùng số bước), nên chỉ báo cáo, không kiểm.

## Khám phá

- H5 phần τ lớn: ở α = 0.1, τ = 50: μ = 0.5 so với μ = 0 (theo seed, theo topology); μ = 0.05 so với μ = 0 ở τ = 20 (kỳ vọng khác biệt nhỏ); ở IID: R@95 của μ = 0.05 so với μ = 0.
- Accuracy theo tổng số bước cục bộ và theo byte cho τ ∈ {1 (E3a), 5, 20, 50} (P6, P9); bytes@95 theo τ và topology.

## Chạy

- **code_version, máy:** như E1. Tier 2.
- **Lệnh:** `python -m dfl sweep --experiment configs/experiments/e3b_tau_mu.yaml --workers 4`; `python -m analysis.sanity --results results/ --san5 e3b_tau_mu`; `python -m analysis.hypotheses --results results/ --only H7`.
- **Ngân sách:** khoảng 14 giờ-lõi.
- **Tiêu chí hợp lệ:** 42 run_id có attempt `ok` hoặc `diverged`, cùng `code_version`.
- **Kết quả** (điền sau khi chạy).
