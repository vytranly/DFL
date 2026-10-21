# E4a — Quantity skew (tiền đăng ký)

- **Mục đích:** REQ-02 (quantity skew được đo) và một phần RQ4. Không có phép so sánh chính: H6 có phép so sánh chính ở E1c; ở đây chỉ mô tả. Định nghĩa dùng chung ở `docs/experiments/E1.md`.
- **Lưới** (`configs/experiments/e4a_quantity.yaml`, 6 lượt, SPEC EXP-2; ≤ 10 lượt nên không cần duyệt `--dry-run`): P3 β = 0.5 (nhãn IID, kích thước node lấy theo $\mathrm{Dir}_N(\beta)$) × {T2 ER p = 0.3, T4 FC} × MH × seed {0, 1, 2}.
- **Cố định:** DFedAvg τ = 20, μ = 0, η = 0.1, N = 20, profile full.

## Khám phá (báo cáo, không kết luận "ủng hộ")

- Mức sụt so với IID (E0b, cùng topology, seed {0, 1, 2}) và so với α = 0.1 (E1 cho FC, E2 cho ER): phần đầu của H6 ("quantity skew gây suy giảm nhỏ hơn nhiều so với label skew").
- Kích thước node (min, max, hệ số biến thiên) và độ lệch mục tiêu `objective_shift_tv` từ manifest (G11, G19); Acc^eff.
- Trọng số trộn với quantity skew ở E4.

## Chạy

- **code_version, máy:** như E1. Chạy sau khi SAN-1…3, SAN-5 của E0b đạt.
- **Lệnh:** `python -m dfl sweep --experiment configs/experiments/e4a_quantity.yaml --workers 4`; `python -m analysis.sanity --results results/ --san5 e4a_quantity`; `python -m analysis.aggregate --results results/ --out build/analysis`.
- **Ngân sách:** khoảng 1.9 giờ-lõi.
- **Tiêu chí hợp lệ:** 6 run_id có attempt `ok` hoặc `diverged`, cùng `code_version`.
- **Kết quả** (điền sau khi chạy).
