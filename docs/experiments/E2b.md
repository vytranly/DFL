# E2b — Gossip công bằng (tiền đăng ký, Tier 2)

- **Mục đích:** phần công bằng của H4 (ADR-009). Gossip trên $K_N$ (T3) được ghép với node bất kỳ, lợi thế mà ring và ER không có; T3b chỉ ghép trên cạnh của đồ thị ER (T2). Không có phép so sánh chính: H4 có phép so sánh chính ở E2. Định nghĩa dùng chung ở `docs/experiments/E1.md`.
- **Lưới** (`configs/experiments/e2b_gossip.yaml`, 9 lượt, SPEC EXP-3): T3b × {IID, α = 0.1, #C = 2} × seed {0, 1, 2}. Đối chứng: T3 (E0b ở IID, E2 ở α = 0.1 và #C = 2) và T2 (E0b, E2), cùng seed.
- **Cố định:** DFedAvg τ = 20, μ = 0, η = 0.1, N = 20, MH trên matching, profile full.

## Khám phá (báo cáo, không kết luận "ủng hộ")

- Accuracy cuối và bytes@95 của T3b so với T3 và T2; message mỗi round (T3b ≤ 20).
- Tốc độ trộn kỳ vọng của T3b ước lượng bằng 10⁴ matching (ADR-009), đặt riêng với spectral gap (P8).

## Chạy

- **code_version, máy:** như E1. Tier 2.
- **Lệnh:** `python -m dfl sweep --experiment configs/experiments/e2b_gossip.yaml --workers 4`; `python -m analysis.sanity --results results/ --san5 e2b_gossip`.
- **Ngân sách:** khoảng 3 giờ-lõi.
- **Tiêu chí hợp lệ:** 9 run_id có attempt `ok` hoặc `diverged`, cùng `code_version`.
- **Kết quả** (điền sau khi chạy).
