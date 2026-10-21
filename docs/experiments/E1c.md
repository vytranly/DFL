# E1c — Đối chứng lệch mục tiêu (tiền đăng ký, Tier 2)

- **Mục đích:** kiểm H6 (G19): một phần suy giảm dưới Dirichlet có thể do DFL với W ngẫu nhiên kép tối ưu $\frac1N\sum_i F_i$ thay vì $\sum_i \frac{n_i}{n} F_i$ (lệch mục tiêu), không chỉ do drift. Định nghĩa dùng chung ở `docs/experiments/E1.md`.
- **Lưới** (`configs/experiments/e1c_objective.yaml`, 15 lượt, SPEC EXP-3): B3 (FedAvg đúng: server ảo chỉ số N tổng hợp theo $n_k/n$, đường server của engine) × α ∈ {100, 1, 0.5, 0.1, 0.05} × seed {0, 1, 2}. Đối chứng: FC + MH (trọng số 1/N) của E1, cùng seed.
- **Cố định:** DFedAvg τ = 20, μ = 0, η = 0.1, N = 20, profile full. Cập nhật model của B3 giống hệt FC với trọng số size-weighted; chỉ khác chi phí (2N message/round qua server).

## H6 — phép so sánh chính

- **Độ đo:** với mỗi seed s ∈ {0, 1, 2}: $e_s$ = accuracy cuối của B3 − accuracy cuối của FC + MH, ở α = 0.1.
- **Chiều kỳ vọng:** $e_s > 0$ (sửa trọng số mục tiêu về $n_k/n$ cải thiện accuracy trên phân bố test). **Quy tắc:** 3 seed, tối đa "gợi ý".
- **Giới hạn đã biết, ghi trước (KH §2.3):** đổi trọng số cũng đổi cách trộn các model đã drift, nên phép so chỉ ước lượng phần lệch mục tiêu, không tách hoàn toàn.

## Khám phá

- Cùng hiệu số trên cả thang α; quan hệ với `objective_shift_tv` của manifest; Acc^eff.
- Phần đầu của H6 (quantity so với label skew) ở E4a, E4.
- Chi phí: B3 gửi 2N = 40 message/round, tải dồn vào server ảo (P13), so với 380 của FC.

## Chạy

- **code_version, máy:** như E1. Tier 2, chạy sau Tier 1.
- **Lệnh:** `python -m dfl sweep --experiment configs/experiments/e1c_objective.yaml --workers 4`; `python -m analysis.sanity --results results/ --san5 e1c_objective`; `python -m analysis.hypotheses --results results/ --only H6`.
- **Ngân sách:** khoảng 5 giờ-lõi.
- **Tiêu chí hợp lệ:** 15 run_id có attempt `ok` hoặc `diverged`, cùng `code_version` với E1.
- **Kết quả** (điền sau khi chạy).
