# E0a — Chọn η trên validation (tiền đăng ký)

- **Mục đích:** chọn η chung của lưới, η riêng của B1, và biết hai đầu dải τ (τ = 1, τ = 50) có bị thiệt vì η chung không. Không kiểm giả thuyết nào; E0a là bước chọn siêu tham số đã đăng ký trước ở ADR-007 (mục "Chọn lr") và KH §3.5.
- **Lưới** (`configs/experiments/e0a_lr.yaml`, 24 lượt, đúng SPEC EXP-2):
  - τ = 20: η ∈ {0.01, 0.05, 0.1} × {FC-IID, ring-α0.1, ring-#C2} = 9 lượt;
  - τ = 1: η × {FC-IID, ring-α0.1} = 6 lượt; τ = 50: η × {FC-IID, ring-α0.1} = 6 lượt;
  - B1: η × một node giữ đủ 55k ảnh, τ·R = 17,200 (khoảng 10 epoch với B = 32, F19) = 3 lượt.
- **Cố định** (profile `configs/profiles/full.yaml`, đã chốt ở T13, docs/PERF.md): DFedAvg (μ = 0), N = 20, τ·R = 4,000, B = 32 lấy có hoàn lại, SmallCNN, Metropolis–Hastings, min_size = 64, 1 luồng mỗi lượt. Seed 100 (ngoài {0…4}). Đánh giá mỗi 100 bước cục bộ trên tập con phân tầng 2,000 ảnh của **validation**; đánh giá cuối trên đủ 5,000 ảnh validation. Không lượt nào đánh giá trên tập test.
- **Quy tắc chọn** (ADR-007; cài ở `analysis/select_lr.py`, T14e):
  - Accuracy validation của một lượt = trung bình `acc_mean` ở 5 mốc cuối (3,600 … 4,000 bước); lượt phân kỳ tính là 0.
  - η chung: η có accuracy validation nhỏ nhất qua 3 ô τ = 20 là lớn nhất. Các η cách giá trị tốt nhất không quá 0.2 điểm % là hòa; hòa thì chọn η có AUC validation lớn hơn ở ring-α0.1.
  - τ = 1, τ = 50: η tốt nhất theo cùng quy tắc trên 2 ô; nếu η chung cách giá trị tốt nhất không quá 0.2 điểm % thì coi η chung là tốt nhất. Khác η chung thì kích hoạt T19c (12 lượt bổ sung ở Tier 2).
  - η của B1: accuracy validation lớn nhất trong 3 lượt B1 (hòa 0.2 điểm % thì AUC lớn hơn).
  - μ của DFedProx không chọn bằng lượt chạy: μ = 0.1/(20η) và 1/(20η), giữ ημτ ∈ {0.1, 1} ở τ = 20 (SPEC QS-2).
- **code_version** (hash nội dung src/ + file khóa): `00fe7f198c9a156bfac710d4d2edfcea9ea1830143a43b752a9d3a2904d46461`, ở commit của nhánh `binh` chứa thẻ này.
- **Lệnh** (từ worktree của commit lưới, `results/` là symlink tới results/ của repo chính; SPEC §5.3):
  - `git worktree add -d ../dfl-grid <commit>`; trong worktree: `ln -s <repo>/results results`
  - `python -m dfl sweep --experiment configs/experiments/e0a_lr.yaml --dry-run`, nhóm duyệt (lưới hơn 10 lượt)
  - `python -m dfl sweep --experiment configs/experiments/e0a_lr.yaml --workers 6` (docs/PERF.md: số lõi rảnh trừ 1)
  - `python -m analysis.select_lr --results results/ --experiment e0a_lr`
- **Ngân sách:** khoảng 7.3 giờ-lõi (docs/PERF.md §5), tức khoảng 1.5 giờ với 6 worker.
- **Tiêu chí hợp lệ:** cả 24 run_id có attempt `ok` hoặc `diverged`, cùng một code_version; `final_eval_split` là validation ở mọi lượt; attempt `failed` giữ nguyên và ghi nguyên nhân ở đây.
- **Điều kiện trước khi chạy** (T14b): Gate G1 (ADR Accepted); `make check-inv` xanh; cây sạch; nhóm duyệt kết quả `--dry-run`.
- **Kết quả** (điền sau khi chạy): η chung, η của B1, μ, η tốt nhất của τ = 1 và τ = 50, kèm ngày và run ID; ghi vào ADR-007 (nhóm duyệt diff) trước E0b.

## Kết quả (chạy ngày 2026-10-16)

- **Chạy:** người dùng xác nhận Gate G1, thẻ, kết quả `--dry-run` (24 lượt) và commit lưới trên nhánh `binh` ngày 2026-10-16. Sweep chạy từ worktree `/root/ecaep/dfl-grid` ở commit `50fef80`, `--workers 6` (đo được khoảng 7.6 lõi rảnh trên 8 ngay trước khi chạy), từ 12:14 tới 13:30 (khoảng 1 giờ 17 phút).
- **Hợp lệ:** 24/24 run_id có attempt `ok` (24 attempt, không có `failed` hay `diverged`), cùng code_version `00fe7f19…6461`; mọi lượt có `final_eval_split = validation`, `final_eval_size = 5000`. SAN-5 đạt cả 24 lượt (`python -m analysis.sanity --results results/ --san5 e0a_lr`).
- **Lệnh chọn:** `python -m analysis.select_lr --results results/ --experiment e0a_lr`:
  - **η chung = 0.1.** Accuracy validation nhỏ nhất qua 3 ô τ = 20: η = 0.01 → 0.6656, η = 0.05 → 0.7625, η = 0.1 → 0.7879 (cả ba lần đều là ô ring-#C2); không hòa.
  - **τ = 1:** η tốt nhất 0.1; **τ = 50:** η tốt nhất 0.1. Không kích hoạt T19c.
  - **η của B1 = 0.1.**
  - **μ của DFedProx (QS-2):** 0.05 (ημτ = 0.1) và 0.5 (ημτ = 1).
- **Accuracy validation** (trung bình `acc_mean` ở 5 mốc cuối, tập con 2,000 ảnh validation) / AUC validation, theo run_id:

| Ô | τ | η = 0.01 | η = 0.05 | η = 0.1 |
|---|---|---|---|---|
| FC-IID | 20 | 0.9790 / 0.9542 (`5423c20605d3b7e9`) | 0.9898 / 0.9825 (`ce80927bb070ac6d`) | 0.9916 / 0.9867 (`4bb6432c2d64dcd6`) |
| ring-α0.1 | 20 | 0.8561 / 0.7368 (`a5ea7fdd638ca930`) | 0.9278 / 0.8483 (`eb33452fb0731501`) | 0.9468 / 0.8718 (`f99dc383062c06ba`) |
| ring-#C2 | 20 | 0.6656 / 0.5315 (`385a7c4bb5e5e8eb`) | 0.7625 / 0.6339 (`76332f2c61c62206`) | 0.7879 / 0.6550 (`46550694fd2fd73d`) |
| FC-IID | 1 | 0.9760 / 0.9528 (`4118c37644fcd3be`) | 0.9894 / 0.9826 (`7100b4dcd82a3453`) | 0.9884 / 0.9845 (`ae6ec9fc53d345b2`) |
| ring-α0.1 | 1 | 0.9351 / 0.8648 (`81378a8ba1c38d7b`) | 0.9707 / 0.9418 (`1ae088051eb58c9f`) | 0.9765 / 0.9561 (`3d136dc6545126f8`) |
| FC-IID | 50 | 0.9785 / 0.9540 (`7ca678ae8ed5cd12`) | 0.9894 / 0.9824 (`75734c3d33bbf0ec`) | 0.9919 / 0.9865 (`a84a7b4823a3b5fa`) |
| ring-α0.1 | 50 | 0.8037 / 0.6691 (`6c0e6508f4af31cb`) | 0.8867 / 0.7800 (`659d65f4e1288bb5`) | 0.9060 / 0.7972 (`1f24718a65053094`) |
| B1 (τ·R = 17,200) | 20 | 0.9881 / 0.9755 (`c9ab944c4756f566`) | 0.9915 / 0.9861 (`d5d1b9e46d2cbcaf`) | 0.9922 / 0.9876 (`a26d22dce62c276a`) |

- **Ghi chú (mô tả, không đổi lựa chọn):**
  - η = 0.1 là đầu trên của dải đã đăng ký, và accuracy còn tăng từ 0.05 lên 0.1 ở mọi ô Non-IID. Không quét thêm η > 0.1: quy tắc chỉ chọn trong {0.01, 0.05, 0.1} (ADR-007), và chọn lại siêu tham số sau khi xem kết quả là việc SPEC §8 cấm. Báo cáo nêu điều này như một giới hạn.
  - Ở FC-IID τ = 1, η = 0.05 (0.9894) hơn η = 0.1 (0.9884) 0.1 điểm %, trong ngưỡng hòa 0.2 điểm %, nên η chung được coi là tốt nhất cho τ = 1.
  - Ring-#C2 thấp hơn hẳn (0.79 ở η tốt nhất), đúng kỳ vọng của ô Non-IID khó nhất; đây là ô quyết định η theo quy tắc trường hợp xấu nhất.
- **ADR-007:** η, μ, η của B1 đã ghi vào mục Decision; nhóm duyệt diff ngày 2026-10-16.
