# ADR-011: Thang bốn model tăng dần độ phức tạp cho thí nghiệm chi phí giao tiếp E8; SmallCNN vẫn là model của thí nghiệm lõi

## Status
Proposed — nhóm soạn ngày 2026-10-01 theo trả lời Q1 của GVHD; chờ nhóm duyệt và GVHD xác nhận Q11 (OPEN_QUESTIONS.md). Bổ sung ADR-004, không thay thế nó.

## Date
2026-10-01

## Context
- GVHD (2026-10-01): "Chạy mấy mô hình đơn giản, sau mình sẽ tăng độ phức tạp lên xem communication cost sẽ thay đổi như thế nào." Nhóm hiểu "độ phức tạp" là kích thước model, tức số tham số. Q11 hỏi lại GVHD; nếu GVHD muốn tăng số node hay giới hạn băng thông thì thêm trục đó vào E8.
- Byte mỗi message = 4 × số tham số (float32, REQ-05). Theo số byte, chi phí giao tiếp tỉ lệ đúng với kích thước model: điều này đúng theo định nghĩa, chưa phải một phát hiện. Phần cần đo là:
  - thời gian giao tiếp thật trên process runtime (ADR-010) khi model lớn dần;
  - tỉ lệ giao tiếp trên tính toán;
  - accuracy đạt được theo byte đã tiêu.
- ADR-004 đã xét hai model chuẩn của McMahan et al. 2017 và không chọn chúng cho lưới lõi vì tốn byte và tốn CPU: 2NN-MLP (199,210 tham số) và CNN (1,663,370 tham số). Hai model này hợp làm các bậc trên của thang.
- Ngân sách: E8 chạy ở N = 8, ít round, song song với Tier 1 trên cùng máy chủ dùng chung.

## Decision
- **Thang bốn model.** Không model nào có BatchNorm hay dropout (G3). Mọi model được tạo không tiêu thụ RNG toàn cục, với w⁰ lấy từ luồng init (T09a).

  | Mã | Model | Số tham số | Byte/message |
  |---|---|---|---|
  | M0 | Logistic regression 784 → 10 | 7,850 | 31,400 |
  | M1 | SmallCNN (ADR-004, model lõi) | 46,730 | 186,920 |
  | M2 | 2NN-MLP 784 → 200 → 200 → 10, ReLU (McMahan et al. 2017) | 199,210 | 796,840 |
  | M3 | CNN của McMahan et al. 2017: Conv 5×5 (1→32, padding 2) → ReLU → MaxPool 2 → Conv 5×5 (32→64, padding 2) → ReLU → MaxPool 2 → FC 3136 → 512 → ReLU → FC 512 → 10 | 1,663,370 | 6,653,480 |

  Từ M0 tới M3, byte mỗi message tăng khoảng 212 lần. Số tham số được kiểm bằng test như SmallCNN (T09c).
- **E8: chi phí giao tiếp theo độ phức tạp model.** Thẻ tiền đăng ký `docs/experiments/E8.md` được commit trước khi chạy (EXP-1).
  - Chạy trên process runtime trong Docker (ADR-010), N = 8 container node và một container observer.
  - DFedAvg, τ = 20, R = 50 (1,000 bước cục bộ), η chung chọn ở E0a.
  - Lưới: {M0, M1, M2, M3} × {T1 ring, T3 gossip, T4 FC} × {IID, α = 0.1} × 3 seed = **72 lượt**. Lưới hơn 10 lượt nên phải trình kết quả `--dry-run` cho nhóm trước khi chạy.
- **Độ đo của E8:**
  - Byte mỗi round: tất định, kiểm bằng message × 4 × số tham số.
  - Thời gian giao tiếp mỗi round (`comm_s`) và thời gian tính (`train_s`), lấy trung vị qua round, cùng tỉ lệ `comm_s / (comm_s + train_s)`.
  - Số byte mạng của container (`docker stats`), làm phép kiểm độc lập với CommLedger.
  - Accuracy cuối, và bytes@90.
  - Tải trung bình của máy chủ, ghi vào manifest để đọc thời gian đúng bối cảnh.
- **Kết quả kỳ vọng ghi trước (H8):**
  - Byte/round đúng bằng message × 4 × số tham số ở mọi ô.
  - Tỉ lệ byte FC/ring bằng (N − 1)/2 = 3.5 ở mọi model.
  - `comm_s` tăng theo số tham số.
  - Phần mang tính khám phá: với cùng số tham số, tỉ lệ giao tiếp trên tính toán của MLP (M2) cao hơn của CNN, vì CNN tính nhiều hơn trên mỗi tham số.
  - η không được chọn lại cho từng model, nên accuracy của M0, M2, M3 chỉ là kết quả phụ; báo cáo ghi rõ điều này.

## Alternatives Considered

### Tăng độ rộng của một MLP theo nhiều mức (ví dụ 2⁴ … 2¹⁰ đơn vị ẩn)
- Ưu: một tham số liên tục, dễ vẽ.
- Nhược: không có model chuẩn của tài liệu FL để đối chiếu; chỉ một họ kiến trúc nên không thấy khác biệt giữa MLP và CNN.
- Kết luận: không chọn. Nếu GVHD muốn nhiều điểm hơn thì thêm M1.5 (MLP 784 → 64 → 10, 50,890 tham số).

### Model lớn hơn (ResNet, CIFAR-10)
- Ưu: chi phí giao tiếp lớn hơn nữa.
- Nhược: không chạy nổi trên CPU dùng chung trong ngân sách; cần dataset khác (ADR-003).
- Kết luận: ngoài E8. CIFAR-10 vẫn là phần mở rộng sau M4.

### Nén hay lượng tử hóa để giảm chi phí
- Kết luận: ngoài phạm vi (QĐ-5).

## Consequences
- **Code** (T09c, phải xong trước E0a):
  - `learning/models.py` có M0, M2 và M3 bên cạnh SmallCNN, mỗi model có test số tham số và test "không có BatchNorm";
  - cấu hình có trường `model`;
  - lõi vẫn dùng SmallCNN, nên REQ-05 và T-INV-6 (4 × 46,730) giữ nguyên.
- **Analysis** (T27c) có hai hình mới:
  - P10: byte/round và `comm_s` theo số tham số, trục log, mỗi đường một topology;
  - P11: accuracy theo byte tích lũy, mỗi đường một model.
- **Tính toán:** M3 tốn CPU nhất, ước tính 40–100 ms một bước. Phải đo lại ở T13 trước khi chốt R của E8.
- **SPEC cần sửa sau khi nhóm duyệt (Ask first):**
  - §1.4: thang model chỉ cho E8;
  - REQ-16;
  - E8 ở §9.5, với Tier bắt buộc theo yêu cầu của GVHD;
  - hai hình P10, P11 ở REQ-10.
