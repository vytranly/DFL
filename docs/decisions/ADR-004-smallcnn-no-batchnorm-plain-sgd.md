# ADR-004: SmallCNN 46,730 tham số, không BatchNorm, SGD thuần, minibatch lấy có hoàn lại

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30

## Context
- Ngân sách chỉ CPU: một lượt gồm 20 node × 4,000 bước cục bộ × batch 32; toàn bộ khoảng 263 lượt (KH §3.7).
- Kích thước model quyết định chi phí giao tiếp: byte = số message × 4 × số tham số (REQ-05).
- BatchNorm là nguồn lỗi đã biết khi dữ liệu Non-IID. Hsieh et al. 2020 thấy thay BatchNorm bằng GroupNorm cải thiện accuracy thêm 10.7 và 19.8 điểm trong thí nghiệm của họ; lưu ý bối cảnh của bài là parameter server với dữ liệu chia theo phân vùng, không phải topology P2P (DIGEST §2.7). Buffer của BatchNorm, gồm cả buffer số nguyên `num_batches_tracked`, cũng không trung bình hóa đúng được (G3).
- Node có thể rất nhỏ: ở α = 0.05, node nhỏ nhất chỉ khoảng 168 mẫu (KH §3.2). Cách lấy minibatch phải định nghĩa rõ với mọi kích thước node.
- Nguồn: KH §1.1 (điều chỉnh 4), §3.3, §3.5, §10; SPEC §3.1 (`learning`); DIGEST §2.7.

## Decision
- **SmallCNN:** Conv(1→16, 5×5) → ReLU → MaxPool 2 → Conv(16→32, 5×5) → ReLU → MaxPool 2 → FC(512→64) → ReLU → FC(64→10). Tổng **46,730 tham số**, tức 0.187 MB mỗi message (float32). Không BatchNorm, không dropout. Nếu sau này cần chuẩn hóa thì dùng GroupNorm và ghi ADR mới.
- **Optimizer:** SGD, momentum 0, weight decay 0. lr η chọn ở E0a (ADR-007) và dùng chung cho mọi τ.
- **Minibatch:** B = 32, lấy **có hoàn lại** từ dữ liệu của node, không `drop_last`. Khi chia dữ liệu, `min_size` = 64 = 2B (G17).
- **Chuẩn hóa ảnh:** mean/std tính từ tập train (MNIST khoảng 0.1307 / 0.3081).

## Alternatives Considered

### CNN của McMahan et al. 2017 (1,663,370 tham số)
- Ưu: model chuẩn trong tài liệu FL.
- Nhược: byte/message gấp khoảng 35 lần; tính toán nặng hơn nhiều trên CPU.
- Kết luận: không chọn.

### 2NN-MLP của McMahan et al. 2017 (199,210 tham số)
- Ưu: đơn giản.
- Nhược: nhiều tham số gấp khoảng 4 lần SmallCNN; là MLP nên không khai thác cấu trúc không gian của ảnh.
- Kết luận: không chọn.

### Model có BatchNorm
- Ưu: huấn luyện tập trung nhanh và ổn định hơn.
- Nhược: hỏng nặng dưới Non-IID (Hsieh et al. 2020); buffer thống kê và buffer số nguyên không tổng hợp đúng.
- Kết luận: không chọn (G3).

### SGD có momentum, hoặc Adam
- Ưu: hội tụ nhanh hơn khi huấn luyện tập trung.
- Nhược: trạng thái optimizer không được trao đổi giữa các node, nên thêm một biến gây nhiễu khi so sánh thuật toán. DFedAvg dùng ở đây là bản không momentum (Sun et al. 2022), và lý thuyết decentralized SGD dùng SGD thuần.
- Kết luận: không chọn.

### Minibatch không hoàn lại, theo epoch
- Ưu: phổ biến.
- Nhược: "một epoch" dài ngắn khác nhau giữa các node có kích thước lệch nhau; khó khớp với τ bước cố định.
- Kết luận: không chọn.

## Consequences
- Chi phí mỗi round với N = 20: ring 40 message = 7.48 MB; FC 380 message = 71.03 MB; gossip trên $K_N$ 20 message = 3.74 MB (KH §3.3).
- Kỳ vọng accuracy khi huấn luyện tập trung khoảng 99%; ngưỡng sanity SAN-1 là ≥ 98.5% (SPEC §9.3).
- T09 phải kiểm đúng 46,730 tham số và không có module BatchNorm; T-INV-6 kiểm byte = message × 4 × số tham số.
- CIFAR-10 (nếu làm, ADR-003) cần model khác và ADR mới.
