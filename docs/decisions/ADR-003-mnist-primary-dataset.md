# ADR-003: MNIST là dataset chính; CIFAR-10 là mở rộng ưu tiên nếu chạy nổi trên CPU; Fashion-MNIST là phương án kiểm chứng rẻ

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30

## Context
- Đề gợi ý "MNIST / Fashion-MNIST (dễ quan sát); CIFAR-10 (thách thức hơn)" (REQ-14). Nhóm đã chốt PyTorch + torchvision + MNIST.
- Chỉ có máy chủ CPU 8 nhân, không GPU (Q4, QĐ-5). Lưới khoảng 263 lượt được ước tính 44–57 giờ-CPU với MNIST (KH §3.7).
- MNIST có trần khoảng 99%, nên khác biệt giữa các điều kiện chỉ lộ rõ ở Non-IID mạnh (rủi ro "MNIST quá dễ", KH §9).
- Q5: CIFAR-10 không bắt buộc, làm tiếp nếu đã xong MNIST và vẫn khả thi. QĐ-4: ưu tiên CIFAR-10 nếu T13 cho thấy chạy nổi trên CPU; Fashion-MNIST (E6) giữ làm phương án kiểm chứng rẻ. Điểm này đổi thứ tự mở rộng so với bản tóm tắt ở KH §10, nơi Fashion-MNIST đứng trước.
- Nguồn: KH §1.1 (điều chỉnh 2 và 3), §3.7, §9, §10; SPEC §1.4; OQ Q5, QĐ-4.

## Decision
- MNIST (qua `torchvision.datasets.MNIST`) là dataset của mọi thí nghiệm Tier 1 và Tier 2. 60k ảnh train được chia thành 55k cho các node và 5k validation phân tầng (chỉ dùng để chọn lr, ADR-007); 10k ảnh test là tập test toàn cục chung.
- Phần mở rộng, chỉ làm sau M4:
  1. **CIFAR-10**, nếu số đo ở T13 cho thấy một lưới thu gọn chạy được trên CPU trong thời gian còn lại. Trước khi bắt đầu phải có ADR mới cho model (ảnh 3×32×32, vẫn không BatchNorm), lưới thu gọn và ngân sách.
  2. **Fashion-MNIST** (E6: {T1–T4} × {IID, α = 0.1, #C = 2}, 3 seed, 36 lượt) là phương án kiểm chứng rẻ, chọn khi CIFAR-10 không khả thi. Dùng lại toàn bộ code vì cùng shape 1×28×28 và cùng giao diện torchvision.

## Alternatives Considered

### Chỉ MNIST, không kiểm chứng trên dataset khác
- Ưu: ít việc nhất.
- Nhược: dễ bão hòa; kết luận chỉ dựa trên một dataset dễ.
- Kết luận: không chọn làm kế hoạch cuối. Rủi ro được giảm bằng độ đo nhạy (R@x, bytes@x, drift, consensus distance, acc^loc), các mức Non-IID cực đoan (#C = 1, 2) và phần mở rộng ở trên.

### CIFAR-10 làm dataset chính
- Ưu: thách thức hơn, khác biệt giữa topology và thuật toán lộ rõ hơn.
- Nhược: cần model lớn hơn và GPU để chạy đủ lưới; không vừa ngân sách CPU (QĐ-5).
- Kết luận: không chọn làm chính; là mở rộng ưu tiên (QĐ-4).

### Fashion-MNIST làm dataset chính
- Ưu: khó hơn MNIST, cùng shape, cùng giao diện torchvision.
- Nhược: nhóm đã chốt MNIST; đề gợi ý MNIST trước.
- Kết luận: không chọn làm chính; giữ làm phương án kiểm chứng rẻ.

## Consequences
- Mọi thí nghiệm lõi vừa ngân sách CPU; tập train MNIST dạng uint8 chỉ khoảng 47 MB, nằm gọn trong RAM.
- Phải báo cáo các độ đo nhạy, không chỉ accuracy cuối, để tránh kết luận "không khác biệt" chỉ vì chạm trần.
- Cách chia 55k/5k được kiểm ở T03; quy tắc không đụng tập test khi chọn siêu tham số nằm ở ADR-007.
- Muốn làm CIFAR-10 phải có ADR mới (model, lưới thu gọn, ngân sách); không tự thêm vào lưới hiện tại.
