# ADR-001: Tự viết mô phỏng DFL trên PyTorch + NetworkX, không dùng framework FL có sẵn

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30

## Context
- Đề 2 là nghiên cứu thực nghiệm có hệ thống về DFL không server (REQ-01). Dự án phải so sánh nhiều topology (REQ-07) và thuật toán (REQ-08), và đo chi phí giao tiếp (REQ-05, được tô vàng trong đề). Vì vậy cần kiểm soát chính xác topology theo từng round, ma trận trộn, thứ tự cập nhật và cách đếm message/byte.
- Nhóm đã chốt PyTorch + torchvision + MNIST. torchvision có `torchvision.datasets.MNIST` và `FashionMNIST` với cùng giao diện, nên đổi dataset chỉ là đổi cấu hình (KH §1.1).
- Ràng buộc: 3 người, 12 tuần, chỉ CPU (OQ: Q4, QĐ-5). Q7 (nhóm trả lời): không bắt buộc dùng Flower. Tin nhắn gửi GVHD đã nêu lựa chọn này và mời góp ý; nếu GVHD muốn dùng Flower thì xem lại ADR này.
- Nguồn: KH §1.1 (bảng so sánh framework), §10; SPEC §2, §3; OQ Q7.

## Decision
Tự viết một bộ mô phỏng mỏng:
- PyTorch 2.14 (bản CPU) cho model và huấn luyện; torchvision 0.29 cho dữ liệu; NetworkX 3.7 cho topology; numpy cho chia dữ liệu và ma trận trộn.
- Lớp Transport + CommLedger tự viết là đường duy nhất để node trao đổi model, và là nơi duy nhất đếm message/byte (INV-2, ADR-002).
- Module theo capability map ở SPEC §3: `data`, `network`, `learning`, `metrics-io`, `sim`, `analysis`.
- Không phụ thuộc framework FL nào; các framework DFL có sẵn chỉ dùng để tham khảo hoặc đối chiếu kết quả.

## Alternatives Considered

### Flower (flwr 1.30)
- Ưu: hệ sinh thái lớn, có baseline NIID-Bench.
- Nhược: kiến trúc xoay quanh server (ServerApp/ClientApp, strategy tổng hợp đặt ở server). Muốn chạy ring hay gossip phải "bẻ" kiến trúc; các nỗ lực như flwr-serverless hay GLow đều nằm ngoài Flower.
- Kết luận: không chọn.

### Framework DFL có sẵn: p2pfl, NEBULA (Fedstellar), DecentralizePy
- Ưu: có P2P thật (gRPC/socket) và nhiều topology.
- Nhược: nặng (Docker, đa tiến trình); khó kiểm soát chính xác thứ tự cập nhật và cách đếm chi phí; tốn thời gian học công cụ.
- Kết luận: chỉ tham khảo; có thể dùng để đối chiếu kết quả.

### gossipy
- Ưu: mô phỏng gossip learning bằng Python.
- Nhược: kém linh hoạt cho D-PSGD và DFedProx đồng bộ.
- Kết luận: chỉ tham khảo.

### Tự viết P2P thật bằng gRPC hoặc ZeroMQ
- Ưu: P2P thật, chạy được nhiều tiến trình hay nhiều máy.
- Nhược: tốn công; không phục vụ câu hỏi nghiên cứu (đề không yêu cầu đo độ trễ); khó tất định.
- Kết luận: không chọn. Nếu GVHD yêu cầu P2P thật thì dùng `torch.distributed` sau cùng giao diện Transport (ADR-002).

## Consequences
- Kiểm soát hoàn toàn topology, trọng số trộn, thứ tự cộng và cách đếm byte. Code nhỏ (ước tính 1.5–2.5 nghìn dòng), dễ debug, chạy nhanh trên CPU.
- Nhóm phải tự viết và tự kiểm thử phần mô phỏng. Rủi ro lỗi mô phỏng được giảm bằng bộ test bất biến T-INV-1…12 (SPEC §9.2), phải xanh trước khi chạy lưới.
- Không có số đo độ trễ mạng thật: chi phí giao tiếp đo bằng số message × kích thước model (ADR-002, ADR-004).
- Báo cáo phải giải thích vì sao không dùng Flower và các framework DFL (đã thêm vào bản nháp docx, mục D17).
- API framework dùng trong code phải được đối chiếu với tài liệu chính thức đúng phiên bản và ghi URL (P1.7).
