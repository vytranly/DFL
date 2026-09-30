# ADR-002: Mô phỏng đơn tiến trình, đồng bộ, message-passing qua giao diện Transport

## Status
Proposed — chờ Gate G1 và câu trả lời Q1 của GVHD.

## Date
2026-09-30

## Context
- REQ-01 yêu cầu "xây dựng hệ thống peer-to-peer", không có server trung tâm.
- Các độ đo đề yêu cầu (accuracy/F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định, phương sai giữa node) không cần độ trễ mạng thật. Chi phí giao tiếp đo được bằng số message × kích thước model.
- Kết quả phải tái lập được: cùng cấu hình + seed + `code_version` + máy + số luồng thì cùng `rounds.jsonl` (INV-6). Lưới khoảng 263 lượt phải chạy trên một máy chủ CPU 8 nhân (QĐ-2).
- A1 (mô phỏng đơn tiến trình được coi là "hệ thống P2P") chưa được GVHD xác nhận (Q1). Nhóm đã chốt phương án: nếu GVHD yêu cầu P2P thật thì T22 thành bắt buộc (OQ Q1).
- Nguồn: KH §1.1 (điều chỉnh 5), §3.1, §4.3, §4.5, §4.9, §10; SPEC §1.4, §3; OQ Q1.

## Decision
- Toàn bộ N node chạy trong một tiến trình, đồng bộ theo round, theo vòng ATC ở ADR-006.
- Node chỉ trao đổi model qua giao diện `Transport` (`send` / `inbox`). Bản mặc định chạy in-memory, không deep-copy: Transport chuyển tham chiếu tới snapshot sở hữu riêng của người gửi. CommLedger bên trong Transport là nơi duy nhất đếm message và byte, tính tổng và theo từng node (INV-2); không có message node tự gửi cho mình.
- Node chỉ đọc model của chính nó và inbox của nó (INV-1). Engine không bao giờ đưa đại lượng toàn cục (ví dụ $\bar w$) vào huấn luyện; các đại lượng đó chỉ Observer tính để đo (INV-5).
- Bước tổng hợp đọc từ snapshot của round, cộng theo id người gửi tăng dần, và chỉ ghi sau khi mọi node đã tổng hợp xong (INV-3). Nhờ vậy kết quả trùng từng bit bất kể thứ tự duyệt node (T-INV-4).
- Tính tất định trên CPU: số luồng CPU của mỗi worker cố định và ghi vào manifest; không có nhánh code cho GPU (QĐ-5).

## Alternatives Considered

### Đa tiến trình với `torch.distributed` (backend gloo)
- Ưu: gần P2P thật hơn, mỗi tiến trình là một node. Tài liệu PyTorch 2.14 xác nhận gloo hỗ trợ `send`/`recv` trên CPU (https://docs.pytorch.org/docs/2.14/distributed.html).
- Nhược: chậm hơn và tốn RAM hơn với N = 20 trên 8 nhân; thứ tự nhận message không tất định nếu không đồng bộ chặt; khó test hơn.
- Kết luận: giữ làm tùy chọn sau M4 (T22, `TorchDistTransport`) sau cùng giao diện Transport, chỉ để demo và kiểm tra tương đương ở N nhỏ (4–8); không dùng cho lưới thí nghiệm.

### Nhiều máy thật và đo độ trễ mạng
- Ưu: thực tế nhất.
- Nhược: ngoài phạm vi (QĐ-5); tốn công hạ tầng; không trả lời thêm câu hỏi nào của đề.
- Kết luận: không chọn.

### Gossip bất đồng bộ
- Ưu: gần gossip learning thực tế (Hegedűs et al. 2021).
- Nhược: khó tất định; khó so sánh công bằng với các thuật toán đồng bộ.
- Kết luận: phần mở rộng sau M4 (SPEC §1.4).

## Consequences
- Chạy nhanh và tất định; kiểm được bằng T-INV-2, T-INV-4, T-INV-5, T-INV-12.
- Chi phí giao tiếp là số đếm (message, byte, tải lớn nhất trên một node), không có thời gian truyền. Báo cáo phải nêu rõ điều này.
- Giao diện Transport cho phép thêm backend đa tiến trình mà không phải sửa engine.
- **Chờ Q1 — phương án đề xuất nếu GVHD yêu cầu P2P thật:**
  - Giữ nguyên quyết định này cho lưới thí nghiệm (tất định, vừa ngân sách CPU).
  - Tạo một ADR mới *bổ sung* (không thay thế ADR này): T22 chuyển thành bắt buộc. Nội dung: `TorchDistTransport` dùng gloo, 4–8 tiến trình; test tương đương với bản in-memory ở N = 4; dùng cho demo và phần kiến trúc của báo cáo.
  - Nếu GVHD yêu cầu mọi thí nghiệm phải chạy đa tiến trình, ADR mới đó sẽ *thay thế* ADR này (đánh dấu "Superseded by ADR-00M"), và phải tính lại ngân sách ở ADR-007.
