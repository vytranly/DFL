# ADR-002: Mô phỏng đơn tiến trình, đồng bộ, message-passing qua giao diện Transport

## Status
Proposed — chờ Gate G1 và câu trả lời Q1 của GVHD.

## Date
2026-09-30 (cập nhật theo hai vòng phản biện P1.9 cùng ngày)

## Context
- REQ-01 yêu cầu "xây dựng hệ thống peer-to-peer", không có server trung tâm.
- Các độ đo đề yêu cầu (accuracy/F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định, phương sai giữa node) không cần độ trễ mạng thật. Chi phí giao tiếp đo được bằng số message × kích thước model.
- Kết quả phải tái lập được: cùng cấu hình + seed + `code_version` + máy + số luồng thì cùng `rounds.jsonl` (INV-6). Lưới khoảng 280 lượt (Tier 1 + 2, SPEC EXP-2, EXP-3) phải chạy trên một máy chủ CPU 8 nhân (QĐ-2); máy dùng chung với người khác, lúc đo ngày 2026-09-30 tải trung bình khoảng 7.
- Đo ngày 2026-09-30: một bước huấn luyện SmallCNN (B = 32) mất 10.5 ms với 1 luồng và 10.0 ms với 8 luồng. Với model nhỏ như vậy, nhiều luồng không làm nhanh hơn; chạy song song nhiều lượt một luồng mới tận dụng được máy.
- A1 (mô phỏng đơn tiến trình được coi là "hệ thống P2P") chưa được GVHD xác nhận (Q1). Nhóm đã chốt phương án: nếu GVHD yêu cầu P2P thật thì T22 thành bắt buộc (OQ Q1).
- Nguồn: KH §1.1 (điều chỉnh 5), §3.1, §4.3, §4.5, §4.9, §10; SPEC §1.4, §3, §5.2; OQ Q1; `docs/decisions/REVIEW-P1.9.md`.

## Decision
- Toàn bộ N node chạy trong một tiến trình, đồng bộ theo round, theo vòng ATC ở ADR-006.
- Node chỉ trao đổi model qua giao diện `Transport` (`send` / `inbox`). Bản mặc định chạy in-memory, không deep-copy: Transport chuyển tham chiếu tới snapshot sở hữu riêng của người gửi, và người nhận chỉ được đọc snapshot. CommLedger bên trong Transport là nơi duy nhất đếm message và byte, tính tổng và theo từng node, lấy số byte từ `Snapshot.nbytes` (INV-2); không có message node tự gửi cho mình.
- Node chỉ đọc model của chính nó và inbox của nó (INV-1). Engine không bao giờ đưa đại lượng toàn cục (ví dụ $\bar w$) vào huấn luyện; các đại lượng đó chỉ Observer tính để đo (INV-5). Ngoại lệ duy nhất về đại lượng tính từ dữ liệu hay model của node khác là n̄ của biến thể size-corrected (ADR-005).
- **Tri thức chung của mọi node:** danh sách id 0..N−1 và N; đồng hồ round đồng bộ; seed chung, dùng cho lịch gossip (ADR-009); $w^0$ chung (A3, ADR-006); bậc của láng giềng, mà MH cần và được trao đổi một lần lúc dựng topology tĩnh. Ở đường server (B3, E1c), server biết $n_k$ của mọi node như FedAvg gốc; đó là baseline có server, không phải DFL. Các siêu dữ liệu này không được đếm vào byte (SPEC REQ-05). Đây là giả định chuẩn của DFL đồng bộ, không phải đại lượng tính từ dữ liệu hay model của node khác. T3 còn giả định mọi node biết địa chỉ của mọi node khác. Báo cáo liệt kê các giả định này trong mục hạn chế (SPEC INT-4).
- Bước tổng hợp đọc từ snapshot của round, cộng theo id người gửi tăng dần, ghi ra bộ đệm mới, và chỉ ghi vào model sau khi mọi node đã tổng hợp xong (INV-3). Nhờ vậy kết quả trùng từng bit bất kể thứ tự duyệt node (T-INV-4).
- **Tính tất định trên CPU:** số luồng mỗi lượt là hằng số của cả lưới (mặc định 1, chốt ở T13) và nằm trong cấu hình chuẩn hóa, tức trong `run_id`. `--workers` chỉ quyết định số lượt chạy song song, không ảnh hưởng kết quả. Không có nhánh code cho GPU (QĐ-5).
  - Mỗi lượt chạy trong một tiến trình con mới (`subprocess` gọi `python -m dfl run`, hoặc `multiprocessing` với `spawn`). Không dùng `fork`: tài liệu Python 3.12 ghi fork một tiến trình nhiều luồng là không an toàn (https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods), mà tiến trình cha có thể đã khởi tạo thread pool của OpenMP.
  - `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` được đặt trong môi trường của tiến trình con trước khi nó khởi động. `dfl run` gọi `torch.set_num_threads(k)` trước mọi phép tính torch (https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html) và ghi vào manifest giá trị đọc lại bằng `torch.get_num_threads()`. Tài liệu torch 2.14 mà dự án đã đọc không nói biến môi trường phải được đặt trước khi import (UNVERIFIED); thiết kế này đặt chúng trước khi tiến trình khởi động nên không phụ thuộc điểm đó.
- `TorchDistTransport` (T22, nếu làm) đặt ở `dfl.sim`, không ở `dfl.network`, vì phải import torch (contract `pure-no-torch`, ENG-6).

## Alternatives Considered

### Đa tiến trình với `torch.distributed` (backend gloo)
- Ưu: gần P2P thật hơn, mỗi tiến trình là một node. Tài liệu PyTorch 2.14 ghi gloo hỗ trợ `send`/`recv` trên CPU (https://docs.pytorch.org/docs/2.14/distributed.html).
- Nhược: chậm hơn và tốn RAM hơn với N = 20 trên 8 nhân; thứ tự nhận message không tất định nếu không đồng bộ chặt; khó test hơn.
- Kết luận: giữ làm tùy chọn (T22, `TorchDistTransport`) sau cùng giao diện Transport, chỉ để demo và kiểm tra tương đương ở N nhỏ (4–8); không dùng cho lưới thí nghiệm.

### Nhiều máy thật và đo độ trễ mạng
- Ưu: thực tế nhất.
- Nhược: ngoài phạm vi (QĐ-5); tốn công hạ tầng; không trả lời thêm câu hỏi nào của đề.
- Kết luận: không chọn.

### Gossip bất đồng bộ
- Ưu: gần gossip learning thực tế (Hegedűs et al. 2021).
- Nhược: khó tất định; khó so sánh công bằng với các thuật toán đồng bộ.
- Kết luận: phần mở rộng sau M4 (SPEC §1.4).

### Số luồng suy ra từ số worker (ví dụ 8 nhân chia cho `--workers`)
- Ưu: tự tận dụng hết máy.
- Nhược: chạy lại với số worker khác thì số luồng khác, kết quả có thể lệch từng bit; một hình có thể trộn các lượt có số luồng khác nhau.
- Kết luận: không chọn; số luồng là hằng số của cả lưới.

## Consequences
- Chạy nhanh và tất định; kiểm được bằng T-INV-2, T-INV-4, T-INV-5, T-INV-12.
- Chi phí giao tiếp là số đếm (message, byte, tải lớn nhất trên một node), không có thời gian truyền. Báo cáo phải nêu rõ điều này.
- Giao diện Transport cho phép thêm backend đa tiến trình mà không phải sửa engine.
- **Chờ Q1 — phương án đề xuất nếu GVHD yêu cầu P2P thật:**
  - Hết tuần 3 (trước C2) chưa có trả lời thì nhóm hỏi lại GVHD. Hết tuần 5 (trước E0a ở tuần 7, đủ hai tuần cho T22) vẫn chưa có trả lời thì vẫn chạy lưới theo quyết định này, để không trễ lịch. Im lặng không phải là đồng ý: cách hiểu "P2P" vẫn là rủi ro của REQ-01 và được ghi trong mục hạn chế. Khi đó T22 làm ngay sau T13 (tuần 8–10), không còn là mở rộng tùy chọn (SPEC QS-10, nhóm chốt ngày 2026-09-30). T22 viết ở cây làm việc hằng ngày; lưới chạy từ worktree riêng nên không bị ảnh hưởng (ADR-008).
  - Giữ nguyên quyết định này cho lưới thí nghiệm (tất định, vừa ngân sách CPU).
  - Tạo một ADR mới *bổ sung* (không thay thế ADR này): T22 thành bắt buộc, gồm `TorchDistTransport` dùng gloo, 4–8 tiến trình, test tương đương với bản in-memory ở N = 4, dùng cho demo và phần kiến trúc của báo cáo.
  - T22 phải xong trước E0a (T14). Mọi thay đổi trong `src/` đều làm đổi `code_version`, nên làm T22 giữa chừng sẽ buộc chạy lại những lượt đã có.
  - Nếu GVHD yêu cầu mọi thí nghiệm phải chạy đa tiến trình, ADR mới đó sẽ *thay thế* ADR này (đánh dấu "Superseded by ADR-00M"), và phải tính lại ngân sách ở ADR-007.
- **Trade-off đã chấp nhận:** giao diện Transport giữ nguyên, nhưng `TorchDistTransport` phải tự tuần tự hóa state thành tensor và dùng gửi/nhận không chặn để tránh deadlock. Bảng backend của tài liệu 2.14 chỉ ghi `send`/`recv` cho gloo, không ghi `isend`/`irecv`, nên T22 phải kiểm lại.
