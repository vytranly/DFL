# ADR-010: Mỗi node một tiến trình độc lập (tiến trình cục bộ hoặc container Docker) qua torch.distributed/gloo; bộ mô phỏng một tiến trình được giữ và phải trùng từng bit

## Status
Proposed — nhóm soạn ngày 2026-10-01 theo trả lời Q1 của GVHD; chờ nhóm duyệt và GVHD xác nhận Q10–Q13 (OPEN_QUESTIONS.md). Bổ sung ADR-002, không thay thế nó (phương án đã ghi sẵn trong ADR-002, mục "Chờ Q1"). Nếu GVHD trả lời Q10 là "mọi thí nghiệm phải chạy đa tiến trình", ADR này thay thế ADR-002 và phải tính lại ngân sách ở ADR-007.

## Date
2026-10-01

## Context
- **Trả lời Q1 của GVHD** (tin nhắn ngày 2026-10-01, 17:27): "Cô cần các process chạy song song, mỗi node chạy độc lập. Các em có thể tạo dockers, số lượng ít cũng được. Chạy mấy mô hình đơn giản, sau mình sẽ tăng độ phức tạp lên xem communication cost sẽ thay đổi như thế nào."
- Như vậy A1 (mô phỏng một tiến trình được coi là "hệ thống P2P") không được chấp nhận làm hệ thống duy nhất. ADR-002 đã ghi sẵn phương án cho trường hợp này: T22 thành bắt buộc, gồm gloo với 4–8 tiến trình và test tương đương với bản in-memory, và phải xong trước E0a để `code_version` của lưới không đổi.
- Phần mới so với phương án đó:
  - Docker được gợi ý, với số container nhỏ.
  - Có thêm một trục thí nghiệm mới: tăng độ phức tạp của model rồi quan sát chi phí giao tiếp. Trục này được quyết định ở ADR-011.
- **Ràng buộc máy:**
  - Máy chủ CPU 8 nhân dùng chung. Tải trung bình 7 lúc đo ngày 2026-09-30, và lên tới 24 lúc đo ngày 2026-10-01.
  - RAM 31 GB, nhưng có lúc chỉ còn khoảng 8 GB trống.
  - Máy có Docker 29.4.0 và Docker Compose v2.36.0. torch 2.14 có gloo (`torch.distributed.is_gloo_available()` là True).
  - Lưới Non-IID có khoảng 200 lượt ở N = 20 (SPEC EXP-2).
- **Tài liệu torch 2.14:** bảng backend ghi gloo trên CPU có `send`, `recv`, `barrier`, `gather` và `all_gather`; `isend`/`irecv` có trong tài liệu hàm nhưng không có dòng trong bảng backend:
  https://docs.pytorch.org/docs/2.14/distributed.html
- INV-1…6 và T-INV-1…12 phải giữ nguyên ý nghĩa, dù node chạy ở đâu.

## Decision
- **Một logic node, hai runtime.** Toàn bộ những gì một node làm trong một round đều nằm trong `sim/node.py` (`Node`, `make_snapshot`, `combine`) và LocalTrainer: Adapt, tạo đúng một snapshot, tổng hợp theo id người gửi tăng dần, ghi sau khi tổng hợp xong. Hai runtime chỉ khác nơi node chạy:
  1. **In-process** (ADR-002): `Engine` chạy cả N node trong một tiến trình. Đây là bộ mô phỏng nhanh cho lưới lớn và là oracle cho test.
  2. **Process runtime** (mới): mỗi node là một tiến trình hệ điều hành riêng, có model, dữ liệu, Generator và Transport của riêng nó. Một tiến trình observer riêng chỉ đo. Các tiến trình nói chuyện qua `torch.distributed` với backend gloo trên TCP: https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.init_process_group
- **Giao thức một round (đồng bộ, không có điều phối viên):**
  - Mỗi node tự tính G_t bằng `graph_at(t)` từ seed chung (G18). Nó trao đổi đúng một snapshot với mỗi láng giềng, theo một thứ tự cố định không gây deadlock: với mỗi cặp (i, j), node có id nhỏ hơn gửi trước rồi mới nhận.
  - Nếu T22a xác nhận được `isend`/`irecv` trên gloo thì dùng chúng; nếu không, ghi UNVERIFIED và dùng `send`/`recv`.
  - Sau khi nhận đủ, node tổng hợp bằng chính hàm `combine`. Ở mốc đánh giá, node gửi state và số đếm của mình cho observer. Kênh này thuộc mặt phẳng đo, không được tính vào chi phí giao tiếp (INV-2, INV-5).
- **Transport trong mỗi tiến trình** (`TorchDistTransport`) giữ nguyên giao diện `Transport`. CommLedger của nó đếm message và byte mà chính node đó gửi và nhận. Observer cộng các số đếm này, nên `messages_cum`, `bytes_cum` và `max_node_bytes_cum` có cùng định nghĩa với bản in-process.
- **Hai cách khởi chạy:**
  - Tiến trình cục bộ: `python -m dfl run --runtime processes`, khởi chạy N + 1 tiến trình bằng start method `spawn` trên 127.0.0.1. Cách này dùng cho test và e2e, không cần Docker: https://docs.pytorch.org/docs/2.14/multiprocessing.html
  - Docker Compose: mỗi node một container cộng một container observer, trên một bridge network. N nhỏ (4–8), theo "số lượng ít cũng được". Dùng cho demo và E8: https://docs.docker.com/compose/
- **Tương đương là điều kiện chấp nhận:**
  - Cùng cấu hình, seed và `code_version`, process runtime phải cho `rounds.jsonl` trùng từng byte với bản in-process. Kiểm ở smoke N = 4 bằng tiến trình cục bộ, và ở một lượt N = 8 trong Docker.
  - Phép kiểm này là T-INV-13 (cần sửa SPEC §9.2 và `make check-inv`, Ask first).
  - Trùng từng bit làm được vì gloo chép đúng byte của tensor, mỗi tiến trình chạy một luồng, và phép cộng vẫn theo id người gửi tăng dần.
- **Dùng runtime nào cho thí nghiệm nào (chờ Q10):**
  - Lưới Non-IID N = 20 chạy bằng bộ mô phỏng in-process (ADR-002), dựa vào T-INV-13.
  - Process runtime và Docker dùng cho:
    - E8 (ADR-011);
    - một bản lặp lại của vài ô lõi ở N = 8, làm bằng chứng hệ thống thật cho cùng kết quả;
    - demo.
- **Đo thời gian giao tiếp** (chỉ có ở process runtime):
  - TimingRecord thêm `comm_s`: thời gian chờ gửi và nhận trong round, lấy lớn nhất qua node. Trường này nằm ở `timing.jsonl` vì không tất định (G21).
  - Ở Docker, số byte mạng của từng container (`docker stats`, cột NET I/O) được ghi làm phép kiểm độc lập với CommLedger: payload cộng phần overhead của gloo và TCP. https://docs.docker.com/reference/cli/docker/container/stats/

## Alternatives Considered

### Chạy cả lưới N = 20 bằng tiến trình hay container
- Ưu: mọi kết quả đều đến từ hệ thống đa tiến trình, đúng nhất với câu chữ của GVHD.
- Nhược:
  - 21 tiến trình torch, mỗi tiến trình khoảng 0.3–0.4 GB, tức khoảng 7 GB cho một lượt, trên máy chủ có lúc chỉ còn khoảng 8 GB trống.
  - Đồng bộ mỗi round khiến lượt chạy chậm theo node chậm nhất khi máy chủ đang tải nặng.
  - Khoảng 200 lượt sẽ khó vừa tuần 8–10.
- Kết luận: không chọn mặc định. Code vẫn hỗ trợ (`--runtime processes`); nếu GVHD trả lời Q10 là có thì chuyển sang và tính lại ngân sách.

### Socket tự viết, ZeroMQ hay gRPC
- Ưu: kiểm soát giao thức hoàn toàn.
- Nhược: thêm phụ thuộc chạy (Ask first) và nhiều code mạng phải test, trong khi gloo đã có sẵn trong torch.
- Kết luận: không chọn.

### Nhiều máy thật
- Ưu: thực tế nhất.
- Nhược: GVHD chỉ yêu cầu tiến trình song song, nhóm chỉ có một máy chủ và laptop, và đo trên mạng thật thì khó tái lập.
- Kết luận: ngoài phạm vi. Docker trên một máy là mức GVHD gợi ý.

### Chỉ dùng tiến trình cục bộ, không dùng Docker
- Ưu: ít hạ tầng nhất.
- Nhược: không đáp ứng gợi ý "tạo dockers"; mỗi node không có môi trường cô lập riêng; không có số đo mạng độc lập.
- Kết luận: dùng cả hai cách. Tiến trình cục bộ cho test; Docker cho demo và E8.

## Consequences
- **Code mới, phải xong trước E0a (K7)** để lưới, E8 và demo cùng một `code_version`:
  - `sim/torchdist.py` (Transport đa tiến trình);
  - `sim/process_runtime.py` (vòng lặp của một node và của observer);
  - lựa chọn runtime trong `sim/run.py`;
  - thư mục `docker/` (Dockerfile và file Compose sinh theo N).
- **Schema đổi trước Tier 1:**
  - TimingRecord thêm `comm_s`;
  - cấu hình thêm `runtime: inprocess | processes` và `deploy: local | docker`.
  - Theo SPEC §8, sửa trước khi có kết quả Tier 1 thì không cần hỏi lại; ADR này ghi lại thay đổi.
- **Test:** T-INV-13 là test tích hợp với tiến trình cục bộ, N = 4, và chạy trong `make check-task`. Lượt Docker là lệnh kiểm tay `make check-docker` có ghi kết quả, không phải test pytest bị skip khi thiếu Docker (CONSTRAINTS F2).
- **Hạn chế ghi vào báo cáo:**
  - Thời gian giao tiếp đo trên một máy (loopback, hoặc bridge network của Docker), không phải mạng thật.
  - Byte vẫn là thước đo chính của chi phí giao tiếp.
  - Mỗi node tự tính cách chia dữ liệu từ seed chung và tập train chung; đây là tiện lợi của mô phỏng, không phải giả định về dữ liệu thật.
- **SPEC, KH và Makefile cần sửa sau khi nhóm duyệt (Ask first):**
  - SPEC: ASSUMPTIONS 1; §1.4 (đa tiến trình thành bắt buộc; Docker; nhiều máy vẫn ngoài phạm vi); REQ-15 (process runtime) và REQ-16 (chi phí giao tiếp theo độ phức tạp model, ADR-011); T-INV-13 ở §9.2; lệnh ở §4; cây thư mục ở §5.1; E8 ở §9.5; Q1 ở §11.1.
  - KH: A1, §2.4, §4.9, §7, §11.
  - Makefile: `check-inv` đếm tới 13, thêm `check-docker`.
- **Ngân sách và lịch:** T22a–T22d và T09c chiếm tuần 5–7 (tasks/plan.md §6). E8 chạy song song với Tier 1 ở tuần 8–9 với N = 8, khoảng 24 lượt ngắn. Nếu tuần 8–10 không đủ, cắt Tier 2 theo thứ tự của ADR-007 (Ask first).
