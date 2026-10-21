# ADR-013: Ghi chi phí giao tiếp theo từng node và vẽ theo loại chi phí (byte, message, thời gian)

## Status
Proposed — soạn ngày 2026-10-21 theo yêu cầu của người dùng (nhánh `binh`, phương án B: "làm luôn"). Bổ sung ADR-008 (định dạng kết quả) và ADR-010 (đo `comm_s`), không thay thế chúng. Có hiệu lực cho lưới khi nhóm quyết đưa code này vào nhánh `grid` (xem Consequences).

## Date
2026-10-21

## Context
- REQ-05 đo chi phí giao tiếp bằng `messages_cum`, `bytes_cum` và `max_node_bytes_cum` (tải lớn nhất qua node) trong `rounds.jsonl`, cùng `comm_s` (lớn nhất qua node) trong `timing.jsonl`. Lúc chạy, cả hai runtime đã có số theo từng node:
  - `CommLedger` giữ tải byte của từng id, nhưng không đếm message theo node;
  - ở process runtime, mỗi node gửi tải và thời gian của mình lên observer, rồi observer chỉ giữ tổng hay giá trị lớn nhất.
  Vì vậy không xem lại được tải phân bố giữa các node ra sao: node bậc cao của ER, node của gossip, hay server ảo của B3.
- Ba loại chi phí cho thông tin khác nhau:
  - message cho thấy ảnh hưởng của bậc node và topology;
  - byte bằng message × `nbytes`, với `nbytes` không đổi trong một lượt, nên byte chỉ khác message khi so các model khác cỡ (E8) hay so với đường server;
  - thời gian là số đo thật nhưng không tất định, và chỉ khác 0 ở process runtime (Q13).
- Review C3 P4.1 S5: ở process runtime, `combine_s` chứa cả `comm_s`. Cột chồng các pha theo thời gian vì thế tính giao tiếp hai lần.
- `analysis/` chưa đọc `timing.jsonl`, và chưa có hình nào về chi phí giao tiếp: P6, P9 (T16c) và P10, P11 (T27c) chưa làm.

## Decision
- **`CommCounters`** có thêm hai dãy theo id: `node_messages_cum` và `node_bytes_cum`. Tải của một node bằng phần gửi cộng phần nhận, cùng định nghĩa với `max_node_bytes_cum`, nên `max(node_bytes_cum) == max_node_bytes_cum` và `sum(node_messages_cum) == 2 × messages_cum`.
  - `CommLedger` đếm thêm message theo node.
  - `TorchDistTransport` đếm thêm message nhận. Dãy của nó chỉ có phần tử của chính node khác 0.
  - Observer của process runtime ghép phần tử của từng node thành dãy của cả lượt.
- **`rounds.jsonl`** (tất định) có thêm `node_messages_cum` và `node_bytes_cum`. Mỗi dãy có N phần tử theo id node; ở đường server có thêm phần tử thứ N cho server ảo, như `InMemoryTransport.node_bytes()`. T-INV-13 phủ cả hai trường này: process runtime phải ghi đúng từng byte như bản in-process.
- **`timing.jsonl`** (không tất định) có thêm `train_s_per_node` và `comm_s_per_node`, mỗi dãy N phần tử, cộng từ lần ghi trước.
  - Ở runtime inprocess, `train_s_per_node` đo quanh lời gọi trainer của từng node, và `comm_s_per_node` bằng 0.
  - `train_s` và `comm_s` giữ ý nghĩa cũ: ở inprocess là tổng qua node, ở process runtime là lớn nhất qua node.
- **Sửa S5:** ở process runtime, `combine_s` không còn chứa thời gian chờ gửi và nhận. Thời gian chờ ở barrier của mặt phẳng đo vẫn không vào pha nào; trên hình, nó là phần "khác" = Δ`wall_time_s` trừ tổng các pha.
- **Hai hình mới** ở REQ-10 (T29c), lấy lượt DFL và đường server có τ = 20, μ = 0, η chung, từ E1, E2 và E0b:
  - **P12 · chi phí theo loại, mức lượt:** ba ô có chung trục bước cục bộ, mỗi đường một topology.
    - (a) message tích lũy;
    - (b) byte tích lũy, trục log;
    - (c) thời gian mỗi round, chồng theo pha: train, comm, combine, eval, khác.
  - **P13 · chi phí theo node:** ba ô, mỗi topology một hộp, phân bố qua node và seed ở mốc cuối.
    - (a) tải byte;
    - (b) tải message;
    - (c) thời gian huấn luyện và giao tiếp mỗi round.
    - Server ảo vẽ riêng thành một điểm có nhãn.
  - Mỗi hình chỉ một runtime. Ô thời gian ghi rõ khi `comm_s` bằng 0 vì runtime là inprocess.

## Alternatives Considered

### Chỉ sửa analysis, không đổi schema (phương án A)
- Ưu: không đổi `code_version`, nên vẽ được ngay từ kết quả E0a và E0b.
- Nhược: không có số theo node. Riêng byte của topology tĩnh tính lại được từ đồ thị (2·deg(i)·`nbytes` mỗi round), nhưng gossip và thời gian thì không.
- Kết luận: người dùng chọn B.

### Ghi riêng phần gửi và phần nhận của từng node
- Ưu: thấy được server ảo nhận N và gửi N message mỗi round.
- Nhược: gấp đôi số dãy trên mỗi dòng. Ở mọi topology đồng bộ của dự án, phần gửi bằng phần nhận, trừ đường server.
- Kết luận: không chọn. Ghi tải (gửi + nhận) để cùng định nghĩa với `max_node_bytes_cum`.

### Ghi dãy theo node vào file riêng, `comm.jsonl`
- Ưu: không đổi schema của `rounds.jsonl`.
- Nhược: thêm một file nữa vào ADR-008, RunLogger và T-INV-13, trong khi số đếm vốn đã nằm ở `rounds.jsonl`.
- Kết luận: không chọn.

## Consequences
- **`code_version` đổi**: lượt cũ thành stale theo ADR-008, và hình không được trộn chúng với lượt mới (INT-2).
  - Nhánh `grid` (E0a ở `00fe7f19…`) không nhận commit này. Lưới chỉ có số theo node nếu nhóm đưa code này vào `grid` trước E0b, như một `code_version` mới của Tier 1, rồi ghi vào `docs/experiments/CODE_VERSIONS.md`.
  - Nếu nhóm quyết như vậy thì K7 ("lưới, E8 và demo cùng một `code_version`") vẫn giữ được, vì E8 chưa chạy. E0a chỉ dùng để chọn η, không vào hình nào cùng lượt Tier 1. Nhóm quyết; cho tới lúc đó, P12 và P13 chỉ vẽ được ở `--draft`.
  - Các trường cũ của `rounds.jsonl` không đổi giá trị: code mới chỉ thêm trường, không đổi huấn luyện.
- **Schema** (SPEC §5.3, `contracts.py`) thêm 4 trường. `tests/unit/test_contracts_smoke.py` được sửa trước.
- **Analysis** đọc thêm `timing.jsonl` (không bắt buộc với lượt cũ). P12 và P13 chỉ vẽ lượt có các trường mới; gặp lượt thiếu thì báo lỗi rõ, không vẽ dở.
- **SPEC cần sửa:** §5.3 (`rounds.jsonl`, `timing.jsonl`), REQ-05 (tải theo node), REQ-10 (P12, P13).
- **Tasks:** T29a (ADR-013 và SPEC), T29b (schema và hai runtime), T29c (analysis, P12 và P13).
