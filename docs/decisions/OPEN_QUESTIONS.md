# Giả định và câu hỏi mở — Đề 2

Chốt ở P1.3 (phỏng vấn theo skill `interview-me`), ngày 2026-09-30. Nguồn câu hỏi: docs/KE_HOACH_DE_TAI_2.md §2.5 (A1–A6) và §11 (Q1–Q7).
"Nhóm" = nhóm trả lời trực tiếp; "Nhóm (mặc định)" = nhóm chấp nhận phương án mặc định đề xuất trong buổi phỏng vấn.

## Câu hỏi và giả định

| ID | Câu hỏi | Trả lời | Người trả lời | Ngày | Ảnh hưởng |
|---|---|---|---|---|---|
| Q1 / A1 | Mô phỏng đơn tiến trình có message-passing có được coi là "hệ thống P2P" không, hay bắt buộc đa tiến trình/đa máy? | **GVHD: cần đa tiến trình.** Nguyên văn: "Cô cần các process chạy song song, mỗi node chạy độc lập. Các em có thể tạo dockers, số lượng ít cũng được. Chạy mấy mô hình đơn giản, sau mình sẽ tăng độ phức tạp lên xem communication cost sẽ thay đổi như thế nào." A1 không còn là giả định đủ. Áp dụng phương án ghi sẵn ở ADR-002 (mức bổ sung): ADR-010 (mỗi node một tiến trình hay container, gloo, trùng từng bit với bộ mô phỏng) và ADR-011 (thang model, E8). Phần chưa rõ hỏi lại ở Q10–Q13 | Nhóm (phương án) · GVHD (trả lời) | 2026-10-01 | §2.4, ADR-002, ADR-010, ADR-011, T22a–T22d, T09c, T27a–c |
| Q2 | Hạn nộp, ngày bảo vệ, hình thức báo cáo (Word/LaTeX, mẫu), có cần slide/demo không? | Thời lượng **12 tuần**. Ngày nộp/bảo vệ chính xác, định dạng và yêu cầu slide/demo: **chờ GVHD** | Nhóm (12 tuần) · GVHD (chờ) | 2026-09-30 | §7 lịch, T24, T25, A6 |
| Q3 | Báo cáo F1 theo từng node (trung bình macro-F1) đã đủ chưa? | Macro-F1 tính cho **từng node trên tập test chung**; báo cáo **trung bình, std và node kém nhất** | Nhóm | 2026-09-30 | §3.6, T09, T15, ADR-007 |
| Q4 / A2 | Nhóm có bao nhiêu người, dùng máy nào? | **3 người**, 12 tuần. Mọi thí nghiệm chạy trên **máy chủ CPU hiện tại** (8 nhân, không GPU); **laptop cá nhân dùng để demo** | Nhóm | 2026-09-30 | §7 phân vai, §3.7 ngân sách, T13, ADR-007 |
| Q5 | CIFAR-10 có bắt buộc không? | **Không bắt buộc.** Làm tiếp nếu đã xong MNIST và vẫn thấy khả thi (là phần mở rộng ưu tiên) | Nhóm | 2026-09-30 | ADR-003, §2.4, T21 |
| Q6 | Có so sánh với FedAvg có server (CFL) không? | **Có**: B3 và E1c, gần như không tốn thêm công | Nhóm (mặc định) | 2026-09-30 | §3.4 B3, E1c, ADR-005 |
| Q7 | GVHD có kỳ vọng dùng Flower không? | **Không bắt buộc.** Không dùng Flower; dùng PyTorch + NetworkX tự viết | Nhóm | 2026-09-30 | ADR-001 |
| Q10 | Cô có cần chạy *toàn bộ* thí nghiệm Non-IID (khoảng 200 lượt, N = 20) bằng tiến trình hay container riêng không, hay chỉ cần hệ thống đa tiến trình chạy thật ở N nhỏ (4–8) cho demo và E8, còn lưới lớn chạy bằng bộ mô phỏng đã chứng minh trùng từng bit? | **Chờ GVHD.** Mặc định (nhóm đề xuất): vế sau (ADR-010) | GVHD (chờ) | 2026-10-01 | ADR-002, ADR-007 (ngân sách), ADR-010 |
| Q11 | "Tăng độ phức tạp" có phải là tăng kích thước model (số tham số) không? Có cần tăng cả số node, hay giới hạn băng thông mạng giả lập không? | **Chờ GVHD.** Mặc định: kích thước model, 4 bậc từ 7,850 tới 1,663,370 tham số (ADR-011) | GVHD (chờ) | 2026-10-01 | ADR-011, E8 |
| Q12 | Docker là bắt buộc hay tùy chọn? Có cần chạy trên nhiều máy thật không? | **Chờ GVHD.** Mặc định: Docker Compose trên một máy (4–8 container) cho demo và E8; tiến trình cục bộ cho test; không chạy nhiều máy | GVHD (chờ) | 2026-10-01 | ADR-010, T22c, T26 |
| Q13 | "Communication cost" cô muốn đo bằng số byte (chính xác, tất định) hay cả thời gian truyền thực tế? | **Chờ GVHD.** Mặc định: cả hai; byte là thước đo chính, thời gian đo trên process runtime và ghi rõ là đo trên một máy | GVHD (chờ) | 2026-10-01 | ADR-010 (`comm_s`), ADR-011, REQ-05 |
| A3 | Mọi node cùng kiến trúc và cùng khởi tạo $w^0$? | **Có**, $w^0$ chung theo seed chung. Ghi là hạn chế: Badie-Modiri coi đây là "coordinated initialisation" (docs/research/DIGEST.md §2.8). Ablation khởi tạo độc lập để ở phần mở rộng | Nhóm (mặc định) | 2026-09-30 | ADR-006, §2.5, T24 (mục hạn chế) |
| A4 | Mọi node tham gia mọi round, đồng bộ? | **Có**; không mô phỏng rớt node | Nhóm (mặc định) | 2026-09-30 | ADR-002 |
| A5 | Đánh giá trên tập test toàn cục chung? | **Có**, kèm acc^loc theo phân bố nhãn cục bộ của từng node | Nhóm (mặc định) | 2026-09-30 | ADR-007, §3.6 |
| A6 | Ngôn ngữ báo cáo và cách tạo hình? | Báo cáo **tiếng Việt**; mọi hình **sinh tự động từ log**. Định dạng file theo câu trả lời Q2 của GVHD | Nhóm (mặc định) | 2026-09-30 | T16, T24 |

## Quyết định suy ra (đã chốt)

| ID | Nội dung | Trả lời | Người trả lời | Ngày | Ảnh hưởng |
|---|---|---|---|---|---|
| QĐ-1 | Phân vai cho 3 người | TV1 — Dữ liệu + phân tích (T03–T05; phân tích E1/E1c/E4). TV2 — Mạng + Engine (T06–T11, T13, T22). TV3 — Thực nghiệm + báo cáo (T12, T14–T20, T24) | Nhóm (mặc định) | 2026-09-30 | §7 |
| QĐ-2 | Ngân sách tính toán khi chỉ có CPU | Giữ 5 seed cho thí nghiệm lõi và chạy cả Tier 2 trên máy chủ CPU (ước tính 44–57 giờ-CPU, chạy đêm ở tuần 8–10). T13 đo tốc độ thật; nếu chậm hơn ước tính thì cắt Tier 2 theo thứ tự ưu tiên ở §3.7 | Nhóm (mặc định) | 2026-09-30 | §3.7, T13, ADR-007 |
| QĐ-3 | Demo trên laptop | Thêm profile `demo` chạy xong trong vài phút trên laptop (ví dụ ring vs fully-connected ở α = 0.1), cộng bộ hình sinh sẵn từ kết quả thật | Nhóm (mặc định) | 2026-09-30 | §3.5 profile, T11, T16 |
| QĐ-4 | Thứ tự phần mở rộng sau M4 | Ưu tiên CIFAR-10 nếu T13 cho thấy chạy nổi trên CPU; Fashion-MNIST (E6) giữ làm phương án kiểm chứng rẻ | Nhóm (mặc định) | 2026-09-30 | ADR-003, T21 |
| QĐ-5 | Ngoài phạm vi | DP / mã hóa / Byzantine; nén model; nhiều máy thật (đa tiến trình và Docker trên một máy thì nay thuộc phạm vi, theo trả lời Q1 ngày 2026-10-01); web/dashboard; model khác nhau giữa các node; GPU/Colab | Nhóm (mặc định; cập nhật 2026-10-01) | 2026-09-30 | §2.4, ADR-010 |

## Tin nhắn gửi GVHD (đã chốt nội dung, nhóm gửi)

> Kính gửi Thầy/Cô,
>
> Nhóm Đề 2 (ảnh hưởng của dữ liệu Non-IID đến Decentralized Federated Learning, 3 thành viên) xin phép hỏi Thầy/Cô hai điểm để chốt kế hoạch:
>
> 1. Nhóm dự định mô phỏng DFL trong một tiến trình: các node chỉ trao đổi model với láng giềng qua một lớp truyền thông điệp có đếm dung lượng, không có server. Cách này có được chấp nhận là "hệ thống peer-to-peer" không, hay nhóm cần chạy nhiều tiến trình/nhiều máy (ví dụ torch.distributed)? Nếu cần, nhóm sẽ bổ sung một bản chạy 4–8 tiến trình để minh họa.
> 2. Thầy/Cô cho nhóm biết ngày nộp báo cáo và ngày bảo vệ, hình thức báo cáo (Word hay LaTeX, có mẫu không) và có yêu cầu slide/demo không ạ.
>
> Để Thầy/Cô nắm: nhóm làm chính trên MNIST (CIFAR-10 làm thêm nếu kịp), dùng PyTorch + NetworkX tự viết thay cho Flower vì cần kiểm soát topology P2P. Nếu Thầy/Cô có góp ý, nhóm sẽ điều chỉnh.
>
> Em cảm ơn Thầy/Cô.
> Nhóm Đề 2 — [tên thành viên]

Khi có trả lời của GVHD: điền vào hàng Q1 và Q2 (cột Trả lời, Người trả lời, Ngày) rồi commit. Q1 đã điền ngày 2026-10-01.

## Tin nhắn hỏi lại GVHD sau trả lời Q1 (nhóm soạn ngày 2026-10-01, chưa gửi)

> Em chào Cô,
>
> Nhóm Đề 2 đã nhận yêu cầu của Cô về tiến trình chạy song song. Nhóm đề xuất làm như sau và xin Cô xác nhận giúp ạ:
>
> 1. Mỗi node là một tiến trình riêng (chạy trên một máy, hoặc mỗi node một container Docker, 4–8 container), trao đổi model qua mạng bằng torch.distributed, không có server. Nhóm sẽ kiểm tra hệ thống đa tiến trình này cho kết quả trùng khớp hoàn toàn với bản mô phỏng một tiến trình. Thí nghiệm lớn về Non-IID (khoảng 200 lượt, 20 node) nhóm xin chạy bằng bản mô phỏng đã kiểm tra đó, vì máy chủ chỉ có 8 nhân dùng chung; hệ thống đa tiến trình dùng cho demo và thí nghiệm chi phí giao tiếp. Cô có cần nhóm chạy toàn bộ thí nghiệm bằng tiến trình riêng không ạ?
> 2. "Tăng độ phức tạp": nhóm hiểu là tăng kích thước model, từ hồi quy logistic (khoảng 7.9 nghìn tham số) tới CNN của McMahan et al. 2017 (khoảng 1.66 triệu tham số), rồi so sánh chi phí giao tiếp theo topology (ring, gossip, fully-connected). Cô có muốn tăng thêm số node hay giới hạn băng thông mạng không ạ?
> 3. Nhóm đo chi phí giao tiếp bằng số byte trao đổi (chính xác) và thời gian truyền thực tế trên các container. Như vậy đã đúng ý Cô chưa ạ?
>
> Em cảm ơn Cô.
> Nhóm Đề 2

## Cần cập nhật ở pha 1B (P1.5–P1.6)

- Kế hoạch §7 và SPEC: phân vai 3 người (QĐ-1). — SPEC: xong (P1.5, SPEC §3.1). Kế hoạch §7: chờ cập nhật sau khi SPEC được duyệt (SPEC §10).
- §3.5 và SPEC: thêm profile `demo` (QĐ-3). — SPEC: xong (P1.5, SPEC §5.2). Kế hoạch §3.5: chờ cập nhật sau khi SPEC được duyệt.
- ADR-002: ghi phương án dự phòng T22 theo Q1. ADR-003: ghi thứ tự mở rộng QĐ-4. ADR-006: ghi hạn chế A3. ADR-007: macro-F1 theo Q3 và ngân sách QĐ-2. — Xong (P1.6, `docs/decisions/ADR-00N-*.md`, trạng thái Proposed).
