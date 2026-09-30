# ADR-007: Giao thức đánh giá — test toàn cục + acc^loc + acc^eff, đo trước và sau tổng hợp, lr chọn trên validation theo trường hợp xấu nhất, 5 seed lõi / 3 seed ablation, thiết kế ghép cặp

## Status
Proposed — chờ Gate G1. Giá trị η (và μ theo SPEC QS-2) được điền vào mục Decision sau E0a.

## Date
2026-09-30

## Context
- Đề yêu cầu đo accuracy/F1, tốc độ hội tụ, độ ổn định và phương sai giữa node (REQ-03, REQ-04, REQ-06, REQ-09), và có baseline IID với các tham số cố định (REQ-12).
- Độ đo suy biến: trên FC với MH, sau tổng hợp mọi node trùng model, nên std giữa node bằng 0 theo cấu trúc (G15).
- Chọn siêu tham số trên tập test là rò rỉ test (G8, G25).
- Đánh giá 20 model ở mỗi round tốn CPU (G12); lưới khoảng 263 lượt phải chạy trên một máy chủ CPU 8 nhân.
- Ít seed: với n ≤ 5, kiểm định dấu hay Wilcoxon hai phía không thể đạt p < 0.05 (KH §3.8).
- Q3: macro-F1 tính cho từng node trên tập test chung; báo cáo trung bình, std và node kém nhất. QĐ-2: giữ 5 seed cho thí nghiệm lõi và chạy cả Tier 2 trên máy chủ CPU (ước tính 44–57 giờ-CPU, chạy đêm trong tuần 8–10); T13 đo tốc độ thật, nếu chậm hơn ước tính thì cắt Tier 2 theo thứ tự ưu tiên.
- Consensus distance là tham số đặc trưng của học phi tập trung (Kong et al. 2021), nhưng mọi thực nghiệm của bài đó dùng dữ liệu IID; áp dụng cho Non-IID là phần mở rộng của dự án (DIGEST §2.6).
- Nguồn: KH §3.5–§3.8, §10; SPEC §5.3, §9; OQ Q3, QĐ-2.

## Decision
- **Tập đánh giá:** tập test toàn cục chung của MNIST. Learning curve dùng một tập con test phân tầng cố định 2,000 ảnh (kích thước và seed là tham số cấu hình, nằm trong `run_id`). Đánh giá cuối trên đủ 10,000 ảnh, lưu ở các trường riêng `final_*_full_test`.
- **Độ đo theo node, sau tổng hợp:** $\mathrm{Acc}_i$; $\mathrm{Acc}^{loc}_i$ (recall theo lớp, lấy trọng số theo phân bố nhãn huấn luyện của node); macro-F1 của từng node, báo cáo mean, std và node kém nhất (Q3); $\mathrm{Acc}^{eff}$ và acc($\bar w$) làm chẩn đoán.
- **Trước tổng hợp:** drift $\Xi^{\frac12}$. **Sau tổng hợp:** consensus distance Ξ.
- **Độ ổn định:** độ dao động và mức sụt tính trên chuỗi accuracy của từng node rồi mới lấy trung bình (G24); độ lệch giữa seed tính trên accuracy cuối (10,000 ảnh).
- **Lịch đánh giá theo bước cục bộ:** các mốc $s_k$ = 20, 40, …, 1,000, rồi 1,100, 1,200, …, 4,000 (khoảng 80 mốc); đánh giá ở round đầu tiên $r$ có $r\tau\ge s_k$.
- **Chọn lr (E0a, 1 seed, chỉ trên validation 5k):**
  - Quét η ∈ {0.01, 0.05, 0.1} với τ = 20 trên 3 ô {FC-IID, ring-α0.1, ring-#C2}.
  - Chọn η có accuracy validation cuối nhỏ nhất qua 3 ô là lớn nhất (tốt nhất trong trường hợp xấu nhất). Nếu nhiều η cách giá trị tốt nhất không quá 0.2 điểm %, chọn η có AUC validation lớn hơn ở ô ring-α0.1.
  - η đã chọn được dùng chung cho mọi τ. Quét thêm η cho τ = 1 trên {FC-IID, ring-α0.1}.
  - E0b kiểm độ nhạy với η/2 và 2η.
- **η đã chọn:** _(điền sau E0a, trước khi chạy E0b)_. **μ của DFedProx:** _(điền cùng lúc với η, theo ADR-006 và SPEC QS-2)_. Điền hai giá trị này là bước đã đăng ký trước (KH §3.5), không phải thay đổi quyết định; ghi kèm ngày và run ID của E0a.
- **Seed:** 5 seed {0…4} cho E0b-DFedAvg, E1, E2; 3 seed {0, 1, 2} cho baseline và ablation (B1, B2, E1c, E2b, E3, E4).
- **Thiết kế ghép cặp:** cùng seed thì cùng partition, cùng hoán vị vị trí node và cùng $w^0$ cho mọi topology và thuật toán (G9). Seed con sinh từ `SeedSequence` theo tên luồng.
- **Thống kê:** mean ± 95% CI (phân phối t) kèm điểm của từng seed; ưu tiên độ lớn hiệu ứng và hiệu số ghép cặp. H2 được kiểm bằng độ dốc theo từng seed trên thang Dirichlet kèm IID. H2 được ủng hộ khi cả 5 độ dốc đều dương (kiểm định dấu một phía, p = 1/32) và CI 95% (t, df = 4) của trung bình các độ dốc nằm trên 0.
- **Ngân sách (QĐ-2):** chỉ CPU. Tier 1 khoảng 29 giờ-CPU; Tier 1 + 2 khoảng 44 giờ-CPU; cộng 30% dự phòng thì khoảng 57 giờ-CPU. T13 đo lại. Nếu chậm hơn, cắt từ cuối thứ tự ưu tiên E0a → E0b → E1 → E2 → E3a → E3b → E1c → E4 → E2b, tức bỏ E2b trước, rồi E4, E1c, E3b. Việc cắt phải được nhóm duyệt (SPEC §8, Ask first).

## Alternatives Considered

### Chọn siêu tham số trên tập test
- Ưu: không cần tách validation.
- Nhược: rò rỉ test, cho kết quả lạc quan giả.
- Kết luận: không chọn (G8, G25).

### Chỉ đo sau tổng hợp
- Ưu: đơn giản, đỡ tốn CPU.
- Nhược: trên FC mọi node trùng nhau sau tổng hợp, nên "phương sai giữa node" luôn bằng 0 và dễ bị diễn giải sai.
- Kết luận: không chọn; đo cả drift trước tổng hợp và acc^loc (T-INV-11).

### Tinh chỉnh lr riêng cho từng τ hoặc từng ô
- Ưu: mỗi cấu hình ở trạng thái tốt nhất của nó.
- Nhược: lẫn hiệu ứng của lr vào kết luận về tần suất giao tiếp (H7); tốn ngân sách.
- Kết luận: không chọn. Chỉ chạy bổ sung D-PSGD với η riêng khi điều kiện ở KH §3.5 xảy ra.

### 3 seed cho thí nghiệm lõi
- Ưu: rẻ hơn khoảng 40% cho E1 và E2.
- Nhược: với 3 seed, nửa độ rộng CI bằng 2.48 lần std; không đủ cho kiểm định H2 theo độ dốc (cần 5 seed để đạt p = 1/32).
- Kết luận: không chọn. Chỉ được hạ khi thiếu tài nguyên nghiêm trọng, phải được nhóm duyệt và ghi rõ trong báo cáo.

### Đánh giá mọi round trên đủ 10,000 ảnh
- Ưu: sai số đo nhỏ hơn.
- Nhược: quá tốn CPU với 20 model và hàng trăm tới hàng nghìn round.
- Kết luận: không chọn. Tập con 2,000 ảnh dùng cho learning curve, 10,000 ảnh cho đánh giá cuối.

## Consequences
- So sánh công bằng, không rò rỉ test, tránh độ đo suy biến trên FC.
- $\mathrm{Acc}^{loc}$ thưởng cho thiên lệch cục bộ (ở #C = 1, một model chỉ đoán lớp của chính node đạt 100%), nên luôn báo cáo kèm $\mathrm{Acc}_i$ và $\mathrm{Acc}^{loc}$ của B2, và không dùng riêng nó để xếp hạng topology.
- Sai số chuẩn của tập con 2,000 ảnh khoảng 0.4–0.5 điểm % ở mức accuracy 95–97%, nên R@97 chỉ là độ đo phụ.
- Schema kết quả có thêm `f1_macro_std`, `f1_macro_min` và macro-F1 của từng node ở đánh giá cuối (Q3; SPEC §5.3; ADR-008).
- Ngưỡng sanity SAN-1…4 (SPEC §9.3; nhóm chốt các số ở P1.8, ghi ở `CONSTRAINTS.md` R7) phải đạt trước khi chạy E1 và E2.
