# ADR-007: Giao thức đánh giá — test toàn cục + acc^loc + acc^eff, đo trước và sau tổng hợp, lưới mốc chung cho mọi τ, lr chọn trên validation theo trường hợp xấu nhất, 5 seed lõi / 3 seed ablation, thiết kế ghép cặp

## Status
Proposed — chờ Gate G1. Giá trị η (và μ theo SPEC QS-2) được điền vào mục Decision sau E0a.

## Date
2026-09-30 (cập nhật theo hai vòng phản biện P1.9 cùng ngày)

## Context
- Đề yêu cầu đo accuracy/F1, tốc độ hội tụ, độ ổn định và phương sai giữa node (REQ-03, REQ-04, REQ-06, REQ-09), và có baseline IID với các tham số cố định (REQ-12).
- Độ đo suy biến: trên FC với MH, sau tổng hợp mọi node trùng model, nên std giữa node bằng 0 theo cấu trúc (G15).
- Chọn siêu tham số trên tập test là rò rỉ test (G8, G25). Điều này gồm cả việc đổi siêu tham số *sau khi* thấy kết quả trên test, dù có ghi lại bằng ADR.
- Đo ngày 2026-09-30 trên máy chủ (1 luồng): một bước huấn luyện SmallCNN (B = 32) mất 10.5 ms; đánh giá một model trên 2,000 ảnh mất 0.40 s. Với lịch khoảng 80 mốc của KH §3.5, đánh giá chiếm gần một nửa thời gian của một lượt (G12). Máy chủ dùng chung, lúc đo tải trung bình khoảng 7 trên 8 nhân.
- Lịch của KH §3.5 đặt mốc theo bước rồi đánh giá ở round đầu tiên vượt mốc. τ = 50 vì thế chỉ có 50 mốc, bắt đầu từ bước 50, trong khi τ ≤ 20 có 80 mốc, bắt đầu từ bước 20. Nội suy tuyến tính trên lưới thưa hơn làm R@x của τ lớn bị trễ, còn AUC và độ dao động phụ thuộc số mốc (REVIEW-P1.9, vòng 2, F8).
- Ít seed: với n ≤ 5, kiểm định dấu hay Wilcoxon hai phía không thể đạt p < 0.05 (KH §3.8).
- Q3: macro-F1 tính cho từng node trên tập test chung; báo cáo trung bình, std và node kém nhất. QĐ-2: giữ 5 seed cho thí nghiệm lõi và chạy cả Tier 2 trên máy chủ CPU; T13 đo tốc độ thật, nếu chậm thì cắt Tier 2 theo thứ tự ưu tiên. QS-3: ngưỡng sanity SAN-1…4 (chốt ở P1.8); đối chứng của SAN-2 là B1-4k (nhóm chốt ở P1.9, 2026-09-30).
- Consensus distance là tham số đặc trưng của học phi tập trung (Kong et al. 2021), nhưng mọi thực nghiệm của bài đó dùng dữ liệu IID; áp dụng cho Non-IID là phần mở rộng của dự án (DIGEST §2.6).
- Nguồn: KH §3.5–§3.8, §5.3, §10; SPEC §5.2, §5.3, §9; OQ Q3, QĐ-2; `docs/decisions/REVIEW-P1.9.md`.

## Decision
- **Tập đánh giá** là tham số cấu hình (split, kích thước tập con cho learning curve, seed của tập con), nằm trong `run_id`.
  - Mọi nhóm thí nghiệm dùng tập test toàn cục của MNIST: learning curve trên một tập con test phân tầng cố định 2,000 ảnh; đánh giá cuối trên đủ 10,000 ảnh.
  - E0a dùng tập validation 5k cho cả hai: learning curve trên tập con phân tầng 2,000 ảnh của validation, đánh giá cuối trên đủ 5,000 ảnh. E0a không đụng tập test.
  - Cách tách validation và tập con đánh giá lấy từ luồng dữ liệu cố định (SPEC §6), không từ seed của lượt, nên mọi lượt dùng cùng ảnh.
  - `summary.json` ghi `final_eval_split` và `final_eval_size`.
- **Độ đo theo node, sau tổng hợp:** $\mathrm{Acc}_i$; $\mathrm{Acc}^{loc}_i$ (recall theo lớp, lấy trọng số theo phân bố nhãn huấn luyện của node); macro-F1 của từng node trên đủ 10 lớp, dạng $2TP_c/(2TP_c+FP_c+FN_c)$ nên không có 0/0 khi một lớp không bao giờ được dự đoán; báo cáo mean, std và node kém nhất (Q3); $\mathrm{Acc}^{eff}$ và acc($\bar w$) làm chẩn đoán.
- **Trước và sau tổng hợp:** drift $\Xi_{t+\frac12}$ trên các $w^{t+\frac12}$, và consensus distance $\Xi_t$ trên các $w^t$. Cả hai là khoảng cách RMS $\sqrt{\frac1N\sum_i\lVert w_i-\bar w\rVert^2}$; chỉ số $t+\frac12$ chỉ nửa bước, không phải căn bậc hai.
- **Độ ổn định:** độ dao động (std của hiệu số liên tiếp trên 20 mốc cuối) và mức sụt tính trên chuỗi accuracy của từng node rồi mới lấy trung bình (G24); độ lệch giữa seed tính trên accuracy cuối.
- **Lịch đánh giá — lưới mốc chung:** đánh giá ở các mốc 100, 200, …, 4,000 bước cục bộ (40 mốc), giống nhau cho mọi τ ∈ {1, 5, 20, 50}. 100 chia hết cho mọi τ đó, nên mọi mốc rơi đúng vào cuối một round. Round cuối luôn được đánh giá. Lịch là tham số của profile; profile smoke đánh giá mỗi round. Vì mọi lượt có cùng lưới, R@x, AUC, độ dao động và mức sụt được tính trên cùng các điểm cho mọi τ. Điều này không làm các độ đo hoàn toàn độc lập với τ: accuracy chỉ đo sau tổng hợp (ADR-006, bước 4).
- **R@x** (định nghĩa đầy đủ ở SPEC REQ-04): tính trên chuỗi `acc_mean`. Lấy lần vượt ngưỡng x đầu tiên mà mọi mốc trong 200 bước kế tiếp vẫn ≥ x − 0.5 điểm %, và sau nó còn ít nhất một mốc. Nội suy tuyến tính giữa mốc trước và mốc vượt. Nếu vượt ngay ở mốc 100 thì ghi "≤ 100"; không có lần vượt bền vững thì ghi "> ngân sách". AUC là trung bình `acc_mean` theo bước trên cùng lưới.
- **R@x trong kiểm định:** R@x và bytes@x bị kiểm duyệt hai phía ("≤ 100", "> ngân sách"), nên không dùng CI t cho chúng. Giả thuyết về tốc độ dùng kiểm định dấu trên thứ tự ghép cặp của từng seed. "> ngân sách" lớn hơn mọi giá trị hữu hạn, và "≤ 100" nhỏ hơn mọi giá trị lớn hơn 100. Hai giá trị cùng bị kiểm duyệt trong một cặp là hòa, và hòa tính là không đúng chiều. "Ủng hộ" cần đúng chiều ở cả 5 seed (p = 1/32), báo cáo kèm trung vị và "k/n seed đạt".
- **B1 (tập trung):** 10 epoch trên 55k mẫu, B = 32 (khoảng 17,190 bước), minibatch lấy có hoàn lại. η riêng của B1 chọn trên validation trong {0.01, 0.05, 0.1} (3 lượt, seed 100, thuộc E0a), rồi chạy 3 seed ở E0b.
- **B1-4k:** như B1 nhưng dùng η chung của lưới và 4,000 bước, bằng số bước cục bộ của một node DFL (3 seed, thuộc E0b). B1-4k là đối chứng của SAN-2 (nhóm chốt ở P1.9, QS-3). B1 mạnh hơn DFedAvg-FC-IID ở cả η lẫn ngân sách, nên khoảng cách tới B1 không phân biệt được lỗi với hiệu ứng ngân sách. B1-4k cùng η, cùng B và cùng số bước, nên DFedAvg-FC-IID kém nó hơn 1.0 điểm % mới là dấu hiệu lỗi. Khoảng cách tới B1 vẫn được báo cáo.
- **B3 (FedAvg có server):** chạy qua đường server của engine (ADR-005), trên IID ở E0b (3 seed) và trên thang Dirichlet ở E1c.
- **Chọn lr (E0a, chỉ trên validation, seed 100):**
  - Seed 100 không thuộc {0…4}, để η không được chọn trên chính partition và $w^0$ của một seed đánh giá.
  - Quét η ∈ {0.01, 0.05, 0.1} với τ = 20 trên 3 ô {FC-IID, ring-α0.1, ring-#C2} (9 lượt).
  - "Accuracy validation" của một lượt là trung bình `acc_mean` ở 5 mốc cuối (3,600 … 4,000 bước) trên tập con validation 2,000 ảnh. Lấy trung bình để một điểm cuối dao động (ring-#C2) không quyết định η. Mọi η được đánh giá trên cùng ảnh, nên phép so giữa các η là ghép cặp. "AUC validation" là AUC của đường `acc_mean` trên tập con validation.
  - Chọn η có accuracy validation nhỏ nhất qua 3 ô là lớn nhất (tốt nhất trong trường hợp xấu nhất). Nếu nhiều η cách giá trị tốt nhất không quá 0.2 điểm %, chọn η có AUC validation lớn hơn ở ô ring-α0.1. Lượt phân kỳ tính là accuracy 0.
  - η đã chọn được dùng chung cho mọi τ.
  - Quét thêm η cho τ = 1 và τ = 50 trên {FC-IID, ring-α0.1} (6 + 6 lượt), để biết hai đầu của dải τ có bị thiệt vì η chung không. "η tốt nhất" cho τ đó được chọn theo đúng quy tắc trên, lấy trên 2 ô này. Nếu η chung cách giá trị tốt nhất không quá 0.2 điểm % thì coi η chung là tốt nhất.
  - Nếu η tốt nhất của τ = 1 (hoặc τ = 50) khác η chung, chạy thêm 12 lượt với η riêng cho giá trị τ đó, làm kết quả bổ sung ở Tier 2: các ô D-PSGD của E3a, hoặc các ô μ = 0, τ = 50 của E3b.
- **E0b kiểm độ nhạy η/2 và 2η** trên {T1, T4} × {IID, α = 0.1} (8 lượt, seed 0). Phép kiểm này chạy trên tập test nên chỉ để mô tả: báo cáo giá trị của từng cấu hình. Vì chỉ có 1 seed, dấu của hiệu (FC − ring ở α = 0.1) − (FC − ring ở IID) có thể đổi chỉ vì nhiễu, nên nó không được dùng để kết luận H2 bền hay không bền với η. Kết quả này không bao giờ được dùng để chọn lại η, kể cả qua ADR mới (SPEC §8).
- **η đã chọn:** _(điền sau E0a, trước khi chạy E0b)_. **μ của DFedProx:** _(điền cùng lúc với η, theo ADR-006 và SPEC QS-2)_. Điền hai giá trị này là bước đã đăng ký trước (KH §3.5), không phải thay đổi quyết định; ghi kèm ngày và run ID của E0a. Nếu một lỗi có test tái hiện buộc sửa `src/` sau E0a, mọi nhóm chạy lại dưới `code_version` mới, bắt đầu từ E0a. η và μ được chọn lại trên validation theo đúng quy tắc trên; giá trị mới thay giá trị cũ, và mục này ghi cả hai cùng lý do.
- **Seed:** 5 seed {0…4} cho E0b-DFedAvg, E1, E2; 3 seed {0, 1, 2} cho baseline và ablation (B1, B1-4k, B2, B3, E1c, E2b, E3, E4, E4a); seed 100 cho E0a. So sánh ghép cặp chỉ dùng giao các tập seed của những điều kiện được so. Ví dụ quét τ (E3b) dùng seed {0, 1, 2} cho cả ô τ = 20 lấy từ E1, và hình ghi rõ seed đã dùng (SPEC INT-2).
- **Thiết kế ghép cặp:** cùng seed thì cùng partition, cùng hoán vị vị trí node và cùng $w^0$ cho mọi topology và thuật toán (G9). Luồng ngẫu nhiên sinh từ `SeedSequence` với spawn_key cố định (SPEC §6).
- **Thống kê và quy tắc quyết định, đăng ký trước:**
  - Báo cáo mean ± 95% CI (phân phối t) kèm điểm của từng seed; ưu tiên độ lớn hiệu ứng và hiệu số ghép cặp.
  - Mỗi giả thuyết H1–H7 được cụ thể hóa trong thẻ `docs/experiments/E<k>.md` trước khi chạy, bằng đúng một phép so sánh chính: độ đo, các ô, chiều kỳ vọng, tập seed. Mọi phép so sánh khác chỉ để khám phá, được báo cáo nhưng không dùng để kết luận "ủng hộ".
  - Có bảy phép so sánh chính, mỗi phép có p = 1/32 với 5 seed. Nếu mọi H0 đều đúng, xác suất có ít nhất một kết luận "ủng hộ" sai là tối đa khoảng 20% ($1-(31/32)^7$). Với 5 seed, không có hiệu chỉnh bội nào đạt được mức 5% cho cả họ, nên báo cáo nêu con số này trong mục hạn chế, và khuyến nghị chính phải kèm độ lớn hiệu ứng và CI.
  - Với 5 seed ghép cặp: "ủng hộ" khi chiều kỳ vọng đúng ở cả 5 seed (kiểm định dấu một phía, p = 1/32) và CI 95% của trung bình hiệu số không chứa 0; "không ủng hộ" khi CI nằm hẳn về phía ngược lại; còn lại là "chưa kết luận".
  - Với 3 seed (ablation): tối đa là "gợi ý", khi chiều kỳ vọng đúng ở cả 3 seed và CI không chứa 0; không bao giờ là "ủng hộ".
  - **Khuyến nghị (REQ-13):** khuyến nghị chính cần kết luận "ủng hộ". Kết luận "gợi ý" chỉ cho phép khuyến nghị tạm, ghi rõ số seed và "cần thêm seed để khẳng định". Kết luận "chưa kết luận" hay "không ủng hộ" thì không khuyến nghị theo hướng đó. Nếu Tier 2 bị cắt, khuyến nghị về chiến lược tổng hợp vẫn dựa được trên E3a (Tier 1), ở mức khuyến nghị tạm.
  - **H2 (kiểm định chính):** trên thang Dirichlet kèm IID (6 mức), với mỗi seed $s$: $g_s(\ell)=\mathrm{Acc}_{FC}-\mathrm{Acc}_{ring}$, dùng `acc_mean` của đánh giá cuối trên đủ tập test, cùng partition, hoán vị và $w^0$. Biến giải thích là JS đo được của partition đó; IID đi vào với JS đo được của nó (khoảng 0.001). Độ dốc $b_s$ ước lượng bằng hồi quy tuyến tính thường. Dốc dương nghĩa là khoảng cách FC − ring tăng theo mức Non-IID. H2 được ủng hộ khi cả 5 độ dốc đều dương (p = 1/32) và CI 95% (t, df = 4) của trung bình $b_s$ nằm trên 0. Các cặp topology khác chỉ để khám phá, báo cáo nhưng không kết luận. Ô IID của cả hai topology có 5 seed ở E0b; các ô Dirichlet có 5 seed ở E1.
  - **RQ2 (khám phá):** hồi quy mức suy giảm ghép cặp (Acc ở IID trừ Acc ở α = 0.1, cùng seed) theo log spectral gap của từng lượt, trên các lượt topology tĩnh (ring, ER theo từng seed, FC); báo cáo độ dốc và CI, không kết luận. Spectral gap chỉ có ba cụm giá trị (ring 0.033, ER khoảng 0.09–0.32, FC 1) và lẫn với bậc, số message và byte. Vì vậy phần "giải thích được bằng spectral gap không" của RQ2 chỉ trả lời được ở mức mô tả, và báo cáo phải nói rõ điều này. Gossip vẽ riêng theo tốc độ trộn kỳ vọng (P8), không đặt chung trục spectral gap.
  - **Lượt phân kỳ:** tính là không đạt R@x. Ô có seed phân kỳ báo cáo "k/n seed phân kỳ" cùng điểm của từng seed, không lấy trung bình qua NaN. Giả thuyết dựa trên ô đó được ghi "chưa kết luận", trừ khi thẻ thí nghiệm đã quy định trước cách xử lý.
- **Ngân sách (QĐ-2), theo số đo ngày 2026-09-30:**
  - Một lượt DFL đủ ngân sách tốn khoảng 20 phút-lõi: huấn luyện 20 node × 4,000 bước × 10.5 ms ≈ 14 phút; đánh giá 40 mốc × 21 model × 0.40 s ≈ 5.6 phút; đánh giá cuối khoảng 0.7 phút. B1 khoảng 4 phút, B1-4k khoảng 1 phút.
  - Lưới lõi của Tier 1 (197 lượt) khoảng 64 giờ-lõi; E8 (72 lượt ngắn, ADR-011) tính riêng. Tier 1 + 2 (278 lượt) khoảng 92 giờ-lõi, 100 nếu chạy đủ 24 lượt có điều kiện, khoảng 130 kể cả 30% dự phòng.
  - Thời gian thực bằng giờ-lõi chia số lõi rảnh. Với khoảng 1 lõi rảnh (tải khoảng 7 trên 8 nhân): Tier 1 khoảng 2.7 ngày, Tier 1 + 2 khoảng 4–5.5 ngày chạy liên tục. Với 4 lõi rảnh: khoảng 1–1.5 ngày. Vẫn nằm trong 4 tuần dành cho thí nghiệm (tuần 7–10), nhưng dự phòng 30% không đủ cho một lần chạy lại toàn bộ Tier 1 (khoảng 64 giờ-lõi) nếu phải sửa lỗi sau khi Tier 1 xong. Khi đó áp dụng thứ tự cắt ở dưới.
  - Đường găng: E0a → ghi η → E0b → SAN-1…3 → E1 (ô α = 100 trước, để có SAN-4 sớm) → SAN-4 → E2. E3a và E4a chạy song song với E1 sau khi qua SAN-1…3.
  - Lịch đánh giá, tập con đánh giá và số luồng nằm trong `run_id`, nên T13 phải chốt chúng trước E0a, sau khi đo tốc độ dưới tải thật. Sau E0a không đổi chúng nữa; nếu chậm hơn dự kiến thì chỉ cắt Tier 2.
  - Việc cắt đi từ cuối thứ tự ưu tiên E0a → E0b → E1 → E2 → E3a → E4a → E3b → E1c → E4 → E2b, tức bỏ E2b trước, rồi E4, E1c, E3b. Cắt phải được nhóm duyệt (SPEC §8, Ask first) và làm theo SPEC EXP-5.
  - Máy chủ là điểm lỗi duy nhất. Nếu mất máy chủ, nhóm quyết (Ask first) việc chạy tiếp trên máy khác. Mọi lượt của một nhóm thí nghiệm phải chạy trên cùng một máy (G13).

## Alternatives Considered

### Chọn siêu tham số trên tập test
- Ưu: không cần tách validation.
- Nhược: rò rỉ test, cho kết quả lạc quan giả.
- Kết luận: không chọn (G8, G25). Điều này gồm cả việc dùng kết quả E0b (η/2, 2η) để chọn lại η.

### Chỉ đo sau tổng hợp
- Ưu: đơn giản, đỡ tốn CPU.
- Nhược: trên FC mọi node trùng nhau sau tổng hợp, nên "phương sai giữa node" luôn bằng 0 và dễ bị diễn giải sai.
- Kết luận: không chọn; đo cả drift trước tổng hợp và acc^loc (T-INV-11).

### Lịch đánh giá của KH §3.5 (mỗi 20 bước tới 1,000, rồi mỗi 100 bước)
- Ưu: phân giải cao ở đầu learning curve, nơi các ô IID đạt 90% rất nhanh.
- Nhược: τ = 50 không trúng các mốc 20, 40, …, nên số mốc khác nhau theo τ (50 so với 80). R@x, AUC và độ dao động vì thế lệch theo τ, đúng ở phép so RQ3/H7. Chi phí đánh giá cũng gấp đôi.
- Kết luận: không chọn. Lưới 100 bước làm R@90 của các ô IID thô hơn ("≤ 100" hay khoảng 100–200 bước). Các so sánh chính dựa trên ô Non-IID, vốn đạt ngưỡng chậm hơn nhiều, và trên R@95.

### Chỉ dùng lưới chung khi so các τ khác nhau
- Ưu: giữ phân giải cao cho E1 và E2 (cùng τ = 20).
- Nhược: cùng một lượt τ = 20 có hai giá trị R@x, tùy hình; dễ nhầm khi viết báo cáo.
- Kết luận: không chọn.

### Tinh chỉnh lr riêng cho từng τ hoặc từng ô
- Ưu: mỗi cấu hình ở trạng thái tốt nhất của nó.
- Nhược: lẫn hiệu ứng của lr vào kết luận về tần suất giao tiếp (H7); tốn ngân sách.
- Kết luận: không chọn. Chỉ chạy bổ sung với η riêng cho τ = 1 hoặc τ = 50 khi điều kiện ở trên xảy ra.

### Hai seed cho E0a
- Ưu: chọn η ít nhiễu hơn.
- Nhược: gấp đôi số lượt của E0a.
- Kết luận: không chọn. Nhiễu được giảm bằng cách lấy trung bình 5 mốc cuối và so ghép cặp trên cùng ảnh. Kiểm tra η/2, 2η ở E0b chỉ báo cáo độ bền của H2, không bù cho nhiễu khi chọn η.

### SAN-2 so với B1 đầy đủ (KH §5.3)
- Ưu: đúng như KH; đối chứng là cận trên thật của model.
- Nhược: B1 có η riêng và ngân sách khoảng 4 lần, nên SAN-2 có thể fail mà không có lỗi. Khi đó quy tắc "chỉ sửa lỗi có test tái hiện, không đổi siêu tham số" không cho lối ra nào.
- Kết luận: không chọn; nhóm chốt B1-4k làm đối chứng (P1.9, 2026-09-30, QS-3). Khoảng cách tới B1 vẫn được báo cáo, nhưng không phải điều kiện dừng.

### 3 seed cho thí nghiệm lõi
- Ưu: rẻ hơn khoảng 40% cho E1 và E2.
- Nhược: với 3 seed, nửa độ rộng CI bằng 2.48 lần std; không đủ cho kiểm định H2 theo độ dốc (cần 5 seed để đạt p = 1/32).
- Kết luận: không chọn. Chỉ được hạ khi thiếu tài nguyên nghiêm trọng, phải được nhóm duyệt và ghi rõ trong báo cáo.

### Đánh giá mọi round trên đủ 10,000 ảnh
- Ưu: sai số đo nhỏ hơn.
- Nhược: quá tốn CPU.
- Kết luận: không chọn. Tập con 2,000 ảnh dùng cho learning curve, đủ tập dùng cho đánh giá cuối.

## Consequences
- So sánh công bằng giữa các τ, không rò rỉ test, tránh độ đo suy biến trên FC.
- $\mathrm{Acc}^{loc}$ thưởng cho thiên lệch cục bộ (ở #C = 1, một model chỉ đoán lớp của chính node đạt 100%), nên luôn báo cáo kèm $\mathrm{Acc}_i$ và $\mathrm{Acc}^{loc}$ của B2, và không dùng riêng nó để xếp hạng topology.
- Sai số chuẩn của tập con 2,000 ảnh khoảng 0.4–0.5 điểm % ở mức accuracy 95–97%, nên R@97 chỉ là độ đo phụ.
- Schema kết quả có `f1_macro_std`, `f1_macro_min`, macro-F1 từng node ở đánh giá cuối (Q3), và `final_eval_split` (SPEC §5.3, ADR-008).
- Ngưỡng sanity SAN-1…4 (SPEC §9.3; nhóm chốt các số ở P1.8, ghi ở `CONSTRAINTS.md` R7) phải đạt trước khi chạy E1 và E2. SAN-5 kiểm trên `rounds.jsonl` của lượt thật rằng bước tổng hợp có tác dụng, vì T-INV chỉ chạy trên MLP tí hon. Khi một SAN không đạt mà debug không tìm ra lỗi, chỉ nhóm được quyết định chạy tiếp (SPEC §9.3).
- **Trade-off đã chấp nhận:**
  - Ngưỡng hòa 0.2 điểm % khi chọn η vẫn nhỏ hơn sai số của một điểm đo. Trung bình 5 mốc và phép so ghép cặp làm nhiễu nhỏ hơn, nhưng phép chọn giữa hai η gần nhau vẫn có thể lật. Khi đó hai η đều gần tương đương trên validation.
  - Chọn η theo ô khó nhất có thể làm nhỏ hiệu ứng Non-IID đo được; điều này được ghi trong mục hạn chế (SPEC INT-4).
  - Lưới 100 bước làm R@x ở các ô hội tụ nhanh (IID, FC) thô hơn lịch của KH.
  - Quantity skew chỉ có một mức (β = 0.5), 3 seed, 2 topology. Với nhãn IID và MH, lệch mục tiêu gần bằng 0 theo cấu trúc, nên E4a chủ yếu đo ảnh hưởng của kích thước node. RQ4 so một điểm này với thang label skew 7 mức, và báo cáo phải nói rõ. Thêm mức β = 0.1 tốn 6 lượt; chỉ làm nếu nhóm yêu cầu (Ask first).
