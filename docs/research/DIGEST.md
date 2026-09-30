# DIGEST tài liệu nghiên cứu — kiểm chứng §1.3 của kế hoạch (P1.2)

Ngày 2026-09-30. Nguồn: 2 PDF trong `docs/research/` và 8 bản arXiv tải về. Ba subagent Explore đọc song song; phiên chính đối chiếu lại 46 trích dẫn/con số với văn bản gốc (tất cả khớp).
Quy ước: **[p.N] = số trang của file PDF** (không phải số trang tạp chí). "(đọc từ hình)" = giá trị xấp xỉ lấy từ đồ thị. Chỉ ghi điều bài báo nêu; ô "không nêu" là đã tìm mà không thấy.

## 1. Đối chiếu với §1.3

| Tài liệu | §1.3 đang nói | Kết quả kiểm chứng | Đề xuất sửa §1.3 |
|---|---|---|---|
| Zhou 2023 | Bài phương pháp, 1 đồ thị 10 node, 1 mức Non-IID, CFL ~78% | Đúng | Thêm cảnh báo về Fig. 6 (mục 2.1) |
| Kalra 2023 | Baseline Regular/Joint, AvgPush, đồ thị mũ, p_major, 5 seed, GroupNorm | Đúng | Thêm: mọi baseline dùng DP-SGD → không so số tuyệt đối |
| Palmieri 2024 | Topology × phân phối dữ liệu; độ trung tâm; hiệu ứng pha loãng | Đúng | Ghi thêm thiết lập (MNIST, 100 node, BA/ER/SBM) |
| NIID-Bench | Có bước "balance", nhãn đầu i mod 10, β mặc định 0.5 | Đúng **theo mã nguồn**, bài báo không mô tả | Ghi nguồn là `utils.py`, không phải bài báo |
| Koloskova 2020 | Hội tụ phụ thuộc dị biệt dữ liệu và spectral gap | Đúng; công thức MH của bài trùng §3.3 | Không |
| Kong 2021 | Consensus distance là tham số đặc trưng | Đúng, nhưng **mọi thực nghiệm là IID** | Ghi rõ: dùng cho Non-IID là mở rộng |
| Hsieh 2020 | Non-IID làm hỏng học phi tập trung, nặng nhất với BatchNorm | Đúng; nhưng bối cảnh là **parameter server theo phân vùng**, không có topology P2P | Ghi rõ bối cảnh |
| Badie-Modiri | Khởi tạo chung là mặc định, độc lập là ablation | Bài coi khởi tạo chung là **"coordinated"**; tiền đề của bài là khởi tạo không phối hợp | Nêu A3 là một hạn chế; ablation dùng hệ số hiệu chỉnh của bài |
| Hegedűs 2021 | So sánh gossip learning với FL | **Chưa kiểm chứng**: không tải được toàn văn | Đánh dấu "chưa đọc toàn văn" |
| Khảo sát | Thuật ngữ, phân loại topology, độ đo Non-IID | Đúng với Beltrán 2023, Jimenez G. 2024; ACM CSUR 2025 chưa đọc | Không |

## 2. Từng tài liệu

### 2.1 Zhou et al. 2023 — *Mathematics* 11(14):3162 (`mathematics-11-03162.pdf`)
- (a) MNIST, CIFAR-10, FEMNIST (lấy 10%) [p.8]. 10 node, đồ thị cố định: node 1 nối mọi node, node 9 chỉ có 3 láng giềng [p.8]. Mỗi round chọn node trung tâm theo điểm log2(1+d_k)/μ_k [p.6–7]. Non-IID: mỗi node 2 nhãn, mỗi nhãn chia nhóm 500 mẫu, node chọn 3 nhóm/nhãn [p.8]. CNN khác nhau giữa các node (Table 2) [p.9]. Không nêu: lr, batch, số bước cục bộ, số seed.
- (b) MNIST sau 50 round, IID → Non-IID (Table 3 → Table 4): CFL 78.34 → 70.23%; FedProx 78.56 → 70.51%; phương pháp đề xuất 87.59 → 81.18% [p.10–11]. Node ít kết nối kém nhất: "The accuracy of recognition is reduced when there are fewer objects connected" [p.11]; node 9 ≈ 44% trên MNIST (đọc từ Fig. 6a).
- **Thận trọng:** đường node 8 ở Fig. 6b (CIFAR-10) và Fig. 6c (FEMNIST) gần như trùng nhau (đã xem hình ở p.12); agent đo pixel thấy trùng cả với node 6 ở Fig. 6a. Không nên dùng số liệu theo từng node của bài này.
- (c) Dùng làm related work; ví dụ về pathological partition; ý tưởng tương quan accuracy của node với bậc (bài chỉ nêu định tính).

### 2.2 Kalra et al. 2023, ProxyFL — *Nature Communications* 14:2899 (`s41467-023-38569-4.pdf`)
- (a) MNIST, FaMNIST, CIFAR-10 [p.3]. "8 V100 GPUs, which correspond to 8 clients" [p.3]; mỗi client 1k ảnh (MNIST) [p.3]. Topology: đồ thị mũ có hướng, thay đổi theo round, mỗi client gửi 1 proxy/round, tổng hợp PushSum [p.2–3, p.8]. Non-IID: một lớp chủ đạo chiếm tỉ lệ p_major (0.8 với MNIST); ablation quét 0.1 → 0.8, "pmajor = 0.1 is the IID setting" [p.4]. Adam lr 1e-3, batch 250, 1 epoch/round; **DP-SGD cho mọi baseline** [p.3]. "5 random seeds" [p.3]. Baseline: FedAvg, AvgPush, CWT, FML, Regular, Joint [p.3]. ResNet-18 đổi BatchNorm sang GroupNorm (histopathology) [p.5].
- (b) "as the setting deviates from IID, most methods degrade" [p.4]. Fig. 5a (MNIST, MLP, có DP), p_major 0.1 → 0.8: Regular ≈ 0.88 → 0.61, AvgPush → ≈ 0.53, FedAvg → ≈ 0.43 (đọc từ hình) [p.5]. Giao thức mũ có thời gian giao tiếp mỗi round gần như không đổi khi tăng số client (Fig. 4) [p.4]. Table 2 (histopathology, mean ± std qua client): Regular 0.734 ± 0.105 … Joint 0.905 ± 0.030 [p.7].
- (c) Baseline Regular/Joint; báo cáo mean ± std qua client × 5 seed; p_major làm thang skew phụ; đồ thị mũ (T5). Không so số tuyệt đối với ta vì bài có DP-SGD.

### 2.3 Palmieri et al. 2024 — *Computer Networks* (arXiv 2402.18606v1)
- (a) MNIST [p.10]; 100 node [p.8]; BA (m = 2, 5, 10), ER-idem (xáo lại cạnh của BA, giữ số cạnh), SBM 4 khối [p.9–10]. Non-IID: lớp 0–4 chia đều cho mọi node, lớp 5–9 chỉ giao cho 10% node có degree/betweenness/clustering cao nhất ("hub") hoặc thấp nhất ("edge") [p.10, p.12]. MLP 512-256-128, "SGD … learning rate 0.01 and momentum 0.5", 100 epoch cục bộ giữa hai lần trao đổi [p.14]. DecAvg có trọng số theo kích thước dữ liệu [p.7]; 200 round [p.17]. Không nêu số seed.
- (b) ER, m = 2: accuracy trung bình của nhóm G1 ≈ 0.87 (hub) so với ≈ 0.67 (edge) (đọc từ Fig. 8) [p.16]; "The gap decreases as we increase the overall connectivity" [p.17]. "local clustering is a poor predictor of performance" [p.31]. Hiệu ứng pha loãng: "a single "good" model may get diluted into the neighbourhood average" [p.31]. Algebraic connectivity giải thích thứ tự BA/ER (Table 3) [p.28]. SBM: "it is very hard for external knowledge to enter" các cộng đồng [p.30].
- (c) Báo cáo algebraic connectivity/spectral gap cho mọi topology; tách thống kê node có/không có lớp hiếm; cơ sở cho E7 (vị trí dữ liệu theo độ trung tâm).

### 2.4 Li et al., NIID-Bench — ICDE 2022 (arXiv 2102.02079v4) và mã nguồn
- (a) 10 party, **có server** (khung FedAvg) [p.2–3, p.7]. #C = k: "randomly assign k different label IDs to each party" [p.6]; p_k ∼ Dir(β), dùng β = 0.5 [p.6, p.8]; q ∼ Dir(β) cho quantity skew [p.7]. CNN 2 conv (6/16 kênh) + FC 120/84 [p.7]; SGD lr 0.01, momentum 0.9, batch 64, 10 epoch cục bộ, 50 round [p.7]. "We run three trials" [p.8].
- **Mã nguồn** (`utils.py`, nhánh main, truy cập 2026-09-30): bước `## Balance` ở dòng 327–328; `min_require_size = 10` ở dòng 305; #label dùng `current=[i%K]` ở dòng 369. Bài báo không mô tả ba chi tiết này.
- (b) Table III, MNIST, FedAvg: IID 99.1 ± 0.1%; Dir(0.5) 98.9%; #C=1 29.8 ± 7.9%; #C=2 97.0%; #C=3 98.0%; quantity 99.2% [p.8]. "label distribution skew setting is more challenging than the quantity skew" [p.2]. Với quantity skew, FedAvg (trọng số theo kích thước) có "almost no accuracy loss" [p.8]. FedProx so với FedAvg trên MNIST #C=1: 40.9 ± 23.1% vs 29.8 ± 7.9%; các trường hợp khác chênh dưới 1% [p.8–9]. "MNIST is a simple task under most non-IID settings" [p.9].
- (c) Định nghĩa partition; Table III làm mốc tham chiếu CFL; lưới μ ∈ {0.001, 0.01, 0.1, 1}; tối thiểu 3 lần chạy.

### 2.5 Koloskova et al. 2020 — ICML (arXiv 2003.10422)
- (a) Lý thuyết. W đối xứng, ngẫu nhiên kép, có thể lấy mẫu lại mỗi bước [p.4]. Tham số trộn p: E‖XW − X̄‖² ≤ (1 − p)‖X − X̄‖² [p.5]. Trọng số MH w_ij = min{1/(deg(i)+1), 1/(deg(j)+1)}, trùng công thức §3.3; p = 1 cho đồ thị đầy đủ, Θ(1/n²) cho vòng [p.5]. Thực nghiệm chỉ có least squares, n = 25 [p.8–9].
- (b) Định lý 2 (non-convex): số hạng dị biệt ζ̂ được nhân với τ/p [p.6]; "rates only depend on the ratio p/τ, but not on p or τ individually" [p.7]. Chênh lệch torus–ring "scales approximately linear in the ratio of their mixing parameters" [p.9]. Spectral gap của ring n = 25 là 0.021 [p.28]; công thức §3.3 cho 0.0209 (khớp).
- (c) Cơ sở lý thuyết cho H2 và cho việc quét τ (E3): nên so sánh ở cùng tỉ lệ p/τ.

### 2.6 Kong et al. 2021 — ICML (arXiv 2102.04828)
- (a) CIFAR-10, ImageNet-32, Multi30k; n ∈ {16, 32, 64}; ring là trường hợp chính, thêm complete, exponential, random matching [p.5, p.19–20]. **Dữ liệu IID:** "re-shuffled periodically among nodes" [p.3]. 3 seed [p.5].
- (b) Consensus distance Ξ_t² = (1/n)Σ_i ‖x̄ − x_i‖² [p.3]; "the CCD does not depend on the graph topology" [p.3]. Table 1, CIFAR-10, n = 64: ring 89.58% so với All-Reduce 92.71% [p.1]. Consensus distance quan trọng nhất ở giai đoạn đầu huấn luyện (Table 2) [p.5–6]. Non-IID chỉ bàn lý thuyết: "harder to satisfy the CCD condition" [p.18].
- (c) Log Ξ (đã có trong §3.6). Kết quả Non-IID của dự án là **mở rộng**, không phải lặp lại bài này.

### 2.7 Hsieh et al. 2020 — ICML (arXiv 1910.00189)
- (a) CIFAR-10, ImageNet, Flickr-Mammal, CASIA-WebFace [p.3–4]. Bối cảnh: dữ liệu chia theo phân vùng địa lý, thuật toán BSP / Gaia / FedAvg / DGC trên **parameter server**; không có topology P2P [p.2–4]. "x% non-IID" = x% dữ liệu chia theo nhãn, phần còn lại chia ngẫu nhiên [p.4]. So sánh BN-LeNet với GN-LeNet [p.7].
- (b) CIFAR-10, K = 5, 100% non-IID: "lose 3% to 74% validation accuracy" [p.4]; "Even BSP cannot completely solve this problem" [p.5]. Fig. 5: BSP non-IID 67.1% với BatchNorm so với 79.2% với GroupNorm [p.7]; GroupNorm "better than their BatchNorm counterparts by an additive 10.7%, 19.8%" [p.7].
- (c) Cơ sở cho ADR-004 (không dùng BatchNorm). Hai ý tưởng có thể bổ sung: thang "x% non-IID" và "model traveling" (đánh giá model của node này trên dữ liệu node khác).

### 2.8 Badie-Modiri et al. — arXiv 2403.15855v4
- (a) Tập con của MNIST, So2Sat, CIFAR-10; n từ 8 đến 512; complete, BA, k-regular, ER, power-law [p.9, p.15–17]. Mọi cấu hình MNIST đều IID; Non-IID (Zipf) chỉ dùng với So2Sat [p.7, p.27]. DecAvg có trọng số theo kích thước [p.5]. **Tiền đề:** "initialisation and operations of the nodes have to be performed in an uncoordinated manner" [p.2]; baseline là mỗi node tự khởi tạo He độc lập [p.4, p.8].
- (b) Khởi tạo He độc lập gây một bình nguyên loss kéo dài theo n (số mũ 0.4–1) [p.8–9]. Đề xuất nhân khởi tạo He với ‖v_steady‖⁻¹ (≈ √n trên k-regular/ER) [p.12–13]. "more frequent communication translates to both a lower final test loss" [p.19].
- (c) Giả định A3 của dự án (khởi tạo chung) thuộc loại "coordinated initialisation" theo bài [p.8] → nêu là hạn chế. Nếu chạy ablation khởi tạo độc lập, dùng hệ số hiệu chỉnh của bài; không so sánh trực tiếp với số liệu MNIST của bài (IID, đo bằng loss).

### 2.9 Khảo sát
- **Beltrán et al. 2023 (arXiv 2211.08413):** phân loại topology thành fully connected / partially connected (star, ring, random Erdős–Rényi) / theo cụm [p.7–8]. Không nêu độ đo định lượng mức Non-IID [p.25].
- **Jimenez G. et al. 2024 (arXiv 2411.12377):** liệt kê độ đo JS, Hellinger, EMD/Wasserstein, KL… [p.12]. Chỉ "13.1% of them employed metrics" định lượng Non-IID; 60.3% bài label skew không nhắc tới quantity skew dù cách chia của họ sinh ra nó [p.17]. FedProx "sensitive to the choice of the proximal term" [p.14]. Cả ba điều này ủng hộ việc §3.2 báo cáo JS kèm CV kích thước.

## 3. Chưa kiểm chứng được
- **Hegedűs, Danner, Jelasity 2021** (JPDC 148:109–124, DOI 10.1016/j.jpdc.2020.10.006; metadata xác minh qua Crossref): proxy chặn bản PDF ở kho SZTE và ScienceDirect, nên nội dung chưa được đọc.
- **ACM CSUR 2025** (DOI 10.1145/3785657): bản trả phí; chỉ đọc được mô tả qua kết quả tìm kiếm.

## 4. Gợi ý bổ sung từ tài liệu (cần nhóm quyết; chưa đưa vào kế hoạch)
- Báo cáo algebraic connectivity cùng với spectral gap (Palmieri, Table 3).
- Bổ sung thang "x% non-IID" của Hsieh bên cạnh Dirichlet và #C, như một mức kiểm soát tuyến tính.
- Độ đo "model traveling" (Hsieh) làm thước đo khác biệt giữa các node.
- Vẽ loss theo "bước × số epoch" để tách chi phí tính toán khỏi chi phí giao tiếp (Badie-Modiri).
- Với E3, so sánh thêm các cặp cấu hình có cùng tỉ lệ p/τ (Koloskova).
