# Ảnh hưởng của dữ liệu Non-IID lên học liên kết phi tập trung (DFL) — Báo cáo

Đề 2, ĐATH HK261. Kiến trúc: `docs/report/kien_truc.md`. Phụ lục (attempt lỗi, phân kỳ, `code_version` bị thay): `docs/report/phu_luc.md`. Mọi hình và bảng sinh bằng `python -m analysis.plots --results results/ --out figures/` và có trong `figures/PROVENANCE.json`; mọi kết luận dẫn chiếu hình hay bảng và run ID (INT-4).

## 1. Bài toán và câu hỏi nghiên cứu

DFL bỏ máy chủ trung tâm: mỗi node tự huấn luyện trên dữ liệu riêng rồi trộn model với láng giềng trên một đồ thị P2P. Khi dữ liệu giữa các node lệch nhau (Non-IID), model cục bộ trôi về các hướng khác nhau và việc trộn qua đồ thị thưa chậm kéo chúng lại.

- **RQ1:** Non-IID tăng thì accuracy, tốc độ hội tụ và phương sai giữa node đổi thế nào?
- **RQ2:** Topology nào chống chịu tốt nhất; mức suy giảm có giải thích được bằng spectral gap hay consensus distance không?
- **RQ3:** Ở cùng tổng số bước cục bộ, τ và μ đánh đổi accuracy với chi phí giao tiếp ra sao?
- **RQ4:** Quantity skew có tác động như label skew không; trọng số trộn có quan trọng không?
- Giả thuyết đăng ký trước H1–H8 (KH §2.3, ADR-011), mỗi giả thuyết một phép so sánh chính khai báo trong thẻ `docs/experiments/E<k>.md` trước khi chạy.

## 2. Phương pháp

### 2.1 Hệ thống

Mô phỏng tự viết trên PyTorch + NetworkX (ADR-001), vòng Adapt-Then-Combine đồng bộ (ADR-006), mọi giao tiếp qua một Transport đếm message và byte (ADR-002). Có bản đa tiến trình: mỗi node một tiến trình hệ điều hành, trao đổi qua `torch.distributed` (gloo), cho kết quả trùng từng byte với bản một tiến trình (ADR-010, T-INV-13). Chi tiết ở `kien_truc.md`.

### 2.2 Dữ liệu và cách chia Non-IID

- MNIST: 55,000 ảnh huấn luyện (sau khi tách 5,000 ảnh validation phân tầng để chọn η), 10,000 ảnh test (ADR-003).
- N = 20 node, cách chia đúng thuật toán NIID-Bench, mỗi node ít nhất 64 mẫu (KH §3.2):
  - P0 IID; P1 Dirichlet label skew với α ∈ {100, 1, 0.5, 0.1, 0.05} (α nhỏ thì lệch mạnh);
  - P2 #C = k: mỗi node chỉ có k ∈ {2, 1} lớp; P3 quantity skew β = 0.5 (nhãn IID, kích thước lệch).
- Mức Non-IID đo bằng JS divergence trung bình giữa phân bố nhãn của node và phân bố gộp (hình P7).

### 2.3 Topology và trọng số trộn

Ring (T1), Erdős–Rényi liên thông p = 0.3 (T2), random pairwise gossip trên $K_N$ đổi mỗi round (T3), gossip giới hạn trên cạnh ER (T3b), fully-connected (T4); trọng số Metropolis–Hastings mặc định (ADR-005, ADR-009). Baseline: B1 tập trung, B1-4k, B2 chỉ cục bộ (W = I), B3 FedAvg có server (trọng số $n_k/n$).

### 2.4 Mô hình và huấn luyện

SmallCNN 46,730 tham số, không BatchNorm, SGD thuần, B = 32, η = 0.1 chọn trên validation ở E0a theo quy tắc trường hợp xấu nhất (ADR-004, ADR-007). τ·R = 4,000 bước cục bộ mỗi node ở mọi điều kiện; mọi node cùng $w^0$. DFedProx: μ = 0.5 (ημτ = 1 ở τ = 20) và μ = 0.05 (QS-2). E8 dùng thang bốn model 7,850 → 1,663,370 tham số (ADR-011).

### 2.5 Độ đo và thống kê

- Accuracy cuối: trung bình qua node trên đủ 10,000 ảnh test; macro-F1, Acc^loc theo phân bố nhãn của node; std và node kém nhất (phương sai giữa node); drift trước và consensus distance sau tổng hợp.
- Tốc độ hội tụ R@x: số bước cục bộ tới lần vượt ngưỡng x bền vững đầu tiên, trên lưới đánh giá 100 bước (REQ-04). Chi phí giao tiếp: message, byte tích lũy, tải theo node, bytes@x, và thời gian giao tiếp đo trên process runtime (REQ-05, ADR-013).
- Seed ghép cặp (cùng partition, vị trí node, $w^0$): 5 seed cho thí nghiệm lõi, 3 cho ablation. Quy tắc quyết định ADR-007: "ủng hộ" khi đúng chiều ở cả 5 seed (p = 1/32) và CI 95% không chứa 0; 3 seed tối đa "gợi ý".

## 3. Thiết kế thí nghiệm

| Nhóm | Mục đích | Lượt | Seed |
|---|---|---|---|
| E0a | chọn η trên validation | 24 | 100 |
| E0b | sanity SAN-1…3, baseline, độ nhạy η | 43 | 3 / 5 / 1 |
| E1 | RQ1, H1: {ring, FC} × 7 mức Non-IID | 70 | 5 |
| E2 | RQ2, H2–H4: {ER, gossip} × {α = 1, 0.1, #C = 2} | 30 | 5 |
| E3a | ba thuật toán, H5 | 24 | 3 |
| E4a | quantity skew | 6 | 3 |
| E8 | chi phí giao tiếp theo độ phức tạp model, H8 (process runtime) | 72 | 3 |
| E1c, E2b, E3b, E4 | Tier 2: H6, H4 công bằng, H5/H7, RQ4 | 81 | 3 |

Mọi lượt chạy trên một máy (i5-12400, 6 nhân), một luồng mỗi lượt, cùng `code_version` `eff6fa19…`.

## 4. Kết quả

(điền sau khi có kết quả: mỗi RQ một mục, mỗi giả thuyết kèm kết luận, giá trị theo seed, CI, hình và run ID)

## 5. Khuyến nghị

(điền sau khi có kết quả: khuyến nghị chính chỉ khi "ủng hộ"; khuyến nghị tạm khi "gợi ý", ghi số seed; REQ-13)

## 6. Hạn chế (INT-4)

- **η chung cho mọi τ,** chọn theo ô Non-IID khó nhất (ring-#C2); có thể bất lợi cho τ = 1 và τ = 50, và có thể làm nhỏ hiệu ứng Non-IID đo được. η = 0.1 là đầu trên của dải đăng ký và không được quét thêm (SPEC §8).
- **Gossip trên $K_N$** giả định mọi node biết địa chỉ của mọi node; E2b đối chứng bằng gossip trên cạnh ER.
- **Lệch mục tiêu:** W ngẫu nhiên kép tối ưu $\frac1N\sum_i F_i$ chứ không $\sum_i \frac{n_i}{n} F_i$; E1c chỉ ước lượng phần này, không tách hoàn toàn với drift.
- **Tri thức chung của DFL đồng bộ:** khởi tạo chung $w^0$, đồng hồ round, seed chung của lịch gossip, bậc láng giềng (ADR-002).
- **Size-corrected** dùng $\bar n$ là đại lượng toàn cục, và tương đương bước học theo node.
- **D-PSGD** ở đây dùng minibatch B = 32 và trọng số MH, khác FedSGD gốc (full-batch).
- **Quantity skew** chỉ có một mức (β = 0.5).
- **Accuracy chỉ đo sau tổng hợp,** nên phần tụt trong round (lớn dần theo τ) không được thấy, và τ lớn có thể được lợi.
- **Siêu dữ liệu** ($n_k$ gửi lên server, bậc láng giềng) không được đếm vào byte; nhỏ hơn 0.005% một message.
- **Bội kiểm định:** bảy phép so sánh chính, mỗi phép p = 1/32 với 5 seed; nếu mọi H0 đúng thì xác suất có ít nhất một kết luận "ủng hộ" sai tối đa khoảng 20% ($1-(31/32)^7$).
- **RQ2:** spectral gap chỉ có ba cụm giá trị (ring, ER, FC) và lẫn với bậc, số message, byte; phần "giải thích bằng spectral gap" chỉ ở mức mô tả.
- **Thời gian giao tiếp** đo trên một máy (loopback), không phải mạng thật; `comm_s` gồm cả thời gian chờ láng giềng chậm hơn (lệch tiến độ), nên tăng theo cả khối lượng tính (thẻ E8). Máy chạy lưới là máy dùng chung (trình duyệt, IDE), nên số đo thời gian có nhiễu.
- **Máy chạy lưới** khác máy chủ của kế hoạch ban đầu: E0a chạy lại trên máy này dưới `code_version` mới (phụ lục).

## 7. Tái lập

Cài theo SPEC §4 từ `requirements.lock`, check out `git_commit` trong manifest, chạy lại lượt bằng `python -m dfl run`; REP-1 kiểm 2 run_id ngẫu nhiên trùng từng byte (`docs/VALIDATION.md`).
