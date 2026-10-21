# Phản hồi góp ý buổi báo cáo tiến độ 2

Tài liệu này đi qua từng góp ý của GVHD ở buổi 2 (theo bản chép lời, mốc thời gian trong ngoặc vuông), ghi dự án (nhánh `binh`) đã đáp ứng thế nào, bằng chứng ở đâu (lệnh, file, số đo), và ý nên đưa lên slide buổi sau. Góp ý được nói với repo của Vy và Nghĩa, nhưng áp dụng cho cả dự án. Thay đổi thiết kế đi kèm ghi ở [ADR-015](../decisions/ADR-015-feedback-2-lattice-hub-leaf-real-time.md).

Mọi số ở đây sinh lại được bằng lệnh ghi cạnh nó. Số có nhãn "lượt thử" chỉ để kiểm đường chạy, không phải kết quả thí nghiệm.

## Tóm tắt

| # | Góp ý | Trạng thái | Bằng chứng chính |
|---|---|---|---|
| 1 | Ring và FC quá đơn giản; thêm lattice bậc 4 và các loại khác [00:57], [21:23] | Đã thêm lattice (T7) và BA (T6); có tổng cộng 7 topology | `src/dfl/network/topology.py`; bảng ở mục 1 |
| 2 | Mô tả % hub (≥ 6) và % leaf (1–2) của đồ thị random ở khoảng 20 node; giải thích xác suất p [27:18] | Đã có: `describe` in bậc, % hub, % leaf, bảng ER theo p và BA theo m | `python -m dfl describe`; mục 1 |
| 3 | Demo chia dữ liệu từng bước trên dataset cụ thể; ảnh hưởng của α và tham số của Pathological; so giữa dataset [12:31], [15:45] | Đã có script demo, kiểm trùng partitioner của dự án | `python -m scripts.demo_partition`; mục 2 |
| 4 | Pathological có thật sự "không thực tế"? Số mẫu mỗi nhãn ở hai client cùng nhãn khác nhau thế nào [05:50]–[10:18] | Dự án giữ cả Dirichlet lẫn #C = k; trả lời bằng số đo | mục 2 |
| 5 | Dùng đúng thuật ngữ FedProx, non-IID, huấn luyện cục bộ [31:24]–[36:30] | Đã đúng trong code và tài liệu; có phần giải thích cho slide | mục 3 |
| 6 | Đưa FedProx vào thực nghiệm với α thấp (0.1–0.5) [37:28], [59:28] | Đã có: E3a chạy DFedProx ở α = 0.1, μ ∈ {0.05, 0.5} | `configs/experiments/` (E3a), ADR-007 |
| 7 | Ngoài số round phải đo thời gian thực [43:45] | Đã có thời gian theo pha và theo node; thêm thời gian để đạt ngưỡng và P17 | mục 4 |
| 8 | Theo dõi từng node; so đường hội tụ hub với leaf [00:00], [46:03], [57:31] | Đã có `acc_per_node` ở mọi mốc; thêm P16 | mục 5 |
| 9 | Tách chi phí truyền thông với chi phí huấn luyện [52:22] | Đã tách từ đầu (INV-2, ADR-013) | mục 6 |
| 10 | F1 không phân biệt được giữa các α; nghi do chạy tuần tự [55:22] | Thiết kế khác (N = 20, ngân sách cố định, mức lệch cực đoan); tuần tự và đa tiến trình cho kết quả trùng từng byte | mục 7 |
| 11 | Chạy song song nhiều process (Docker hoặc multiprocessing) [29:45], [55:22] | Đã có process runtime: mỗi node một tiến trình qua gloo; Docker là tùy chọn, chưa làm | ADR-010 |
| 12 | Vẽ đồ thị theo từng round, không chỉ round cuối [59:28] | Đã có P1 (theo bước), P12, P16, P17 | mục 8 |
| 13 | Ghi thư viện của từng hàm lên slide [19:11] | Bảng thư viện theo module | mục 9 |

## 1. Topology và cấu trúc đồ thị

**GVHD:** ring là thưa nhất, FC là dày nhất, nên chỉ có hai cái đó thì quá đơn giản; nên thêm lattice (mỗi node 4 kết nối) và các loại khác. Đồ thị random phải được mô tả: ở khoảng 20 node, bao nhiêu % là hub (≥ 6 kết nối), bao nhiêu % là leaf (1–2 kết nối); từ mô tả đó mới chọn được p.

**Dự án có 7 topology** (N = 20, trọng số Metropolis–Hastings), sinh bằng `python -m dfl describe --config configs/base.yaml --seeds 20 --out build/describe`:

| Topology | Hàm NetworkX | Bậc TB (min–max) | % hub | % leaf | Spectral gap | Message mỗi round |
|---|---|---|---|---|---|---|
| star (tham chiếu) | `star_graph` | 1.9 (1–19) | 5% | 95% | 0.050 | 38 |
| T1 ring | `cycle_graph` | 2 (2–2) | 0% | 100% | 0.033 | 40 |
| T6 Barabási–Albert m = 2 | `barabasi_albert_graph` | 3.6 (1–15) | 16% | 48% | TB 0.091 | 72 |
| T7 lattice 4 × 5 (xuyến) | `grid_2d_graph(periodic=True)` | 4 (4–4) | 0% | 0% | 0.276 | 80 |
| T2 Erdős–Rényi p = 0.3 | `erdos_renyi_graph` | 5.8 (2–12) | 56% | 2% | TB 0.233 | khoảng 114 |
| T3 random gossip | (ghép cặp ngẫu nhiên mỗi round) | 1 mỗi round | — | — | tốc độ trộn 0.312 | 20 |
| T4 fully connected | `complete_graph` | 19 (19–19) | 100% | 0% | 1.000 | 380 |

- **Đồng nhất** (mọi node cùng bậc): ring, lattice, FC. Theo định nghĩa của GVHD, mọi node của ring là leaf, mọi node của FC là hub, lattice không có hub hay leaf.
- **Không đồng nhất:** BA (có cả hub lẫn leaf: dùng để so hub với leaf) và ER (phần lớn là hub, gần như không có leaf).
- Spectral gap đo tốc độ trộn của W: gần 0 là thông tin lan chậm (ring), bằng 1 là trộn xong trong một round (FC).

**Giải thích p của ER** (bảng ER theo p của `describe`, 1,000 đồ thị mỗi dòng):

| p | Liên thông (đo) | exp(−N(1 − p)^{N−1}) | Bậc TB (min–max) | % hub | % leaf |
|---|---|---|---|---|---|
| 0.10 | 0.057 | 0.067 | 2.5 (1–8) | 2% | 54% |
| 0.15 | 0.411 | 0.402 | 3.2 (1–10) | 7% | 37% |
| 0.20 | 0.754 | 0.750 | 3.9 (1–12) | 18% | 21% |
| 0.25 | 0.929 | 0.919 | 4.8 (1–13) | 34% | 10% |
| 0.30 | 0.982 | 0.977 | 5.7 (1–14) | 54% | 4% |
| 0.40 | 1.000 | 0.999 | 7.6 (1–16) | 84% | 0% |
| 0.50 | 1.000 | 1.000 | 9.5 (2–18) | 97% | 0% |

- Trong G(N, p), mỗi cặp node nối với nhau với xác suất p, độc lập. Bậc của một node có trung bình (N − 1)p.
- Đồ thị không liên thông chủ yếu vì có node cô lập. Số node cô lập xấp xỉ Poisson với trung bình N(1 − p)^{N−1}, nên P(liên thông) ≈ exp(−N(1 − p)^{N−1}). Cột đo khớp cột công thức ở mọi p.
- Ngưỡng liên thông của Erdős–Rényi (1960) là ln N / N = 0.15 với N = 20: dưới ngưỡng này phần lớn đồ thị có node cô lập.
- **Đây chính là ý nghĩa của `safe_p = log(n)/n + 0.15` của nhóm Vy:** ln N / N là ngưỡng liên thông, cộng 0.15 để vượt hẳn ngưỡng. Ở N = 20 công thức cho đúng 0.30, tức p của dự án, và khi đó 98% đồ thị rút ra đã liên thông. Ở N = 5 (repo của Vy, Nghĩa) nó cho 0.47.
- Dự án không chuyển sang ring khi rút phải đồ thị không liên thông: nó rút lại tới khi liên thông (tối đa 1,000 lần). Chuyển sang ring như repo của Vy, Nghĩa sẽ làm nhãn "random" sai.
- **Chọn p theo cấu trúc mong muốn:** cần mạng thưa nhiều leaf thì p ≈ 0.15–0.2, nhưng khi đó 25–60% đồ thị rút ra không liên thông. Muốn có cả hub lẫn leaf mà luôn liên thông thì dùng BA m = 2 (16% hub, 48% leaf). ER p = 0.3 là mạng dày vừa: khoảng một nửa số node là hub.

**Thí nghiệm có lattice và BA:** E9 ([thẻ](../experiments/E9.md)) chạy {ring, lattice, ER, BA, FC} × {IID, α = 0.1} × 3 seed = 30 lượt (khoảng 9.6 giờ-lõi), chờ nhóm duyệt. Giả thuyết chính H10: trên BA ở α = 0.1, hub có accuracy cuối cao hơn leaf.

**Ý cho slide:** bảng 7 topology (ghi tên hàm NetworkX); bảng ER theo p cùng công thức Poisson để giải thích p = 0.3; vì sao BA là đồ thị để so hub với leaf.

## 2. Chia dữ liệu: Dirichlet, Pathological và các cách khác

**GVHD:** không chấp nhận lập luận nói miệng; cần demo từng bước trên một dataset cụ thể; cho thấy α đổi phân bố ra sao (α nào lệch nhiều, α nào gần như không đổi gì), tham số tương tự của Pathological ảnh hưởng thế nào, và cùng α trên dataset khác có giống không.

**Demo:** `python -m scripts.demo_partition --out build/demo_partition` (khoảng 6 s). Script làm lại từng bước của hai thuật toán, theo đúng thứ tự rút số ngẫu nhiên của `src/dfl/data/partition.py`, in các bước trung gian, rồi kiểm kết quả **trùng từng chỉ số** với partitioner của dự án (cả hai đều trùng). Kết quả ghi vào `build/demo_partition/demo_partition.md`, kèm hình `alpha_effect.png` và `P7-describe.png` (heatmap node × lớp).

**Dirichlet (NIID-Bench `noniid-labeldir`), từng bước,** với mỗi lớp c:
1. xáo các mẫu của lớp;
2. rút tỉ lệ p ~ Dir_N(α), mỗi node một phần;
3. node đã đủ cỡ trung bình (tổng số mẫu / N) không nhận thêm: tỉ lệ của nó về 0 và phần còn lại chuẩn hóa lại;
4. cắt dãy mẫu theo tỉ lệ tích lũy.

Hết 10 lớp, nếu có node nhỏ hơn 64 mẫu thì rút lại cả lượt. Vì vậy số mẫu của một lớp ở một node là ngẫu nhiên.

**Ảnh hưởng của α** (MNIST, 55,000 mẫu của node, N = 20, trung bình 5 seed):

| α | Nα | CV lý thuyết | JS TB (bit) | Số lớp ≥ 5% mỗi node | % ô node × lớp rỗng | Kích thước node (min–max) |
|---|---|---|---|---|---|---|
| 1000 | 20000 | 0.03 | 0.000 | 10.0 | 0% | 2,700–2,799 |
| 100 | 2000 | 0.10 | 0.001 | 10.0 | 0% | 2,594–2,907 |
| 10 | 200 | 0.31 | 0.022 | 9.6 | 1% | 2,285–3,039 |
| 5 | 100 | 0.43 | 0.047 | 8.9 | 3% | 2,153–3,179 |
| 1 | 20 | 0.95 | 0.176 | 6.1 | 8% | 1,611–3,797 |
| 0.5 | 10 | 1.31 | 0.287 | 4.7 | 15% | 1,054–5,153 |
| 0.3 | 6 | 1.65 | 0.389 | 3.6 | 23% | 1,052–4,995 |
| 0.1 | 2 | 2.52 | 0.561 | 2.4 | 53% | 331–5,969 |
| 0.05 | 1 | 3.08 | 0.633 | 1.9 | 65% | 200–6,151 |

- **Lý thuyết:** phần của một lớp mà một node nhận có phân bố Beta(α, (N − 1)α), trung bình 1/N, hệ số biến thiên √((N − 1)/(Nα + 1)). Đại lượng quyết định mức lệch là **Nα**, không phải α riêng lẻ: cùng α nhưng N khác thì mức lệch khác.
- **α lệch ít:** α ≥ 10 (Nα ≥ 200) gần như IID: JS ≤ 0.02, mọi node có gần đủ 10 lớp. Tăng α từ 100 lên 1000 gần như không đổi gì.
- **α lệch nhiều:** α ≤ 0.1 (Nα ≤ 2): mỗi node chỉ còn khoảng 2 lớp đáng kể, hơn nửa số ô node × lớp rỗng. Từ 0.1 xuống 0.05 thay đổi ít hơn từ 1 xuống 0.1.
- Thang α của lưới lõi (100, 1, 0.5, 0.1, 0.05) vì vậy đi từ gần IID tới gần Pathological.

**Pathological #C = k (NIID-Bench `noniid-#label{k}`), từng bước:**
1. node i giữ nhãn i mod 10, để nhãn nào cũng có chủ;
2. k − 1 nhãn còn lại của mỗi node rút ngẫu nhiên, không trùng;
3. mẫu của mỗi nhãn được xáo rồi **chia đều** cho các node giữ nhãn đó.

**Trả lời câu hỏi [09:14]–[10:18] của GVHD** ("số mẫu của cùng một nhãn ở hai client khác nhau thay đổi thế nào, ngẫu nhiên hay được phân bổ cụ thể"):
- Hai client cùng giữ một nhãn nhận số mẫu của nhãn đó gần như bằng nhau, chênh nhau tối đa 1. Số đó không ngẫu nhiên.
- Cái ngẫu nhiên là **client nào giữ nhãn nào**. Vì vậy số client giữ mỗi nhãn khác nhau, và nhãn ít chủ thì mỗi chủ nhận phần lớn hơn.
- Ví dụ MNIST, k = 2, seed 0: nhãn 6 có 7 node giữ, mỗi node nhận 775 mẫu; nhãn 1 có 3 node giữ, mỗi node nhận 2,060 mẫu. Kích thước node vì vậy không đều.

**Ảnh hưởng của k** (MNIST, N = 20, 5 seed):

| k | JS TB (bit) | Số lớp mỗi node | Kích thước node (min–max) |
|---|---|---|---|
| 1 | 0.758 | 1 | 2,484–3,090 |
| 2 | 0.614 | 2 | 1,973–4,141 |
| 3 | 0.499 | 3 | 2,069–3,943 |
| 5 | 0.317 | 5 | 2,377–3,128 |
| 10 | 0.000 | 10 | 2,747–2,755 |

- k là tham số **rời rạc**: không tạo được mức lệch nằm giữa k = 1 và k = 2. α thì liên tục.
- #C = 2 (JS 0.614) lệch gần bằng Dirichlet α = 0.05 (0.633), nhưng theo cách khác: mỗi node có đúng 2 lớp, không có "đuôi" ít mẫu.

**Pathological có "không thực tế" không?** GVHD nhận xét đúng [05:50]: trong thực tế một client có thể chỉ có vài nhãn. Vì vậy dự án **giữ cả hai họ**:
- Dirichlet cho thang liên tục: α ∈ {100, 1, 0.5, 0.1, 0.05};
- #C = k cho các mức rời rạc cực đoan: k ∈ {2, 1}.

Cả hai đều chạy trong E1, và mọi cách chia được đặt lên cùng một trục là JS đo được (REQ-02). Không nên nói một cách "đúng" và cách kia "sai": chúng là hai mô hình khác nhau của label skew.

**Các cách chia khác** (trả lời [11:05] "chỉ có 2 phương pháp thôi hay sao"; MNIST, N = 20):

| Cách chia | Lệch gì | JS TB | Kích thước node (min–max) |
|---|---|---|---|
| IID | không | 0.001 | đều |
| Quantity skew β = 0.5 | chỉ số lượng | 0.004 | 85–11,536 |
| Shards (McMahan 2017) | nhãn, kích thước đều | 0.596 | 2,750–2,750 |
| rare_classes K = 2 (E7) | lớp 5–9 chỉ ở 2 node | 0.289 | 1,402–14,880 |

Feature skew (cùng nhãn nhưng ảnh khác kiểu, ví dụ FEMNIST theo người viết) nằm ngoài phạm vi, vì MNIST không có thông tin người viết.

**So giữa các dataset** (cùng script, N = 20, trung bình 5 seed; CIFAR-10 tải bằng `python -m dfl download cifar10`):

| Dataset | Số mẫu của node | Mẫu mỗi lớp | JS α = 100 | JS α = 1 | JS α = 0.5 | JS α = 0.1 | JS α = 0.05 | JS #C = 2 | JS #C = 1 |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | 55,000 | 4,969–6,180 | 0.001 | 0.176 | 0.287 | 0.561 | 0.633 | 0.614 | 0.758 |
| CIFAR-10 | 45,000 | 4,500 mỗi lớp | 0.002 | 0.166 | 0.297 | 0.573 | 0.635 | 0.613 | 0.758 |

- **Cùng α thì cùng mức lệch nhãn** trên hai dataset (đường JS theo α của hai dataset gần như trùng nhau trong `alpha_effect.png`). Lý do: độ lệch nhãn của một cách chia chỉ phụ thuộc số lớp, mức cân bằng giữa các lớp và Nα, không phụ thuộc nội dung ảnh. Kết quả này khớp với số 20 seed đã đo trước: CIFAR-10 ở ADR-012 và MNIST ở bảng KH §3.2.
- **Khác ở kích thước node:** CIFAR-10 có ít mẫu hơn (45,000 so với 55,000), nên ở α = 0.05 node nhỏ nhất chỉ còn 158 mẫu (MNIST 200) và phải rút lại trung bình 4.6 lần để mọi node có ít nhất 64 mẫu (MNIST 1.4 lần). Với dataset nhỏ hơn nữa hay nhiều lớp hơn, cùng α sẽ lệch khác.
- **Khác ở accuracy:** CIFAR-10 khó hơn nhiều, nên cùng một mức lệch nhãn sẽ làm giảm accuracy nhiều hơn. Điều này phải đo bằng thí nghiệm, không suy ra được từ JS.

**Ý cho slide:** bốn bước của Dirichlet trên MNIST, kèm tỉ lệ rút được ở lớp 0 và ma trận node × lớp; bảng α với cột Nα; bảng k và câu trả lời "chia đều cho các chủ"; hình `alpha_effect.png`; heatmap P7.

## 3. Thuật ngữ: huấn luyện cục bộ, tổng hợp, FedProx, non-IID

**GVHD [31:24]–[36:30]:** huấn luyện cục bộ là học máy thông thường trên dữ liệu của node, chưa phải FL; FL bắt đầu khi giao tiếp và tổng hợp. FedProx là tên một cơ chế; số hạng proximal nằm trong huấn luyện. Non-IID là chênh lệch dữ liệu **giữa các node**; mất cân bằng **trong một node** là vấn đề của học máy thông thường.

Cách dự án dùng (khớp với góp ý, và khớp bài gốc):
- **Một round** (ADR-006, kiểu Adapt-then-Combine): node huấn luyện τ bước SGD trên dữ liệu của nó (Adapt), gửi model cho láng giềng, rồi tổng hợp model nhận được bằng ma trận W (Combine). Cả repo của Vy, Nghĩa lẫn dự án đều theo thứ tự này.
- **FedProx** (Li et al. 2020) là một **thuật toán FL**, gồm hai phần:
  - ở bước huấn luyện, hàm mất mát của node có thêm số hạng (μ/2)‖w − w_đầu round‖², kéo model không trôi quá xa model chung;
  - bước tổng hợp giữ như FedAvg (trong DFL: trộn bằng W như gossip).
- Nói "FedProx là cơ chế tổng hợp" thì đúng theo nghĩa nó là tên của cả thuật toán FL. Nói "proximal là một số hạng trong huấn luyện" cũng đúng. Hai cách nói không mâu thuẫn. Dự án gọi bản phi tập trung là **DFedProx**, đặt cạnh **DFedAvg** (μ = 0) và **D-PSGD** (τ = 1) trong cùng một vòng lặp tham số (τ, μ).
- **Non-IID** trong dự án luôn là chênh lệch phân bố nhãn **giữa các node**, đo bằng JS giữa phân bố của từng node và phân bố chung (mục 2). Dự án không xử lý mất cân bằng trong một node (không reweight, không oversample), vì đó không phải câu hỏi của đề tài.
- **Vì sao FedProx giúp với non-IID:** khi dữ liệu các node lệch nhau, τ bước cục bộ kéo model mỗi node về cực tiểu riêng của nó (client drift). Số hạng prox giới hạn độ trôi đó. GVHD đề xuất thử ở α thấp [59:28]: E3a chạy DFedProx ở α = 0.1, với μ ∈ {0.05, 0.5} chọn theo quy tắc ημτ ∈ {0.1, 1} (ADR-007); E3b quét thêm μ và τ.

**Ý cho slide:** sơ đồ một round (huấn luyện cục bộ → gửi → tổng hợp bằng W); một dòng "FedProx = huấn luyện có số hạng prox + tổng hợp như FedAvg"; định nghĩa non-IID là giữa các node, đo bằng JS.

## 4. Thời gian thực, không chỉ số round

**GVHD [43:45]:** mạng thưa có round ngắn nhưng cần nhiều round; mạng dày có round dài (xử lý nhiều thông tin) nhưng cần ít round. Chỉ so số round thì đánh giá sai; phải so thời gian.

Dự án có ba loại số đo:

1. **Thời gian đo được** (`timing.jsonl` của mọi lượt, ADR-013): giây mỗi round cho từng pha (huấn luyện, giao tiếp, tổng hợp, đánh giá) và cho từng node (`train_s_per_node`, `comm_s_per_node`). Hình P12 (c) và P13 (c).
   - Ở **runtime inprocess**, các node chạy lần lượt trong một tiến trình, nên thời gian là tổng qua node, không phải thời gian của một hệ phân tán.
   - Ở **process runtime** (mỗi node một tiến trình), thời gian giao tiếp là thời gian thật qua gloo, nhưng trên một máy nên không có giới hạn băng thông của mạng thật (ADR-010).
2. **Thời gian phân tán ước tính** (mới, ADR-015): mỗi khoảng giữa hai mốc tốn (thời gian huấn luyện của node chậm nhất) + (byte gửi và nhận của node bận nhất) / B, với B là băng thông.
   - Thời gian huấn luyện của từng node là số đo thật; byte của từng node do Transport đếm (INV-2), kể cả server ảo của FedAvg có server.
   - Giả định: các node huấn luyện song song; gửi và nhận dùng chung B; bỏ qua độ trễ và thời gian cộng vector khi tổng hợp.
   - Mô hình này thể hiện đúng lập luận của GVHD: với topology dày, node bận nhất phải nhận nhiều model mỗi round. Ví dụ, mỗi model SmallCNN là 0.187 MB; mỗi round, node của FC N = 20 gửi và nhận 2 × 19 model = 7.1 MB, node của ring chỉ 0.75 MB. Ở 10 Mbit/s, phần giao tiếp mỗi round tốn khoảng 5.7 s với FC và 0.6 s với ring, trong khi huấn luyện 20 bước chỉ khoảng 0.2 s.
3. **Thời gian để đạt ngưỡng** (mới): `python -m analysis.aggregate` ghi thêm các cột `s90_measured`, `s95_measured` (thời gian đo) và `s90_est`, `s95_est` (thời gian ước tính ở 10 Mbit/s), nội suy tại bước đạt 90% hay 95% (REQ-04), cạnh R@x (số bước) và bytes@x.

**Hình P17** (mới): (a) accuracy theo round; (b) accuracy theo thời gian ước tính ở 10 Mbit/s; (c) thời gian ước tính để đạt 90% theo băng thông 1–1,000 Mbit/s (trục log). Ô (c) trả lời thẳng câu hỏi "mạng dày có nhanh hơn không": nó cho thấy ở băng thông nào topology nào về đích trước.
- Lượt thử trên dữ liệu tổng hợp (build/t31b) cho đúng hình mẫu GVHD mô tả: FC cần ít round nhất, nhưng theo thời gian ước tính ở 10 Mbit/s thì ring và gossip về đích trước; khoảng cách thu hẹp khi băng thông tăng.
- **Lượt thử trên MNIST** (α = 0.1, ngân sách fast 1,000 bước, τ = 10, η = 0.1, seed 0, ghi ở `build/t31g`; 1 seed nên chỉ để kiểm đường chạy, không phải kết quả):

| Topology | Accuracy cuối TB (node thấp nhất) | Bước để đạt 90% | Thời gian ước tính để đạt 90%, 10 Mbit/s | Tổng thời gian ước tính, 10 Mbit/s (trong đó giao tiếp) |
|---|---|---|---|---|
| FC | 97.7% (97.7%) | 156 | 89.6 s | 574.6 s (568.2 s) |
| lattice | 95.5% (94.0%) | 344 | 43.5 s | 126.2 s (119.6 s) |
| ER p = 0.3 | 95.4% (86.5%) | 310 | 94.6 s | 305.5 s (299.1 s) |
| BA m = 2 | 92.1% (85.9%) | chưa đạt bền vững | — | 305.5 s (299.1 s) |
| ring | 90.4% (82.4%) | chưa đạt bền vững | — | 66.3 s (59.8 s) |

  - Đúng lập luận của GVHD: FC cần ít bước nhất (156 so với 344 của lattice), nhưng ở 10 Mbit/s, lattice đạt 90% sau 43.5 s, FC sau 89.6 s. Theo ô (c) của P17, FC chỉ nhanh hơn lattice khi băng thông từ khoảng 380 Mbit/s trở lên.
  - Với SmallCNN ở 10 Mbit/s, phần tính toán chỉ khoảng 6.4 s trong tổng thời gian ước tính, tức giao tiếp chiếm hơn 90%. Vì vậy, khi băng thông hẹp, thứ tự các topology theo thời gian gần như là thứ tự theo bậc lớn nhất của node.
  - ER và BA của seed 0 có cùng tổng thời gian ước tính: node bận nhất của cả hai đồ thị đều có bậc 10.

**Ý cho slide:** công thức thời gian ước tính và các giả định; P17; nói rõ số round (P1) và thời gian (P17) cho hai thứ tự khác nhau.

## 5. Theo dõi từng node; hub so với leaf

**GVHD [00:00], [46:03], [57:31]:** không chỉ xem F1 trung bình; theo dõi từng node. Với mạng đồng nhất, các node có dữ liệu khác nhau hội tụ khác nhau thế nào. Với mạng không đồng nhất, so đường hội tụ của hub (nhiều kết nối) với leaf (ít kết nối).

- Mọi mốc đánh giá (mỗi 100 bước) đều ghi accuracy của từng node: `acc_per_node` trên tập đánh giá chung, `acc_local_per_node` trên phân bố nhãn của chính node. Cuối lượt có thêm F1 của từng node. Phương sai giữa node, độ dao động và mức sụt tính trên từng node (REQ-06, REQ-09).
- **P16** (mới): mỗi ô một topology × một mức dữ liệu. Mỗi node của mỗi seed là một đường mảnh; đường đậm là trung bình của hub (bậc ≥ 6), node giữa (3–5) và leaf (1–2). Lượt thử trên dữ liệu tổng hợp (build/t31c): ở BA và ER, hub lên nhanh nhất và leaf chậm nhất; ring toàn leaf, FC toàn hub. Lượt thử trên MNIST (α = 0.1, ngân sách fast, seed 0, `build/t31g`): ở BA, đường hub nằm trên đường node giữa, đường leaf thấp nhất suốt quá trình; accuracy cuối của hub cao hơn leaf 4.8 điểm ở BA và 4.4 điểm ở ER (1 seed, chưa phải kết quả). Ở ring, cùng là leaf nhưng accuracy cuối của các node vẫn trải từ 82.4% tới 95.7% (khoảng 13 điểm) vì dữ liệu mỗi node khác nhau: đó là trường hợp "mạng đồng nhất, dữ liệu khác nhau" mà GVHD nói.
- Bậc của node lấy từ đồ thị ghi trong manifest (ADR-014). Lượt cũ của lưới lõi không có đồ thị trong manifest, nên bậc được suy từ tải message: mỗi round node gửi và nhận đúng deg(i) message (ADR-015). Vì vậy P16 vẽ được cả cho các lượt ER của E2.
- **P14, P15** (E7): accuracy cuối theo bậc, và đồ thị mạng tô màu node theo accuracy, cỡ node theo tải byte. E7 hỏi riêng chuyện lớp hiếm nằm ở hub hay ở leaf. Lượt thử: node không giữ lớp hiếm đạt 86.2% khi lớp đó ở hub, so với 63.0% khi ở leaf (1 seed, ngân sách 1/4; [thẻ E7](../experiments/E7.md)).

**Ý cho slide:** P16 cạnh P1 (cùng lượt, trung bình và từng node); P15 cho trực quan.

## 6. Chi phí truyền thông và chi phí huấn luyện

**GVHD [52:22]:** công thức `model_size × 2 × số cạnh × số round` của nhóm Vy gộp hai loại chi phí; chi phí huấn luyện như nhau ở mọi node, còn chi phí truyền thông phụ thuộc số kết nối.

Dự án tách hai loại từ đầu:
- **Truyền thông:** mọi message đi qua Transport, và Transport là nơi duy nhất đếm (INV-2). Byte của một message là kích thước tensor thật (SmallCNN: 4 × 46,730 = 186,920 byte). Số đếm có cho cả lượt (message và byte tích lũy, tải lớn nhất trên một node) và cho từng node (gửi + nhận, ADR-013). Tải của node i mỗi round là 2 · deg(i) message, nên **phụ thuộc topology**: N = 20 thì ring 40 message (7.5 MB) mỗi round, lattice 80 (15.0 MB), FC 380 (71.0 MB).
- **Huấn luyện:** τ bước SGD mỗi round, như nhau ở mọi node và mọi topology (τ·R = 4,000 bước cố định, ADR-006). Thời gian huấn luyện đo riêng cho từng node (`train_s_per_node`), **không phụ thuộc topology**.
- P12 vẽ ba loại chi phí của cả lượt (message, byte, thời gian theo pha); P13 vẽ chúng theo từng node.

Với N = 5 như repo của Vy, Nghĩa, FC chỉ có 10 cạnh so với 5 của ring, nên "FC chỉ đắt gấp đôi" [58:16]. Với N = 20 của dự án, FC đắt gấp 9.5 lần ring mỗi round.

## 7. Vì sao F1 của nhóm Vy, Nghĩa không phân biệt được các α, và tuần tự hay song song

**GVHD [55:22]:** F1 ở mọi α đều 96–100%, round hội tụ như nhau; có thể do chạy tuần tự, chạy đa tiến trình sẽ thấy rõ hơn.

- **Chạy tuần tự hay song song không làm đổi kết quả** của mô phỏng đồng bộ: dự án chạy cùng một cấu hình bằng bộ mô phỏng một tiến trình và bằng process runtime (mỗi node một tiến trình qua gloo), và `rounds.jsonl` trùng **từng byte** (test bất biến T-INV-13, ADR-010). Song song chỉ đổi **thời gian** (mục 4), không đổi accuracy.
- F1 không phân biệt được ở repo của Vy, Nghĩa là do thiết kế thí nghiệm: 5 client; mỗi round 5 epoch trên toàn bộ dữ liệu của node (khoảng 90,000 bước mỗi node sau 50 round); α thấp nhất là 0.1 với N = 5 (Nα = 0.5); MNIST dễ. Với ngân sách lớn như vậy, mọi cấu hình đều kịp hội tụ.
- Dự án dùng N = 20, ngân sách cố định 4,000 bước cục bộ cho mọi điều kiện, và các mức lệch tới cực đoan (α = 0.05, #C = 2, #C = 1). Kết quả E0a (validation, η = 0.1, τ = 20) đã cho thấy khác biệt rõ: FC-IID 99.2%, ring-α0.1 94.7%, ring-#C2 78.8% (docs/experiments/E0a.md).
- Ngoài accuracy cuối, dự án so **tốc độ** (R@90, R@95 theo bước, quy ra round, byte và thời gian), AUC của đường học, phương sai giữa node và độ dao động. Hai cấu hình cùng đạt 99% vẫn khác nhau ở các số này.

## 8. Biểu đồ

| Hình | Nội dung | Theo từng round | Theo từng node |
|---|---|---|---|
| P1 | Accuracy trung bình theo bước cục bộ, mỗi ô một topology, màu theo mức Non-IID | có | — |
| P7 | Heatmap node × lớp của từng cách chia | — | có |
| P12 | Message, byte tích lũy, thời gian mỗi round theo pha | có | — |
| P13 | Tải message, byte, thời gian của từng node | — | có |
| P14, P15 | E7: accuracy cuối theo bậc; đồ thị mạng tô màu theo accuracy | — | có |
| P16 | Đường hội tụ của từng node, tô theo hub/giữa/leaf | có | có |
| P17 | Accuracy theo round và theo thời gian ước tính; thời gian để đạt 90% theo băng thông | có | — |

Lệnh: `python -m analysis.plots --results results/ --out figures/`; bản nháp từ `build/`: thêm `--draft` và chọn `--out` trong `build/`. P2–P6 và P8–P11 đang được làm ở nhánh `grid`.

## 9. Thư viện theo module (cho slide [19:11])

| Module | Việc chính | Thư viện, hàm |
|---|---|---|
| `data` | nạp MNIST, CIFAR-10; chia dữ liệu; đo mức Non-IID | torchvision (`datasets.MNIST`, tải CIFAR-10 bản binary rồi đọc bằng `numpy.fromfile`); NumPy (`Generator.dirichlet`, `permutation`, `array_split`) |
| `network` | topology, ma trận trộn W, Transport đếm byte | NetworkX (`cycle_graph`, `grid_2d_graph`, `erdos_renyi_graph`, `barabasi_albert_graph`, `complete_graph`, `star_graph`, `is_connected`); NumPy (W, spectral gap bằng `linalg.eigvalsh`) |
| `learning` | model, huấn luyện cục bộ, số hạng prox | PyTorch (`nn.Conv2d`, `nn.Linear`, autograd); SGD tự viết đúng phép cập nhật của `torch.optim.SGD` |
| `metrics-io` | accuracy, F1, consensus distance; ghi kết quả | NumPy; PyTorch; `json` |
| `sim` | vòng round, mô phỏng một tiến trình, process runtime, sweep, describe | PyTorch (`torch.distributed`, backend gloo); `subprocess`; `argparse`; PyYAML |
| `analysis` | gom kết quả, độ đo suy ra, hình | pandas, NumPy, Matplotlib, NetworkX (`spring_layout` cho P15) |

## 10. Còn mở

- **Nhóm duyệt và chạy:** E7 (9 lượt) và E9 (30 lượt) đã tiền đăng ký, chưa chạy (lưới lớn hơn 10 lượt cần nhóm duyệt). Chúng chạy ở `code_version` khác lưới lõi nên có hình riêng.
- **Accuracy ngay sau huấn luyện cục bộ** (trước tổng hợp) của từng node: là một cách đọc khác câu [00:00] của GVHD, sẽ cho thấy hiệu ứng "pha loãng" khi tổng hợp. Chưa làm: tốn thêm khoảng 30% thời gian mỗi lượt và phải sửa engine (ADR-015, Alternatives). Hỏi lại GVHD nếu cần.
- **Docker:** process runtime đã chạy mỗi node một tiến trình; Docker Compose là tùy chọn (T22c), chưa làm.
- **CIFAR-10:** có code (ADR-012); thí nghiệm vẫn sau M4.
