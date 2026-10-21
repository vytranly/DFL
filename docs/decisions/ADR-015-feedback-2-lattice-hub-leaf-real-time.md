# ADR-015: Theo góp ý buổi 2 — topology lattice, mô tả hub/leaf của đồ thị, đường hội tụ theo node (P16), thời gian thực và thời gian ước tính theo băng thông (P17), E9

## Status
Proposed — soạn ngày 2026-10-21 theo yêu cầu của người dùng: làm hết các mục còn thiếu so với góp ý của GVHD ở buổi báo cáo tiến độ 2 (bản chép lời `transcript.md`, bản tóm tắt `tom_tat_bao_cao_buoi_2.md`), liên tục, không dừng. Người dùng duyệt luôn các việc thuộc Ask first trong yêu cầu đó (ADR, SPEC, topology mới, lưới mới chỉ tiền đăng ký). Chờ nhóm duyệt. Bổ sung ADR-009 (topology), ADR-013 (chi phí theo node), ADR-014 (bậc trong manifest), không thay thế chúng.

## Date
2026-10-21

## Context
Góp ý của GVHD ở buổi 2 (nói với repo của Vy và Nghĩa, nhưng áp dụng cho cả dự án):
- [00:57] Ring và FC là hai đầu thưa nhất và dày nhất, "quá đơn giản"; nên thêm một dạng ở giữa, ví dụ **lưới (lattice) mỗi node 4 kết nối**, và tìm thêm các loại khác.
- [27:18] Đồ thị random phải được mô tả rõ: với khoảng 20 node, **bao nhiêu % là hub (≥ 6 kết nối), bao nhiêu % là leaf (1–2 kết nối)**; từ mô tả đó mới biết cần xác suất p bao nhiêu. Nhóm của Vy chưa giải thích được `safe_p = log(n)/n + 0.15`.
- [43:45] Ngoài số round phải **đo thời gian thực**: mạng thưa có round ngắn nhưng cần nhiều round, mạng dày có round dài nhưng cần ít round, nên chỉ so số round là đánh giá sai.
- [00:00], [46:03] Phải **theo dõi từng node**: với mạng không đồng nhất, so **đường hội tụ của hub với leaf**; với mạng đồng nhất, so đường hội tụ của các node có dữ liệu khác nhau.
- [52:22] Chi phí truyền thông và chi phí huấn luyện phải tách riêng. Dự án đã tách (Transport đếm byte, INV-2; thời gian theo pha và theo node, ADR-013), nên ADR này không đổi gì ở đó.

Hiện trạng trước ADR này:
- Topology có T1 ring, T2 ER, T3 gossip, T3b, T4 FC, T0 star, T6 BA (ADR-014); không có lattice.
- `describe` in spectral gap và message mỗi round của từng topology, không có bậc hay tỉ lệ hub/leaf. p = 0.3 của ER có lý do ở KH §3.3 (spectral gap, chi phí) nhưng chưa nối với mô tả cấu trúc.
- `rounds.jsonl` có `acc_per_node` ở mọi mốc; chưa có hình đường hội tụ theo node. P14, P15 (ADR-014) chỉ là mốc cuối và chỉ cho lượt `rare_classes`.
- `timing.jsonl` có thời gian theo pha và `train_s_per_node` (ADR-013); chưa có độ đo "thời gian để đạt ngưỡng". Ở runtime inprocess các node chạy lần lượt, nên thời gian đo là tổng qua node, không phải thời gian của một hệ phân tán. Ở process runtime giao tiếp đi qua gloo trên một máy, nên không có giới hạn băng thông của mạng thật (ADR-010).
- Lượt của lưới lõi (nhánh `grid`) không có `graph_edges` (trước ADR-014), nhưng có `node_messages_cum` (ADR-013).

Đo ngày 2026-10-21 (N = 20, 2,000 đồ thị mỗi dòng, seed từ luồng GRAPH, networkx 3.7; hub = bậc ≥ 6, leaf = bậc 1–2):

| Đồ thị | P(liên thông) khi chưa lấy mẫu lại | Bậc TB (min–max) | % hub | % leaf | Spectral gap (MH) |
|---|---|---|---|---|---|
| ring | 1 | 2 | 0% | 100% | 0.033 |
| lattice 4×5 (xuyến) | 1 | 4 | 0% | 0% | 0.276 |
| ER p = 0.10 | 0.056 | 2.5 (1–8) | 2% | 55% | — |
| ER p = 0.15 (≈ ln N / N) | 0.402 | 3.2 (1–10) | 7% | 37% | — |
| ER p = 0.20 | 0.757 | 3.9 (1–12) | 17% | 21% | — |
| ER p = 0.25 | 0.927 | 4.8 (1–13) | 34% | 10% | — |
| ER p = 0.30 (lưới lõi) | 0.984 | 5.7 (1–14) | 53% | 4% | TB 0.233 |
| ER p = 0.40 | 1.000 | 7.6 (1–16) | 84% | 1% | — |
| BA m = 1 (cây) | 1 | 1.9 (1–19) | 5% | 81% | 0.016 |
| BA m = 2 (E7) | 1 | 3.6 (1–18) | 16% | 46% | 0.096 |
| BA m = 3 | 1 | 5.1 (1–19) | 32% | 2% | 0.151 |
| FC | 1 | 19 | 100% | 0% | 1.000 |

- Cột "P(liên thông)" khớp xấp xỉ Poisson của số node cô lập, P ≈ exp(−N(1 − p)^{N−1}) (0.067, 0.402, 0.750, 0.919, 0.977 ở p = 0.10 … 0.30). Ngưỡng liên thông của G(N, p) là ln N / N = 0.150 với N = 20 (Erdős–Rényi 1960). Công thức `safe_p = ln N / N + 0.15` của nhóm Vy cho đúng 0.30 ở N = 20, tức p của lưới lõi; ở N = 5 nó cho 0.47.
- ER p = 0.3 vì vậy là mạng "dày vừa": khoảng một nửa số node là hub, gần như không có leaf. Muốn có nhiều leaf phải hạ p xuống dưới ngưỡng liên thông, khi đó phần lớn đồ thị rút ra không liên thông và phải lấy mẫu lại nhiều lần, nên phân bố bị lệch về các đồ thị "may mắn". BA m = 2 cho cả hub (16%) lẫn leaf (46%) mà luôn liên thông, nên là đồ thị không đồng nhất đúng nghĩa cho câu hỏi hub/leaf.
- Lattice 4×5 xuyến có spectral gap 0.276, cao hơn ER p = 0.3 trung bình: với N = 20, xuyến 4×5 có đường kính chỉ 4. Lattice là mạng đồng nhất bậc 4, nằm giữa ring (bậc 2) và FC (bậc 19).

## Decision
- **Topology T7 `lattice`:** lưới 2-D hai đầu nối vòng (xuyến), mỗi node đúng 4 kết nối.
  - Kích thước a × b = N với a là ước lớn nhất của N không vượt √N, và cần a ≥ 3 (với a = 2, nối vòng tạo cạnh trùng nên bậc chỉ còn 3). N = 20 cho 4 × 5; N = 9 cho 3 × 3; N = 8 bị từ chối.
  - Dựng bằng `networkx.grid_2d_graph(a, b, periodic=True)` rồi `convert_node_labels_to_integers(ordering="sorted")`, nên node (r, c) có id r·b + c. Không có tham số cấu hình, không có ngẫu nhiên; vẫn qua `_check_connected` (INV-4) và ghi `graph_edges` (ADR-014).
- **`describe` mô tả bậc của mọi đồ thị tĩnh:** mỗi hàng có bậc TB, min, max, % hub (bậc ≥ 6) và % leaf (bậc 1–2) theo định nghĩa của GVHD; ER và BA lấy trung bình qua seed. Thêm hàng `T6 BA` (m của cấu hình hay 2) và `T7 lattice`. Thêm bảng **ER theo p** (p ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}): tần suất liên thông khi chưa lấy mẫu lại, xấp xỉ exp(−N(1 − p)^{N−1}), bậc và tỉ lệ hub/leaf của đồ thị liên thông. Bảng **BA theo m** (m ∈ {1, 2, 3}).
- **Analysis đọc bậc của lượt cũ từ tải message:** với topology tĩnh, mỗi round node i gửi và nhận đúng deg(i) message, nên deg(i) = `node_messages_cum`[i] / (2 · round). `centrality.degrees` dùng `graph_edges` khi có, không thì dùng công thức này (chỉ với topology tĩnh, và kiểm kết quả là số nguyên). Nhờ vậy lượt ER của lưới lõi vẫn tách được hub/leaf mà analysis không phải import `dfl.network`.
- **P16 · đường hội tụ theo node:** mỗi ô một topology tĩnh (sắp theo bậc trung bình), hàng là mức dữ liệu (tối đa hai mức: ít lệch nhất và lệch nhất theo JS tính từ `node_class_counts`). Mỗi node của mỗi seed là một đường mảnh `acc_per_node` theo bước cục bộ, tô theo loại bậc: hub (≥ 6), giữa (3–5), leaf (1–2); đường đậm là trung bình của mỗi loại, kèm kiểu nét riêng.
- **P17 · số round so với thời gian:** cùng một mức dữ liệu (mức có nhiều topology nhất, hòa thì lệch hơn), mỗi màu một topology:
  - (a) `acc_mean` theo round;
  - (b) `acc_mean` theo **thời gian phân tán ước tính** ở băng thông tham chiếu B = 10 Mbit/s;
  - (c) thời gian ước tính để đạt 90% theo B ∈ [1, 1,000] Mbit/s (trục log), để thấy topology nào nhanh hơn ở băng thông nào.
- **Thời gian phân tán ước tính** (mô hình, ghi rõ trên hình): mỗi khoảng giữa hai mốc đánh giá tốn max_i(thời gian huấn luyện của node i) + max_i(byte gửi + nhận của node i) / B. Thời gian huấn luyện là `train_s_per_node` đo được; byte là `node_bytes_cum` đếm bởi Transport, kể cả server ảo của đường server. Giả định: các node huấn luyện song song, gửi và nhận dùng chung băng thông B, bỏ qua độ trễ và thời gian tổng hợp (cộng vector, nhỏ so với huấn luyện, xem P12).
- **Thời gian để đạt ngưỡng** trong bảng của `analysis.aggregate`: `s90_est`, `s95_est` (ước tính ở 10 Mbit/s) và `s90_measured`, `s95_measured` (thời gian đo của các pha train, comm, combine, không tính đánh giá). Nội suy tuyến tính tại R@x như bytes@x.
- **E9 · topology trung gian và không đồng nhất** (`docs/experiments/E9.md`, `configs/experiments/e9_topology_shapes.yaml`): {ring, lattice, ER p = 0.3, BA m = 2, FC} × {IID, α = 0.1} × seed {0, 1, 2} = **30 lượt**, khoảng 9.6 giờ-lõi, profile `full`, η = 0.1.
  - H10 (phép so sánh chính): trên BA m = 2 ở α = 0.1, accuracy cuối trung bình của node hub (bậc ≥ 6) cao hơn của node leaf (bậc 1–2), hiệu số tính trong từng lượt rồi ghép theo seed. 3 seed nên tối đa "gợi ý".
  - Mọi so sánh khác (lattice so với ring và FC, ER, thời gian, P16, P17) chỉ để khám phá.
  - Chỉ tiền đăng ký; chạy khi nhóm duyệt (lưới lớn hơn 10 lượt).

## Alternatives Considered

### Lattice không nối vòng
- Ưu: giống hình ảnh "lưới" trực quan.
- Nhược: bậc 2 (góc), 3 (cạnh) và 4 (trong), nên không còn là mạng đồng nhất; góc lưới lại thành leaf.
- Kết luận: không chọn; GVHD nói "mỗi node có 4 kết nối".

### Watts–Strogatz (small-world) hay random k-regular thay cho lattice
- Ưu: có tham số nối từ lattice tới random.
- Nhược: thêm một tham số và một nguồn ngẫu nhiên; GVHD gợi ý lattice. Random k-regular vẫn là phần mở rộng T5 của SPEC §1.4.
- Kết luận: để sau; bảng ER theo p và BA theo m đã cho dải từ thưa tới dày, từ đồng nhất tới không đồng nhất.

### Đo thời gian thật bằng process runtime với băng thông bị giới hạn
- Ưu: số đo thật, không phải mô hình.
- Nhược: gloo trên một máy không giới hạn được băng thông nếu không dùng công cụ của hệ điều hành (tc/netem, cần quyền root); 20 tiến trình trên 8 lõi làm méo thời gian huấn luyện (docs/PERF.md). SPEC §1.4 xếp giới hạn băng thông vào mở rộng của E8 (Q11).
- Kết luận: dùng mô hình từ số đo thật (thời gian huấn luyện của từng node, byte của từng node), ghi rõ giả định; process runtime vẫn đo `comm_s` thật cho E8.

### Đo accuracy của node ngay sau huấn luyện cục bộ (trước tổng hợp)
- Ưu: thấy được hiệu ứng "pha loãng" khi tổng hợp, một cách đọc khác của câu [00:00] của GVHD.
- Nhược: đánh giá thêm N model ở mỗi mốc (khoảng +30% thời gian một lượt), sửa engine, observer và process runtime (T-INV-13).
- Kết luận: không làm trong ADR này; P16 theo dõi từng node sau tổng hợp ở mọi mốc. Ghi vào câu hỏi mở cho nhóm.

### Dựng lại đồ thị của lượt cũ trong analysis
- Nhược: analysis chỉ được import `dfl.contracts` (ADR-014). Bậc suy từ tải message là chính xác với topology tĩnh và không cần dựng lại.
- Kết luận: không chọn.

## Consequences
- **`code_version` đổi** (topology mới, `describe` mới). Nhánh `grid` không nhận các commit `src/` này, như ADR-013, ADR-014. Phần analysis (bậc từ message, P16, P17, s@x) không đổi `code_version` và chạy được trên lượt của lưới lõi.
- **Cấu hình chuẩn hóa không thêm khóa:** `lattice` chỉ là một giá trị mới của `topology.kind`, nên `run_id` của cấu hình cũ không đổi.
- **SPEC cần sửa:** §1.4 (lattice có code), §5.2 (lưới E9), REQ-04 (thời gian để đạt ngưỡng), REQ-07 (T7), REQ-10 (P16, P17), US-4 (`describe` in bậc và tỉ lệ hub/leaf).
- **Tasks:** T31a (ADR-015, SPEC), T31b (thời gian ước tính, s@x, P17), T31c (bậc từ message, P16), T31d (lattice), T31e (`describe` mô tả bậc, ER theo p, BA theo m), T31f (demo chia dữ liệu từng bước), T31g (thẻ E9, lưới, lượt thử, tài liệu phản hồi góp ý).
- Hạn chế cần nêu: thời gian phân tán là ước tính từ một mô hình băng thông đơn giản; ở runtime inprocess, thời gian đo là tổng qua node.

## Nguồn
- Erdős P., Rényi A. (1960). *On the evolution of random graphs*. Publ. Math. Inst. Hung. Acad. Sci. 5, 17–61 (ngưỡng liên thông ln N / N).
- networkx.grid_2d_graph (periodic=True nối hai đầu mỗi chiều): https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.lattice.grid_2d_graph.html
- networkx.convert_node_labels_to_integers: https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.relabel.convert_node_labels_to_integers.html
- networkx.barabasi_albert_graph, networkx.erdos_renyi_graph: xem ADR-014, ADR-009.
