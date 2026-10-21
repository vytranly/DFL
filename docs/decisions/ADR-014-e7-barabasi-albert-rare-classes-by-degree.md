# ADR-014: E7 — đồ thị Barabási–Albert (m = 2), lớp 5–9 chỉ ở K = 2 node đặt theo bậc (hub, biên, ngẫu nhiên), đồ thị ghi vào manifest

## Status
Proposed — soạn ngày 2026-10-21 theo yêu cầu của người dùng ("làm đầy đủ" E7, nhánh `binh`). Đây là phần mở rộng (SPEC §1.4, QĐ-4), thuộc Ask first; bốn lựa chọn được hỏi và chốt trong phiên ngày 2026-10-21: K = 2 node giữ lớp hiếm, lưới 9 lượt, chỉ làm code và tiền đăng ký (chưa chạy lưới), commit thẳng lên `binh`. Chờ nhóm duyệt. Bổ sung ADR-005 (partition, vị trí node), ADR-008 (manifest) và ADR-009 (topology), không thay thế chúng.

## Date
2026-10-21

## Context
- KH §3.9 xếp E7 ("vị trí dữ liệu theo độ trung tâm", theo Palmieri et al. 2024) là mở rộng Tier 3, khoảng 12 lượt. Người dùng hỏi repo có so được node hub với node biên và có vẽ được node mạnh không. Trả lời ngày 2026-10-21:
  - accuracy và tải giao tiếp của từng node đã được ghi (`acc_per_node`, `final_acc_per_node`, `node_bytes_cum`; ADR-013);
  - nhưng topology của lưới gần như không có hub: ring và FC đều bậc, gossip đổi cặp mỗi round, ER p = 0.3 với N = 20 chỉ lệch bậc vừa phải (ví dụ 3 → 11);
  - analysis không biết đồ thị: manifest chỉ ghi spectral gap, và analysis chỉ được import `dfl.contracts` (import-linter `analysis-reads-only`), nên không dựng lại được đồ thị từ cấu hình.
- Palmieri 2024 (DIGEST §2.3; arXiv 2402.18606v1):
  - MNIST, 100 node, BA m ∈ {2, 5, 10}, ER-idem, SBM [p.9–10];
  - lớp 0–4 chia đều cho mọi node, lớp 5–9 chỉ giao cho 10% node có degree, betweenness hay clustering cao nhất ("hub") hoặc thấp nhất ("edge") [p.10, p.12];
  - ER-idem, m = 2: accuracy nhóm không giữ lớp 5–9 khoảng 0.87 khi lớp đó ở hub, khoảng 0.67 khi ở biên (đọc từ Fig. 8) [p.16]; "The gap decreases as we increase the overall connectivity" [p.17];
  - DecAvg có trọng số theo kích thước, MLP, SGD momentum 0.5, 100 epoch cục bộ [p.7, p.14]. Dự án giữ vòng ATC, SmallCNN, SGD thuần và Metropolis–Hastings (ADR-004, ADR-006), nên chỉ so được chiều hiệu ứng, không so được con số.
- Đo trên networkx 3.7 (`barabasi_albert_graph(20, 2, seed)`, seed 0…4): luôn 36 cạnh và liên thông; bậc lớn nhất 9–12, nhì 8–10; bậc nhỏ nhất 1–2, và 7–10 node có bậc 2. Hub và biên vì vậy khác nhau rõ, nhưng bậc nhỏ nhất thường hòa nhau.
- Ràng buộc phải giữ: G9 (cùng seed thì cùng partition, cùng đồ thị, cùng w⁰ ở mọi ô), F7 (mọi nguồn ngẫu nhiên đi qua Generator của `seeding.py`), INV-4 (đồ thị tĩnh liên thông), INV-6, T-INV-13; lượt cũ phải giữ nguyên `rounds.jsonl` từng byte.

## Decision
- **Topology T6 `barabasi_albert`:**
  - Cấu hình: `topology.kind: barabasi_albert`, `topology.m` (mặc định 2; để null với topology khác, như `p` của ER).
  - Dựng bằng `networkx.barabasi_albert_graph(N, m, seed=s)`, với s là một số nguyên rút từ luồng GRAPH của lượt, như T2. Đồ thị tĩnh, nên vẫn qua `_check_connected` (INV-4) và có spectral gap trong manifest.
  - Cần 1 ≤ m < N.
- **Cách chia P5 `rare_classes`** (tên theo KH §3.9: "lớp hiếm" là lớp chỉ vài node có, không phải lớp ít mẫu):
  - Cấu hình: `partition.scheme: rare_classes`, `partition.holders: K` (bắt buộc với scheme này, 1 ≤ K < N).
  - Lớp 0–4 (lớp chung): xáo bằng luồng PARTITION rồi `array_split` đều cho N phần.
  - Lớp 5–9 (lớp hiếm): xáo bằng cùng luồng rồi `array_split` đều cho K phần đầu.
  - Kết quả: phần 0…K−1 là phần giữ lớp hiếm, theo đúng thứ tự đó. Với MNIST (55k ảnh của node), K = 2, N = 20: phần thường khoảng 1,400 ảnh, phần giữ lớp hiếm khoảng 1,400 + 13,500 ảnh. Không lấy mẫu lại; vẫn kiểm `min_size`.
- **Vị trí node** (`partition.placement`):
  - `random` (mặc định): hoán vị ngẫu nhiên hiện có (`assign_positions`, luồng PLACEMENT). Lượt cũ không đổi gì.
  - `hub` và `edge`: chỉ với `rare_classes` và topology có bậc khác nhau: `barabasi_albert` hay `erdos_renyi`.
    - Rút `perm = rng.permutation(N)` từ luồng PLACEMENT làm khóa phá hòa.
    - Xếp node theo bậc giảm dần (`hub`) hay tăng dần (`edge`); bậc bằng nhau thì xếp theo thứ hạng trong `perm`.
    - K node đầu nhận K phần giữ lớp hiếm. N − K phần còn lại đi vào các node còn lại, theo thứ hạng trong `perm`.
  - Vì phần thường đều là IID của lớp 0–4, cách gán chúng không quan trọng; dùng `perm` để quy tắc tất định và không phụ thuộc thứ tự node.
  - Hệ quả của G9: cùng seed thì ba ô hub, edge, random có cùng partition, cùng đồ thị, cùng w⁰; chỉ khác chỗ đặt các phần.
  - Muốn biết bậc thì phải có lịch topology trước khi gán vị trí. `assemble` dựng lịch trước bước gán vị trí. Hai bước lấy ngẫu nhiên từ hai luồng riêng (GRAPH, PLACEMENT), nên đổi thứ tự không đổi số nào của lượt cũ.
- **Manifest thêm `graph_edges`:** danh sách cạnh `[u, v]` với u < v, sắp tăng, của đồ thị tĩnh (ring, FC, ER, BA, star; `local_only` là `[]`); null với gossip, ER-gossip, đường server, central và manifest của attempt chưa lắp xong. Analysis tính bậc từ đây, nên không phải import `dfl.network`.
- **Hai hình mới ở REQ-10** (lấy lượt của E7):
  - **P14 · accuracy cuối theo bậc:** mỗi vị trí (hub, edge, random) một ô; mỗi điểm là một node của một seed, trục x là bậc, trục y là `final_acc_per_node`; node giữ lớp hiếm vẽ khác dấu.
  - **P15 · đồ thị mạng:** seed 0 của mỗi vị trí; vị trí node theo bố cục cố định (spring layout, seed 0) của đồ thị trong manifest; màu node là accuracy cuối, kích thước là tải byte cuối (`node_bytes_cum`), node giữ lớp hiếm có viền.
- **H9 và phép so sánh chính** (đăng ký trong `docs/experiments/E7.md`):
  - Độ đo: trung bình `final_acc_per_node` trên đủ tập test của N − K node **không** giữ lớp hiếm. Node đó chỉ học được lớp 5–9 qua láng giềng, nên độ đo này cho thấy tri thức lan từ node giữ lớp hiếm ra mạng.
  - H9: đặt lớp hiếm ở hub cho độ đo cao hơn đặt ở biên. Hiệu số hub − edge ghép cặp theo seed.
  - 3 seed nên tối đa là "gợi ý" (ADR-007): chiều kỳ vọng đúng ở cả 3 seed và CI 95% (t, df = 2) không chứa 0.
  - Ô `random` và mọi số khác (accuracy của node giữ lớp hiếm, accuracy theo bậc, consensus distance, tải byte của hub) chỉ để khám phá.
- **Lưới E7** (`configs/experiments/e7_centrality.yaml`): profile `full`, N = 20, τ = 20, η = 0.1 (ADR-007), μ = 0, MH; BA m = 2 × `rare_classes` K = 2 × {hub, edge, random} × seed {0, 1, 2} = **9 lượt**, khoảng 3 giờ-lõi (docs/PERF.md §5: 19.2 phút-lõi một lượt τ = 20). Lưới chạy sau khi nhóm duyệt ADR này và thẻ E7.

## Alternatives Considered

### Chỉ phân tích ER của lưới hiện có (mức rẻ)
- Ưu: không đổi `src/`; dùng lại lượt E2.
- Nhược: bậc của ER p = 0.3 chỉ lệch vừa phải, và dữ liệu không được đặt theo bậc, nên không trả lời được câu hỏi hub hay biên. Vẫn cần ghi đồ thị để analysis biết bậc.
- Kết luận: không đủ; P14 vẫn vẽ được cho lượt ER sau khi có `graph_edges`.

### Star (T0) thay cho BA
- Ưu: có sẵn; hub rõ nhất.
- Nhược: chỉ một hub, N − 1 lá cùng bậc 1, nên "biên" không phân biệt được, và K = 2 không đặt được hai node ở hub. Đồ thị cũng không giống mạng thật như BA.
- Kết luận: không chọn.

### Thêm ER-idem như Palmieri (18 lượt)
- Ưu: tách ảnh hưởng của phân bố bậc lệch với số cạnh.
- Nhược: gấp đôi lưới, thêm một topology nữa; người dùng chọn 9 lượt.
- Kết luận: để sau, nếu H9 có kết quả "gợi ý".

### K = 4 node giữ lớp hiếm (20%)
- Ưu: nhóm giữ lớp hiếm lớn hơn, số theo nhóm ổn định hơn.
- Nhược: top-4 và bottom-4 của BA N = 20 gần nhau hơn, hiệu ứng nhỏ hơn; khác Palmieri (10%).
- Kết luận: người dùng chọn K = 2.

### Chọn hub theo betweenness
- Ưu: Palmieri dùng cả degree lẫn betweenness.
- Nhược: với BA, hai thước đo gần trùng nhau ở top và bottom; degree dễ giải thích và dễ kiểm.
- Kết luận: dùng degree; betweenness có thể tính trong analysis để khám phá.

### Dựng lại đồ thị trong analysis từ cấu hình
- Ưu: không đổi manifest.
- Nhược: analysis phải import `dfl.network` và `dfl.seeding`, trái contract `analysis-reads-only`; sửa code topology sau này sẽ làm hình sai mà không ai biết.
- Kết luận: không chọn; ghi cạnh vào manifest.

### Trọng số theo kích thước (DecAvg của Palmieri)
- Ưu: giống bài gốc; node giữ lớp hiếm (lớn hơn khoảng 10 lần) có trọng số lớn.
- Nhược: `size_weighted` chỉ dùng với FC hay server (ADR-005); đổi nó là một biến nữa của E7.
- Kết luận: giữ MH như mọi ô DFL khác; nêu là khác biệt với Palmieri.

## Consequences
- **Cấu hình chuẩn hóa thêm ba khóa** (`topology.m`, `partition.holders`, `partition.placement`), nên `run_id` của mọi cấu hình đổi, như khi ADR-012 thêm `augmentation`. `code_version` cũng đổi. Huấn luyện của lượt cũ không đổi: `rounds.jsonl` của smoke và fast phải trùng từng byte trước và sau (kiểm ở T30e).
- **Lưới và E7:** nhánh `grid` không nhận các commit này, như ADR-013. E7 chỉ so các ô trong chính nó (mọi ô cùng một `code_version`), nên không trộn với Tier 1 (INT-2).
- **Schema:** `ManifestRecord` thêm `graph_edges`; SPEC §5.3 sửa theo.
- **SPEC cần sửa:** §1.3 (H9), §1.4 (E7 có code trước M4), §5.2 (khóa cấu hình, lưới E7), §5.3 (manifest), REQ-07 (T6), REQ-10 (P14, P15).
- **Process runtime** dùng chung `assemble`, nên nhận BA và vị trí theo bậc mà không phải sửa; T-INV-13 phải xanh trên một ô E7 nhỏ.
- **Tasks:** T30a (ADR-014, SPEC), T30b (topology BA), T30c (`rare_classes`, vị trí theo bậc, `graph_edges`), T30d (analysis, P14, P15), T30e (thẻ E7, lưới, chạy thử, so byte lượt cũ).
- Hạn chế cần nêu: N = 20 nhỏ hơn 100 của Palmieri; K = 2 nên mỗi nhóm giữ lớp hiếm chỉ có 2 node; 3 seed nên tối đa "gợi ý"; MH thay vì DecAvg.

## Nguồn
- Palmieri L., Boldrini C., Valerio L., Passarella A., Conti M. (2024). *Impact of network topology on the performance of Decentralized Federated Learning*. https://arxiv.org/abs/2402.18606 (tóm tắt ở docs/research/DIGEST.md §2.3)
- networkx.barabasi_albert_graph: https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html
- numpy.random.Generator.permutation: https://numpy.org/doc/2.5/reference/random/generated/numpy.random.Generator.permutation.html
