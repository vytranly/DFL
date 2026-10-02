# Rà soát kiểm thử ở checkpoint C3 (P3.7)

- **Ngày:** 2026-10-02
- **Phạm vi:** test của `src/dfl`, ưu tiên engine, mixing/topology, partition, metrics. Đối chiếu với T-INV-1…12 (KH §5.2, SPEC §9.2) và CONSTRAINTS.md, theo checklist `testing-patterns.md`.
- **Cách làm:**
  - Một subagent test-engineer rà soát ở chế độ chỉ đọc.
  - Coverage dòng và nhánh được đo trên toàn bộ 611 test.
  - Mutant được cài trong bộ nhớ qua plugin pytest, không chép hay sửa `src/`.
  - Tiến trình chính tự chạy lại để xác minh các khẳng định quan trọng nhất (§6).
- **Không sửa code.** Các test dưới đây là đề xuất; test nào ghi "FAIL" là test Prove-It cho một lỗi, phải viết trước và xác nhận đỏ trước khi sửa (P3.4).

## Kết luận nhanh

- **1 lỗi thật (P0), đã tái hiện:** process runtime không kiểm INV-4 "hợp các round của gossip liên thông".
  - Cùng một cấu hình, runtime inprocess báo `ValueError`, còn runtime processes ghi `ok`. T-INV-13 vì thế bị lệch ở những cấu hình này.
  - Lưới N = 20 chạy inprocess nên không bị ảnh hưởng. E8 và demo chạy processes thì có.
- **3 khoảng trống mức P0**, tức có thể làm sai kết luận mà không test nào báo:
  - dây seed của 5 luồng ngẫu nhiên trong `sim/run.py`;
  - các trường tổng hợp F1, acc^loc, acc^eff của RoundRecord không được kiểm giá trị;
  - thứ tự P_i (`node_class_counts`) theo id node không được kiểm.
- **1 lỗi tiềm ẩn (P2):** `mixing.uniform` đọc thuộc tính cạnh `weight`. Có thêm 3 lỗi nhẹ: thiếu kiểm đầu vào, hoặc thông báo lỗi nêu sai nguyên nhân.
- **Tổng:** P0 = 4, P1 = 8, P2 = 14; 15 test chờ T15b vì `analysis/convergence.py` chưa có.
- **Mutant:** 18 mutant mới sống sót qua các test hiện có. Mỗi mutant đều bị một test đề xuất giết (đã chạy dưới dạng script).

## 1. Số liệu

| Module | Dòng + nhánh | Ghi chú |
|---|---|---|
| Tổng | 94.6% | 611 test |
| `sim/engine.py`, `sim/node.py`, `sim/observer.py`, `metrics/*`, `network/topology.py`, `network/transport.py`, `sim/sweep.py`, `io/manifest.py`, `config.py` | 100% | Coverage cao không có nghĩa ca biên đã được kiểm (§4) |
| `data/partition.py` | 99% | Thiếu dòng 91, nhánh "mọi xác suất còn lại bằng 0" (P1-3) |
| `network/mixing.py` | 95% | Thiếu dòng 71 (kiểm sizes) và 90 (spectral gap với 1 node) (P2-2, P2-4) |
| `sim/process_runtime.py`, `sim/torchdist.py` | 65%, 61% | Thân node chạy trong tiến trình con, coverage không đo; câu hỏi mở ở SESSION_LOG |

## 2. Độ trung thành của T-INV-1…12

| T-INV | Test hiện có | Đủ định nghĩa? | Khoảng trống |
|---|---|---|---|
| 1 | `test_engine_consensus.py::test_t_inv_1_*` (3 test), `test_topology.py` (E[W], N = 20) | Đủ | — (chỉ chạy N = 4 qua engine) |
| 2 | `test_engine_core.py::test_t_inv_2_identity_mixing_matches_local_only_training` | Đủ | — |
| 3 | `test_engine_fedavg.py::test_t_inv_3_*` | Đủ | Kích thước chỉ lệch vừa; ca lệch mạnh ở P2-8 |
| 4 | `test_engine_core.py::test_t_inv_4_node_visiting_order_does_not_change_result[ring, er]` | **Một phần** | Không chạy đường server (`_exchange_via_server` là code riêng): mutant "server cộng theo thứ tự duyệt" sống (P1-1) |
| 5 | `test_seeds.py::test_t_inv_5_*` (6 test), fixture RNG toàn cục | **Một phần** | Các test "đúng luồng" đều dùng seed 0, và test "khác seed thì khác cách chia" qua được chỉ nhờ hoán vị vị trí node (P0-2) |
| 6 | `test_engine_messages.py::test_t_inv_6_*` (5 ca), `test_process_runtime.py`, `test_smoke_run.py` | Đủ | Với N lẻ chỉ kiểm số message, chưa kiểm byte (P2-7) |
| 7 | `test_local_train.py::test_t_inv_7_*` (4 test) | Đủ | — |
| 8 | `test_partition.py::test_t_inv_8_*` (7 test) | **Gần đủ** | Chưa chạm biên đúng min_size; nhánh xác suất bằng 0 chưa chạy; `retries` của quantity chưa được so (P1-3…P1-5) |
| 9 | `test_mixing.py::test_t_inv_9_*` | Đủ | FC với N = 20 cho 0.9999999999999998, nên mọi phép so phải dùng sai số (đã đúng) |
| 10 | `test_classification.py::test_t_inv_10_*` | Đủ | — (nhưng các trường tổng hợp dựng từ các độ đo này thì chưa được kiểm: P0-3) |
| 11 | `test_observer.py::test_t_inv_11_*` | **Một phần** | Chỉ chạy N = 4, nơi 1/4 biểu diễn đúng ở hệ nhị phân. Consensus chỉ được kiểm ≤ 1e-6·drift, dù đúng ra phải bằng 0.0. Mutant "cộng bằng float64 rồi ép về float32" sống (P1-2) |
| 12 | `test_observer.py::test_t_inv_12_*` | Đủ | Tính thuần của lịch T3b chưa có unit test (P2-11) |

## 3. Các ca biên mà P3.7 yêu cầu

| Ca | Trạng thái |
|---|---|
| N = 2 | Ring bị từ chối theo thiết kế (topology.py:119, đã có test). Nhưng config chỉ đòi N ≥ 1, nên ring N = 2 chỉ lỗi lúc lắp và để lại attempt failed (P2-6, cần nhóm quyết). FC, gossip, ER, star với N = 2, và Ξ của 2 node: chưa test. Probe cho đúng W = ½J và drift = ‖w₁ − w₂‖/2 (P2-5) |
| Node có đúng 64 mẫu | Chưa test ở cả phía nhận (`>=` ở partition.py:97, :157) lẫn phía từ chối (`<` ở partition.py:41); mutant `>=` → `>` sống (P1-4) |
| α rất nhỏ | Nhánh "mọi xác suất bằng 0" (partition.py:90-91) chưa từng chạy (P1-3). Trên MNIST, N = 20: α = 0.01 cạn số lần lấy mẫu lại ở 2/3 seed, α ≤ 1e-3 cạn ở cả 3 seed. Ở α = 1e-4, 784/1,001 lần thử thất bại vì xác suất bằng 0, nhưng thông báo lỗi chỉ nói về min_size (P2-3). numpy 2.x không sinh NaN ở α ≥ 1e-6 |
| Gossip với N lẻ | Đã test: mỗi round đúng một node đứng ngoài, có W_ii = 1 và 4 message/round với N = 5. Chưa test: byte và tải với N lẻ, E[W] với N lẻ, phép kiểm hợp các round tường minh (P2-7) |
| R@x không đạt | Chờ T15b (§5) |
| FC suy biến | Spectral gap ≈ 1 và pstdev(acc) = 0 đã test. Chưa test: consensus đúng bằng 0.0 ở N = 20, FC với N = 2, size-weighted với kích thước lệch mạnh (P1-2, P2-5, P2-8) |

## 4. Khoảng trống và test Prove-It đề xuất

Mỗi mục ghi tên test, file, phác thảo Arrange → Act → Assert, và kỳ vọng trên code hiện tại.

### P0

**P0-1: NGHI LỖI THẬT, đã tái hiện. Process runtime không kiểm INV-4 "hợp các round liên thông".**
- *Nguyên nhân:* `check_union_connected` chỉ được gọi trong `Engine.check_schedule` (engine.py:358, qua `run_rounds`). `ProcessRun.execute` và `run_node` (`sim/process_runtime.py`) không gọi hàm này. Docstring ở engine.py ghi process runtime phải dùng lại nó.
- *Tái hiện:* smoke rút gọn (2 round), gossip, seed 1, nơi round 0 và round 1 dùng cùng một matching trên K₄:
  ```
  inprocess -> raised ValueError : hợp các round 0…1 của lịch gossip không liên thông … (INV-4)
  processes -> ok
  ```
- *Test:* `test_t_inv_13_processes_refuse_a_gossip_schedule_whose_union_never_connects` (`tests/integration/test_process_runtime.py`).
  - Cùng cấu hình chạy ở cả hai runtime.
  - Cả hai phải raise `ValueError` khớp "hợp các round" và ghi SummaryFailed với cùng lý do.
  - **Kỳ vọng: FAIL.**
- *Sửa đề xuất:* gọi `check_union_connected(schedule, N, rounds)` trong `ProcessRun` trước khi khởi chạy tiến trình con. Tùy chọn gọi thêm ở mỗi node.
- *Ảnh hưởng:* lưới N = 20 chạy inprocess nên không bị. Với R lớn (E8: R = 50, N = 8) xác suất rất nhỏ; với smoke gossip (N = 4, R = 6) khoảng 0.4% số seed.

**P0-2: dây seed của các luồng trong `sim/run.py` không được test với seed khác 0.**
- *Bằng chứng:*
  - Các mutant ghim seed 0 cho luồng PARTITION, PLACEMENT, BATCH, GRAPH, GOSSIP hay GOSSIP_ER đều qua 28–34 test của test_seeds, test_baselines, test_describe. Tiến trình chính đã chạy lại PARTITION và BATCH: 28 passed.
  - Chỉ luồng INIT bị bắt.
  - Test "khác seed thì khác cách chia" (test_seeds.py:58) chỉ đòi các phần nằm ở vị trí khác nhau, nên riêng hoán vị vị trí node đã đủ cho nó qua.
- *Vì sao P0:* nếu lỗi xảy ra, các seed dùng chung cách chia, đồ thị và dãy minibatch, nên khoảng tin cậy hẹp giả và kết luận bị phóng đại (SPEC §6).
- *Test:* `test_t_inv_5_every_seeded_stream_of_a_run_follows_config_seed[seed=0, seed=3]` (`test_seeds.py`).
  - Dựng giá trị kỳ vọng độc lập từ chính seed s: cách chia và vị trí node (từ `numpy_rng(s, PARTITION)` và `numpy_rng(s, PLACEMENT)`), Generator batch của từng node, cạnh ER, matching gossip và ER-gossip của 8 round đầu, Generator B1_BATCH.
  - So với `run.build` và `run.assemble(...).schedule`.
  - Kèm: sửa test_seeds.py:58 để so multiset các phần (`sorted(tuple(sorted(p)) for p in parts)`) giữa seed 0 và seed 1.
  - **Kỳ vọng: PASS**, và giết cả 5 mutant.

**P0-3: các trường tổng hợp của RoundRecord không được kiểm giá trị** (observer.py:66-77).
- *Bằng chứng:* không test nào kiểm `f1_macro_mean`, `f1_macro_min`, `acc_eff_mean`, `acc_local_min`. Chỉ có một test std = 0 khi mọi node bằng nhau. Bốn mutant sau đều qua 100/100 test liên quan:
  - `acc_local_min` dùng max;
  - `f1_macro_mean` lấy trung bình của acc;
  - `acc_eff_mean` lấy trung bình của acc_local;
  - `f1_macro_std` tính trên acc.
- *Vì sao P0:* rounds.jsonl không ghi F1 hay acc^eff theo từng node, nên analysis không tính lại được. Sai dây ở đây làm sai đường F1, một độ đo chính của đề.
- *Test:* `test_round_record_aggregates_are_population_statistics_of_the_matching_per_node_values` (`test_observer.py`).
  - Evaluator giả trả acc = (0.1, 0.2, 0.3, 0.6), acc_local = (0.5, 0.4, 0.9, 0.2), acc_eff = (0.7, 0.1, 0.3, 0.25), f1 = (0.05, 0.15, 0.35, 0.95).
  - 12 trường phải bằng đúng fmean, pstdev, min, max của bộ tương ứng; ví dụ `f1_macro_mean == 0.375`, `acc_std == 0.18708286933869706`.
  - **Kỳ vọng: PASS.**

**P0-4: P_i không được kiểm khớp id node** (run.py → Assembly.evaluator, manifest).
- *Bằng chứng:* e2e chỉ kiểm tổng mỗi hàng bằng 450, mà với IID mọi hàng như nhau. Mutant "thống kê theo các phần trước khi gán vị trí" (tương đương gõ nhầm `raw.parts`) qua 46/46 test. Hậu quả: Acc^loc_i được tính bằng phân bố nhãn của node khác.
- *Test:* `test_node_class_counts_follow_node_ids_so_acc_local_uses_each_nodes_own_labels` (`test_seeds.py`).
  - Smoke với Dirichlet α = 0.1, min_size = 8.
  - Với mọi i, `node_class_counts[i] == bincount(nodes[i].data.y)`, ở cả setup lẫn manifest.
  - **Kỳ vọng: PASS.**

### P1

| Mục | Test đề xuất (file) | Nội dung | Kỳ vọng |
|---|---|---|---|
| P1-1 | `test_t_inv_4_server_path_is_bitwise_independent_of_node_order` (`test_engine_core.py`) | Kích thước (64, 96, 128, 200), ba thứ tự duyệt, `server_path=True`, 3 round. State trùng `torch.equal`; snapshot đã gửi, kể cả snapshot server phát xuống, không đổi. Tham số hóa thêm gossip(4) và FC(4) | PASS |
| P1-2 | Tham số hóa `test_t_inv_11_*` với n ∈ {4, 5, 20} (`test_observer.py`) | FC + MH float32: mọi state trùng từng bit; `consensus_distance == 0.0`; `acc_std == 0.0`; Acc^loc_i = P_i · recall (1e-12). SAN-5 đòi đúng 0 ở N = 20 | PASS |
| P1-3 | `test_t_inv_8_dirichlet_resamples_when_every_remaining_probability_is_zero` (`test_partition.py`) | `repeat(arange(10), 10)`, α = 1e-4, min_size = 1, N = 2, rng(0). `retries` và các phần trùng hàm tham chiếu; ghép lại phủ đủ | PASS |
| P1-4 | `test_t_inv_8_a_node_with_exactly_min_size_samples_is_accepted` (`test_partition.py`) | IID 256 nhãn, N = 4: 4 × 64 được nhận; 255 nhãn thì ValueError. Dirichlet α = 0.1 trên MNIST N = 20, seed 2: min_size 72 cho retries 0, 73 cho retries 1 | PASS |
| P1-5 | `test_t_inv_8_quantity_retries_count_the_resamples_of_the_reference` (`test_partition.py`) | Quantity β = 0.5, min_size = 61, N = 20, rng(2): `retries == 19`, các phần trùng tham chiếu | PASS |
| P1-6 | `test_exhausting_retries_makes_one_try_plus_exactly_1000_resamples` (`test_partition.py`) | Cấu hình không thể đạt; sau ValueError, trạng thái rng bằng một rng sinh đôi đã rút đúng 1,001 lần thử. Bắt lỗi `range(MAX_RETRIES)` lệch một | PASS |
| P1-7 | `test_t_inv_13_processes_match_the_simulator_on_a_sparse_evaluation_grid` (`test_process_runtime.py`) | `local_steps = 10`, `every_steps = 4`, τ = 2, nên các mốc là round 2, 4, 5 (đường E8 dùng); gossip và ring. Hiện T-INV-13 chỉ chạy với lịch "mỗi round", nên lỗi lệch round ở `is_due` của node không bị bắt | PASS (probe 7 s) |
| P1-8 | Ratchet F5 | `MIN_TESTS` ở Makefile vẫn là 34, trong khi bộ test có 611. Nâng lên 611 ở C3 (thắt chặt, đúng CONSTRAINTS "cập nhật ở mỗi checkpoint") | — |

### P2

| Mục | Test đề xuất | Kỳ vọng |
|---|---|---|
| P2-1 **(lỗi tiềm ẩn, đã tái hiện)** | `test_uniform_ignores_edge_attributes_named_weight` (`test_mixing.py`): `path_graph(3)` có `weight = 5` trên cạnh 0–1 phải cho cùng W như khi không có. Hiện `nx.to_numpy_array` đọc `weight`, cho W sai mà vẫn qua INV-4; sửa bằng `weight=None`. Các đồ thị của dự án không có thuộc tính cạnh nên chưa gây hại | **FAIL** |
| P2-2 | `size_weighted` với sizes `[0,1,1]`, `[-1,1,1]`, sai shape, `[nan,1,1]` phải raise ValueError. Ca NaN hiện trả W toàn NaN; engine chặn ở bước sau | FAIL ở ca NaN |
| P2-3 | Khi cạn số lần lấy mẫu lại, thông báo phải nêu cả nguyên nhân "xác suất còn lại bằng 0" (`test_partition.py`) | **FAIL** |
| P2-4 | `spectral_gap(np.eye(1))` raise ValueError. Trên ER(20, 0.3), gap của uniform phải bằng gap của ma trận đối xứng đồng dạng D^{-1/2}(A + I)D^{-1/2}. Mutant "luôn dùng eigvalsh" cho 0.2643 thay vì 0.3275 mà vẫn qua test hiện có | PASS |
| P2-5 | N = 2 qua engine (FC, gossip, ER, star): W = ½J, 2 message/round, consensus 0.0, drift = ‖w₁ − w₂‖/2; unit test Ξ cho 2 node | PASS |
| P2-6 **(cần nhóm quyết)** | Config có nên từ chối ring với N < 3 không? Hiện chỉ lỗi lúc lắp. Nếu giữ hành vi hiện tại thì cần test khẳng định attempt failed có lý do rõ | FAIL nếu chọn từ chối ở config |
| P2-7 | Gossip với N = 21: E[W_ii] = (N + 1)/(2N), E[W_ij] = 1/(2N), trong ±4 SE; byte mỗi round với N lẻ; gọi tường minh `check_union_connected` | PASS |
| P2-8 | FC size-weighted với kích thước (64, 64, 64, 5000) và (64, 5000): mọi node trùng từng bit, khớp FedAvg tham chiếu | PASS |
| P2-9 | Lớp chiếm đúng 5% được tính là có mặt (`classes_ge_5pct_mean`); nhãn âm bị partition từ chối | PASS |
| P2-10 | Gossip và ER-gossip ở mức lượt chạy: `spectral_gap is None`, π = 1/N | PASS |
| P2-11 | T3b (ER-gossip): `graph_at` là hàm thuần của (seed, round); chạy qua engine cho 2 message mỗi cặp ghép, mọi matching là tập con của cạnh ER | PASS |
| P2-12 | `confusion_matrix([0, 1, 2], [0])` phải raise ValueError (hiện broadcast im lặng; Evaluator luôn truyền hai mảng cùng độ dài) | **FAIL** |
| P2-13 | Đường server: node không nhận được snapshot server phát xuống thì raise RuntimeError "thiếu snapshot"; Transport không có id server thì lỗi ngay ở lần gửi đầu | PASS |
| P2-14 | T-INV-13 thêm các ca của E8: gossip với Dirichlet α = 0.1, và ring với model M0 (UNVERIFIED, khoảng 10 s mỗi ca) | PASS |

## 5. R@x, bytes@x, AUC: chờ T15b

`analysis/convergence.py` chưa có, nên các test dưới đây chính là test RED của T15b, và sẽ là P0 khi T15b được làm.

**Chuẩn áp dụng là SPEC REQ-04:**
- Lần vượt là mốc p có acc(p) ≥ x trong khi mốc trước đó < x.
- Lần vượt bền vững khi mọi mốc trong (p, p + 200] đều ≥ x − 0.5 điểm %, và sau p còn ít nhất một mốc.
- SPEC §10 dòng 18 đã thay quy tắc "3 điểm kế tiếp" của KH §3.6. Test số 7 phân biệt hai quy tắc.

Các ví dụ dưới đây dùng lưới 100, 200, … bước và x = 0.95:

1. acc = [0.80, 0.90, 0.94, 0.96, 0.955, 0.96] → R@95 = **350** (nội suy giữa 300 và 400).
2. Không bao giờ đạt (max 0.94) → **"> ngân sách"**.
3. Mốc đầu tiên đã ≥ x và bền vững → **"≤ 100"**, dùng 100 khi tính trung vị (cần chốt trong T15b).
4. Lần vượt không bền vững bị bỏ qua: [0.90, 0.96, 0.93, 0.94, 0.96, 0.96, 0.97] → **450**.
5. Sụt đúng bằng x − 0.5 điểm % vẫn bền vững: [0.90, 0.96, 0.945, 0.95] → **183.33…**
6. Cửa sổ là p + 200 bước:
   - sụt ở p + 200: [0.90, 0.96, 0.95, 0.944, 0.96, 0.96] → **437.5**;
   - sụt ở p + 300 không ảnh hưởng.
7. Chỉ vượt ở mốc cuối: [0.90, 0.92, 0.96] → **"> ngân sách"** theo SPEC (theo KH sẽ là "đạt").
8. Vượt ở mốc kế cuối chỉ cần mốc còn lại: [0.90, 0.96, 0.95] → **183.33…**
9. Mốc bằng đúng x được tính là vượt: [0.90, 0.95, 0.95, 0.95] → **200**.
10. Lượt phân kỳ: R@x là "không đạt", AUC là "không xác định", và lượt được đếm vào "k/n seed phân kỳ".
11. Lưới khác lưới chuẩn thì báo lỗi. Ngoại lệ: lưới của B1/B1-4k, và tiền tố lưới của lượt phân kỳ.
12. R@x quy ra round (chia cho τ); bytes@x nội suy trên `bytes_cum`.
13. Trung vị qua seed coi "> ngân sách" là +∞: [350, ">", "≤ 100", 450, 500] → **450**, "4/5 seed đạt".
14. AUC là trung bình theo quy tắc hình thang trên lưới: [0.5, 0.7, 0.9, 1.0] trên 100…400 → **0.78333** (T15b cần chốt đầu mút).
15. Độ dao động tính theo từng node, không tính trên acc_mean (G24): hai node dao động ngược pha thì acc_mean phẳng mà độ dao động vẫn lớn hơn 0.

## 6. Tiến trình chính đã xác minh lại

| Khẳng định | Cách xác minh | Kết quả |
|---|---|---|
| P0-1 lỗi thật | Chạy lại probe hai runtime | inprocess raise ValueError (INV-4), processes ghi `ok`. Grep: `check_union_connected` chỉ được gọi ở engine.py:358 |
| P0-2 mutant seed sống | Chạy lại plugin ghim seed 0 cho PARTITION và BATCH với test_seeds + test_baselines | 28 passed với cả hai |
| P0-3 thiếu kiểm trường tổng hợp | Grep `tests/` | Chỉ có `test_observer.py:396` (std = 0) và danh sách khóa ở test_contracts_smoke |
| P2-1 `uniform` đọc `weight` | Chạy lại probe | W khác khi cạnh có `weight = 5`, mà tổng hàng vẫn bằng 1 và vẫn không âm (qua INV-4) |
| P1-8 ratchet | Đọc Makefile:17 và CONSTRAINTS.md:79 | `MIN_TESTS := 34` |
| Quy tắc R@x | Đọc SPEC REQ-04 và §10 dòng 18 | Đúng như §5 |

Các khẳng định khác lấy từ báo cáo của subagent; tiến trình chính đã đọc đối chiếu với code nhưng không chạy lại.

## 7. Đề xuất

1. **P0-1 trước tiên**, theo quy trình debug P3.4: viết test Prove-It, xác nhận FAIL, sửa `ProcessRun`, rồi chạy `make check-full`. Phải xong trước E8 và trước khi dùng T-INV-13 làm bằng chứng.
2. Thêm các test PASS của P0-2, P0-3, P0-4, rồi P1-1 tới P1-7. Mỗi test dưới 1 s, trừ test cạn số lần lấy mẫu lại trên MNIST (khoảng 2 s).
3. Ở C3: nâng `MIN_TESTS` lên số test hiện có (P1-8).
4. Các lỗi P2 có test FAIL (P2-1, P2-2 ca NaN, P2-3, P2-12) sửa theo P3.4 khi tiện. P2-6 chờ nhóm quyết.
5. Viết 15 test của §5 làm RED cho T15b.
