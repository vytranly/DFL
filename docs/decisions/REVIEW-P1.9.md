# Phản biện thiết kế P1.9 (doubt-driven-development)

Ngày 2026-09-30. Skill `doubt-driven-development`, chạy ở phiên chính.

- **CLAIM** (chỉ để ghi lại, không đưa cho reviewer): SPEC + ADR đáp ứng REQ-01…14 và cho kết luận khoa học hợp lệ.
- **ARTIFACT:** `SPEC.md`, ADR-002, ADR-005, ADR-006, ADR-007, ADR-009.
- **CONTRACT:** H§2.1 (REQ-01…14), H§2.4 (bẫy G1–G25); ràng buộc: nhóm sinh viên 3 người, chủ yếu CPU, 12 tuần.
- **Reviewer:**
  - Nội bộ: subagent general-purpose ngữ cảnh mới, nhận nguyên văn prompt đối kháng của skill cùng một file chứa ARTIFACT + CONTRACT, và chỉ được đọc file đó. Hai phát hiện (F2d, F22) có dẫn thêm CLAUDE.md từ ngữ cảnh phiên của nó.
  - Cross-model: nhóm tự chạy thủ công trên một model khác với cùng prompt (`build/review/P1.9-cross-model-cycle<k>.md`, không commit). Vòng 1: có làm. Vòng 2 và 3: xem ghi chú ở từng vòng.
- **Phân loại** theo thứ tự ưu tiên của skill: *hiểu sai contract* / *đúng, sửa* / *đúng, chấp nhận trade-off* / *nhiễu*.

## Vòng 1

Phát hiện của reviewer nội bộ: F1–F33. Phát hiện cross-model: X1–X13.

| ID | Mức | Tóm tắt | Phân loại | Xử lý |
|---|---|---|---|---|
| F1 | cao | Không có trường cho validation; E0a có thể phải chọn η trên test; SAN đo trên test nên debug dễ thành chọn theo test | đúng, sửa | Tập đánh giá là tham số cấu hình (E0a dùng validation); `final_eval_split` trong `SummaryOk`; quy tắc "khi SAN không đạt chỉ sửa lỗi có test tái hiện" (SPEC §5.2, §5.3, §9.3; ADR-007; CONSTRAINTS R7) |
| F2 | cao | Không nói lượt phân kỳ được ghi thế nào; NaN có thể lọt khỏi thống kê | đúng, sửa | Trạng thái `diverged` (`SummaryDiverged`), không chạy lại, không có NaN, báo "k/n seed phân kỳ" (SPEC §5.3, INT-3, EXP-2/3; ADR-005/007/008; CONSTRAINTS R4) |
| F3 | cao | Thứ tự cắt Tier 2 xóa bằng chứng duy nhất của REQ-02 (quantity skew), RQ3, RQ4, H4, H6 mà không có quy tắc | đúng, sửa | Thêm E4a (6 lượt quantity skew) vào Tier 1; bảng ánh xạ nhóm thí nghiệm → H/RQ/REQ; quy tắc "chưa kết luận (cắt vì ngân sách)" (SPEC EXP-5; ADR-007) |
| F4 | cao | Mọi thay đổi trong `src/` làm stale kết quả; `analysis` đọc qua `dfl.io` nằm trong `src/`; T22 làm sau M4 sẽ phá lưới | đúng, sửa + trade-off | `analysis` chỉ import `dfl.contracts`, tự đọc JSON (SPEC §3; contract import-linter); T22 phải xong trước E0a (ADR-002); độ chặt của `code_version` giữ nguyên và ghi là trade-off (ADR-008) |
| F5 | cao | Thứ tự ưu tiên đặt H (chứa REQ) thấp nhất; khác biệt với H không được liệt kê | đúng, sửa | Đề là gốc; CLAUDE.md có trong thứ tự; SPEC §10 liệt kê cả khác biệt với H (dòng 12–14); `sweep` luôn đòi cây sạch như CLAUDE.md |
| F6 | trung bình | $c_i$ của size-corrected cần $\bar n$ toàn cục (trái INV-1); π ghi nhầm tên | đúng, sửa | Ngoại lệ INV-1 duy nhất, ghi là hạn chế; đổi thành `objective_weights` (ADR-005; SPEC §8, §10 dòng 23; `contracts.py`) |
| F7 | trung bình | Sổ sách attempt/dùng lại chưa định nghĩa: phạm vi tra cứu, trùng attempt `ok`, tiến trình bị ngắt, hai sweep tranh nhau | đúng, sửa | `results/runs/<run_id>/` + chỉ mục theo nhóm; mỗi run_id nhiều nhất một attempt xong; attempt không có summary được chạy lại; khóa sweep; định nghĩa cấu hình chuẩn hóa (SPEC §5.3; ADR-008) |
| F8 | trung bình | Ngân sách B1 chưa định nghĩa; INT-2 chặn so sánh với B1; SAN-1 có thể fail vì η của ô khó; B3 đếm 380 thay vì 40 message | đúng, sửa | B1 10 epoch, η riêng trên validation; baseline là đường tham chiếu có nhãn; B3/E1c đếm theo star (SPEC §5.2, REQ-05, REQ-12, INT-2; ADR-007) |
| F9 | trung bình | Mốc đánh giá trùng round ở τ = 50; "3 điểm bền vững" phụ thuộc τ; REQ-04 chưa nói đơn vị là bước cục bộ | đúng, sửa | Bỏ mốc trùng; cửa sổ 200 bước cục bộ; R@x theo bước (SPEC REQ-04, §10 dòng 18; ADR-007) |
| F10 | trung bình | Kiểm định H2 thiếu biến giải thích, chiều, cặp topology; H1, H3–H7 không có quy tắc quyết định; RQ2 không có kế hoạch phân tích | đúng, sửa | Quy tắc quyết định đăng ký trước; định nghĩa đầy đủ H2; hồi quy khám phá cho RQ2 (ADR-007; SPEC EXP-4) |
| F11 | trung bình | Chọn η: 1 seed, ngưỡng hòa 0.2 điểm % nhỏ hơn nhiễu; "accuracy validation" chưa định nghĩa; ô khó nhất quyết định η; điều kiện "≥ 2 lần" vô nghĩa trên lưới | đúng, sửa + trade-off | Định nghĩa accuracy/AUC validation; bỏ "≥ 2 lần"; hạn chế ghi vào INT-4; ngưỡng hòa giữ nguyên (G25) và ghi là trade-off (ADR-006/007) |
| F12 | trung bình | Lượt bẩn mang `code_version` của commit sạch; cài đặt không dùng file khóa; file chưa track trong `src/` | đúng, sửa | `code_version` = sha256 nội dung file; kiểm môi trường theo `requirements.lock`; định nghĩa cây sạch tính cả file chưa track; `analysis` từ chối lượt `dirty` (SPEC §5.3, INT-2; ADR-008) |
| F13 | trung bình | Manifest thiếu commit; REP-1 không nói check out commit nào; hình không ghi nguồn; `results/` không sao lưu | đúng, sửa | `git_commit` trong manifest; REP-1 check out commit đó; `figures/PROVENANCE.json`; sao lưu sau mỗi nhóm, cấm `git clean -x` (SPEC §5.3, §8, §9.3; ADR-008) |
| F14 | trung bình | Số luồng chưa cố định, có thể phụ thuộc `--workers`; OMP/OpenBLAS chưa ghim; máy ảo có thể đổi phần cứng | đúng, sửa | Số luồng mỗi lượt là hằng số (mặc định 1) trong `run_id`; ghim biến môi trường; manifest ghi hash cờ CPU (SPEC §5.2; ADR-002) |
| F15 | trung bình | "Pathological partition" của đề không ánh xạ rõ tới cách chia nào | đúng, sửa | Ánh xạ bốn loại đề nêu ở SPEC §1.4 và REQ-02 |
| F16 | trung bình | Uniform trùng MH trên đồ thị đều, nên E4 có thể là ablation rỗng | đúng, sửa | Uniform chỉ chạy trên ER (ADR-005) |
| F17 | trung bình | Seed `[seed, round]` có thể trùng luồng khác; ánh xạ tên → số chưa có; w⁰ dùng RNG toàn cục; tạo Generator không seed vẫn qua kiểm tra | đúng, sửa | spawn_key với ID luồng cố định; w⁰ lấy mẫu từ luồng init; chỉ `seeding.py` được tạo Generator (SPEC §6; ADR-006, ADR-009; `check-floor` F7) |
| F18 | trung bình | `nbytes` do người gọi đặt nên T-INV-6 tự đúng; người nhận có thể sửa snapshot dùng chung | đúng, sửa | `nbytes` tính từ tensor thật; T-INV-4 kiểm snapshot không đổi; tổng hợp ghi ra bộ đệm mới (SPEC §6, T-INV-4, T-INV-6; `contracts.py`) |
| F19 | trung bình | TorchDistTransport phải import torch nên không nằm được trong `network`; gloo cần tensor; Q1 chưa có hạn | đúng, sửa + trade-off | Đặt ở `sim`; hạn hỏi lại cuối tuần 3; tuần tự hóa và `isend`/`irecv` ghi là trade-off cần kiểm ở T22 (SPEC ASSUMPTIONS, §3.1; ADR-002) |
| F20 | trung bình | Macro-F1 chia 0/0 khi một lớp không được dự đoán; sklearn cần `labels`, `zero_division` | đúng, sửa | Công thức $2TP/(2TP+FP+FN)$ trên đủ 10 lớp; oracle với `labels=range(10), zero_division=0`; thêm ca kiểm (SPEC T-INV-10; ADR-007) |
| F21 | trung bình | Golden so với spike của chính nhóm; min_size khác NIID-Bench; vòng lấy mẫu lại không có giới hạn | đúng, sửa + trade-off | Tối đa 1,000 lần lấy mẫu lại, ghi `partition_retries` (`PartitionResult`); min_size ghi rõ là khác có chủ đích; golden chỉ kiểm nhất quán với spike, còn độ trung thành của spike với NIID-Bench đã kiểm bằng đọc mã (DIGEST §2.4) — trade-off |
| F22 | trung bình | Lệnh smoke trong CLAUDE.md không có `--out`, sẽ ghi vào `results/` | đúng, sửa | CLAUDE.md thêm `--out build/smoke`; profile smoke không bao giờ ghi vào `results/` (SPEC §4, §5.3) |
| F23 | trung bình | Đường găng T06–T11 dồn cho một người (TV2) | đúng, cần nhóm quyết | SPEC QS-9 (đề xuất TV1 nhận T09) |
| F24 | thấp | `eigvalsh` sai cho W không đối xứng; mẫu MH không chặn đồ thị có hướng/đa cạnh | đúng, sửa | Mẫu mới chặn cả hai (đã chạy thử); ghi chú dùng `eigvals` cho uniform (SPEC §6) |
| F25 | thấp | Ngoại lệ W = I của B2 với yêu cầu liên thông chưa nói rõ | đúng, sửa | Ghi rõ trong docstring `TopologySchedule`, `MixingPolicy` |
| F26 | thấp | #C = k cũng lệch kích thước node và mục tiêu nhưng không bắt báo cáo | đúng, sửa | Báo cáo kèm mọi kết quả Non-IID (SPEC §8; ADR-005) |
| F27 | thấp | check-task đỏ trước T10; ENG-5 mâu thuẫn "chỉ cảnh báo" | một phần nhiễu, một phần sửa | Đã kiểm: check-task xanh vì gom cả `tests/unit`; ghi chú ở SPEC §4; ENG-5 thành mục tiêu |
| F28 | thấp | Profile smoke không có chỗ cho validation 5k và tập con 2,000 ảnh | đúng, sửa | Smoke: 200 validation, dùng toàn bộ 500 test làm tập đánh giá (SPEC §5.2) |
| F29 | thấp | import-linter tính cả import dưới TYPE_CHECKING | nhiễu | Đã đặt `exclude_type_checking_imports = true` ở P1.8 (reviewer không thấy `pyproject.toml`); ghi rõ ở ENG-6 |
| F30 | thấp | PyYAML đọc `5e-2` thành chuỗi; dạng chuẩn để hash chưa định nghĩa | đúng, sửa | Định nghĩa cấu hình chuẩn hóa và kiểm kiểu (SPEC §5.3; ADR-008) |
| F31 | thấp | "giờ-CPU" mơ hồ | đúng, sửa | Đo thật và đổi sang giờ-lõi (ADR-007; SPEC §10 dòng 21) |
| F32 | thấp | D-PSGD với minibatch không phải FedSGD | đúng, sửa | Ghi rõ khác biệt; đề xuất hỏi GVHD (Q8) (ADR-006; SPEC REQ-08, §11.1) |
| F33 | thấp | "T3 rẻ nhất trong các topology" sai (T3b, B2 ít hơn) | đúng, sửa | Sửa câu (ADR-009) |
| X1 | cao | Smoke chỉ có 12 bước, không tới mốc đánh giá đầu tiên (20), nên `rounds.jsonl` rỗng | đúng, sửa | Lịch là tham số profile; smoke đánh giá mỗi round; round cuối luôn được đánh giá (SPEC §5.2, ENG-4; ADR-007) |
| X2 | cao | Ngân sách CPU bị ước tính thấp khoảng 10 lần (giả định 85 ms/bước, 578 giờ-CPU) | đúng về lo ngại, sai về số | Đo thật ngày 2026-09-30: 10.5 ms/bước (1 luồng), đánh giá 2,000 ảnh 0.40 s, nên khoảng 26 phút-lõi/lượt và Tier 1 + 2 khoảng 120 giờ-lõi. Rủi ro thật là máy chủ dùng chung, tải khoảng 7 (ADR-007; SPEC §10 dòng 21) |
| X3 | cao | TorchDistTransport mâu thuẫn contract `pure-no-torch` | đúng, sửa | Như F19 |
| X4 | cao | "Node không được ghép giữ nguyên model" mơ hồ: có chạy Adapt không? | đúng, sửa | Node vẫn chạy Adapt, chỉ bỏ Combine (W_ii = 1) (ADR-006, ADR-009) |
| X5 | cao | `git rev-parse HEAD:src` không phản ánh cây bẩn và lỗi khi không có git | đúng, sửa | Như F12: sha256 nội dung file trên đĩa (SPEC §5.3; ADR-008) |
| X6 | cao | Size-corrected không tương đương mục tiêu có trọng số khi có τ bước cục bộ; $c_i$ làm lệch số hạng prox | đúng, chấp nhận trade-off | Ghi rõ là gần đúng ("objective inconsistency", FedNova — Wang et al. 2020, arXiv 2007.07481, cần đối chiếu trước khi dẫn trong báo cáo); chỉ dùng với μ = 0; báo cáo như hạn chế (ADR-005) |
| X7 | trung bình | T-INV-10 dùng trọng số $P_i$ cho cả $\mathrm{Acc}^{eff}$ | đúng, sửa | $\mathrm{Acc}^{eff}$ dùng $P_{eff}(y)/P_{eval}(y)$ (SPEC T-INV-10) |
| X8 | trung bình | Snapshot dùng chung có thể bị người nhận sửa tại chỗ | đúng, sửa | Như F18 |
| X9 | trung bình | Nhầm Ξ là khoảng cách bình phương, nên cận $\rho^t$ lỏng | hiểu sai (do ký hiệu mơ hồ) | Ξ là khoảng cách RMS; chỉ số ½ là nửa bước. Đã làm rõ ký hiệu (SPEC T-INV-1; ADR-007; `contracts.py`) |
| X10 | trung bình | REQ-05 gắn cứng 46,730 tham số | đúng, sửa | Byte = tổng `nbytes`; 46,730 chỉ là ví dụ của SmallCNN (SPEC REQ-05, T-INV-6) |
| X11 | thấp | η chung có thể bất lợi cho τ = 50, không có kiểm tra riêng | đúng, sửa | E0a quét thêm η cho τ = 50 (6 lượt), có quy tắc chạy bổ sung (ADR-006/007; SPEC EXP-2/3) |
| X12 | thấp | Macro-F1 chia 0 ở #C = 1, 2 | đúng, sửa | Như F20 |
| X13 | thấp | Mẫu MH không chặn DiGraph/MultiGraph | đúng, sửa | Như F24 |

Kết quả vòng 1 (46 phát hiện):
- 43 đúng, đã sửa hoặc ghi trade-off. Trong đó F27 chỉ đúng một phần; X2 đúng về lo ngại nhưng sai về số, đã thay bằng số đo.
- 1 cần nhóm quyết (F23 → QS-9).
- 1 nhiễu (F29).
- 1 hiểu sai do ký hiệu mơ hồ (X9); đã làm rõ ký hiệu.

`make check-full` xanh sau khi sửa (34 test).

## Vòng 2

Reviewer nội bộ là một subagent ngữ cảnh mới, nhận nguyên văn prompt đối kháng và bundle sau vòng 1. Nó trả về F1–F30 (đánh số lại từ 1; khác F1–F33 của vòng 1). Cross-model vòng 2: file `build/review/P1.9-cross-model-cycle2.md` đã gửi nhóm; lúc ghi mục này chưa có kết quả.

| ID | Mức | Tóm tắt | Phân loại | Xử lý |
|---|---|---|---|---|
| F1 | cao | SAN-2 không công bằng: B1 có η riêng và khoảng 17,190 bước, DFedAvg-FC-IID có η chung và 4,000 bước. SAN-2 có thể fail mà không có lỗi, và không có lối ra hợp lệ; Ask first mâu thuẫn Never | đúng, sửa (nhóm quyết phần nới đối chứng) | Thêm B1-4k (3 lượt ở E0b). Thêm quy trình khi debug không tìm ra lỗi: dừng, báo nhóm, báo cáo SAN không đạt (SPEC §9.3; CONSTRAINTS R7). Làm rõ Ask first/Never: không bao giờ đổi vì kết quả test. Đổi đối chứng sang B1-4k là nới, nên đã hỏi nhóm: nhóm chốt B1-4k, giữ ngưỡng 1.0 điểm % (2026-09-30, QS-3); R7 đã sửa theo |
| F2 | cao | E0b so η/2, η, 2η trên test; Never chỉ cấm đổi lr "không có ADR", nên có đường chọn η trên test qua ADR | đúng, sửa | E0b chỉ báo cáo: đổi hướng thì H2 ghi "không bền với η". Never: không đổi siêu tham số, lịch hay ngưỡng vì kết quả test, kể cả qua ADR (SPEC §8; ADR-007; CONSTRAINTS R6) |
| F3 | cao | `code_version` hash cả cây `src/`: thêm file không liên quan hay sửa lỗi làm stale mọi lượt; phần mở rộng không so được; không có ngân sách chạy lại | đúng, chấp nhận trade-off (một phần sửa) | Giữ hash chặt (như F4 vòng 1). SPEC §1.4 và ADR-008: phần mở rộng muốn so định lượng thì viết code trước E0a; E5 không cần code mới. Phương án hash theo module bị loại, có lý do. Nêu chi phí chạy lại Tier 1 (khoảng 64 giờ-lõi, vượt dự phòng) và thứ tự cắt |
| F4 | cao | Cài đặt không dùng `requirements.lock`, nên sweep từ chối sau khi dựng lại venv, trên clone sạch (REP-1) và trên laptop demo; CRLF đổi `code_version` | đúng, sửa | Cài bằng `-r` và `-c requirements.lock`; file khóa chỉ gồm gói chạy; kiểm từng gói bằng `importlib.metadata`; `--out` ngoài `results/` cho phép lệch môi trường, ghi `env_match: false`. `.gitattributes eol=lf` và chuẩn hóa xuống dòng khi hash (SPEC §2, §4, §5; ADR-008). Cơ chế pip, git đã đối chiếu tài liệu chính thức |
| F5 | trung bình | `run` không kiểm môi trường; manifest chỉ ghi 5 phiên bản | đúng, sửa | Cả `run` và `sweep` kiểm. Manifest ghi mọi gói trong file khóa và `env_match` (`contracts.py`, test) |
| F6 | trung bình | File khóa gồm cả công cụ dev, nên nâng ruff làm stale mọi kết quả, đúng nhược điểm của phương án đã loại | đúng, sửa | File khóa sinh trong venv chỉ có gói chạy; phương án cũ ghi ở ADR-008 |
| F7 | trung bình | Không bắt báo cáo bộ kết quả của `code_version` cũ bị thay thế, nên có thể thay lặng lẽ một bộ kết quả bất lợi | đúng, sửa | `docs/experiments/CODE_VERSIONS.md`; `analysis` báo lỗi khi thiếu dòng; phụ lục báo cáo (SPEC INT-6; CONSTRAINTS R9) |
| F8 | trung bình | Mật độ mốc đánh giá khác nhau theo τ (τ = 50: 50 mốc, τ ≤ 20: 80), nên R@x, AUC và độ dao động lệch theo τ; chưa rõ chuỗi nào dùng cho R@x, cách xử lý lần vượt không bền và lần vượt ở mốc cuối | đúng, sửa | Lưới chung 100, 200, …, 4,000 bước cho mọi τ; định nghĩa đầy đủ R@x; danh sách unit test mới (SPEC REQ-04, §5.2, §10 dòng 18; ADR-007) |
| F9 | trung bình | Lời gọi RNG của torch thiếu `generator=` không bị bắt; vì cấm `manual_seed` nên nó cho cùng số ở mọi seed | đúng, sửa | T-INV-5: `torch.get_rng_state()` không đổi sau một lượt; khác seed thì khác w⁰ ở mọi tensor; `skip_init`/`fork_rng` (đã đối chiếu tài liệu 2.14) cho bước tạo module (SPEC §6; ADR-006; CONSTRAINTS F7) |
| F10 | trung bình | Không có cơ chế đếm theo star cho B3/E1c mà vẫn giữ INV-2; "tải" chưa định nghĩa | đúng, sửa | Đường server với server ảo chỉ số N qua Transport (2N message/round); tải = gửi + nhận; T0 tách riêng; T-INV-3, T-INV-6 mở rộng (SPEC REQ-05; ADR-005; `contracts.py`) |
| F11 | trung bình | Miễn trừ của B1 trong INT-2 không đủ (η, τ·R, lịch); không kiểm tập seed khi so ghép cặp | đúng, sửa | Baseline là đường tham chiếu có nhãn ngân sách và η, miễn kiểm biến cố định; so ghép cặp dùng giao các tập seed (SPEC INT-2; ADR-006, ADR-007) |
| F12 | trung bình | Size-corrected với SGD thường chính là bước học theo node (tới 4 lần), nhưng lại được dùng cho khuyến nghị về trọng số trộn | đúng, sửa + trade-off | Gọi đúng tên can thiệp; khuyến nghị về trọng số trộn chỉ dựa trên MH so với uniform; ghi hiệu ứng lẫn (ADR-005; SPEC REQ-13, EXP-5, INT-4) |
| F13 | trung bình | #C = 2 không tương đương shards của McMahan | đúng, sửa | Viết lại SPEC §1.4: hai dạng khác nhau, báo cáo nói rõ dùng #C = k |
| F14 | trung bình | Chọn η dựa trên 1 seed, 1 điểm cuối; seed của E0a chưa nêu; "η tốt nhất" của τ = 1/50 chưa định nghĩa | đúng, sửa + trade-off | E0a dùng seed 100; accuracy validation = trung bình 5 mốc cuối trên cùng ảnh; quy tắc cho τ = 1/50, hòa thì giữ η chung. Ngưỡng hòa 0.2 giữ nguyên, ghi là trade-off (ADR-007; SPEC §10 dòng 25) |
| F15 | trung bình | Với tải khoảng 7, "15 giờ" thực ra là 120–155 giờ; các gate nối tiếp; đòn bẩy giảm chi phí đánh giá đổi `run_id`; 24 lượt có điều kiện chưa tính; chỉ có một máy chủ | đúng, sửa | Tính lại theo lưới 40 mốc: 64 / 92 / 100 / 130 giờ-lõi. Thời gian thực = giờ-lõi chia số lõi rảnh (4–5.5 ngày với 1 lõi). T13 chốt lịch trước E0a; đường găng; điểm lỗi duy nhất (ADR-007; SPEC §3.3, §10 dòng 21) |
| F16 | trung bình | Q1 không có hạn chót hay mặc định; không có ngày của M2–M4 | đúng, sửa | Mặc định hết tuần 5 (SPEC ASSUMPTIONS 1; ADR-002); tuần của các mốc ở SPEC §3.3; tải của TV2 vẫn là QS-9 |
| F17 | trung bình | `analysis/` không được hash hay kiểm cây sạch; có thể đổi định nghĩa độ đo sau khi thấy kết quả mà không để lại dấu vết | đúng, sửa | `analysis.plots` từ chối khi `analysis/` bẩn; `analysis_version` trong PROVENANCE; định nghĩa độ đo có test và commit trước Tier 1, đổi sau đó phải báo cáo cả hai (SPEC §8, INT-1; ADR-008; CONSTRAINTS R1) |
| F18 | thấp–trung bình | Số lượt không chia theo ô (E1 70 so với 8 × 2 × 5 = 80; ring-IID ở đâu?) | đúng, sửa | Bảng chi tiết ở EXP-2/3: ô IID của E1 lấy từ E0b (5 seed). Khi lập bảng phát hiện thêm B3 không được chạy ở Tier 1, nên thêm B3-IID (3 lượt); cùng B1-4k, Tier 1 = 197 |
| F19 | thấp–trung bình | Vượt giới hạn lấy mẫu lại thì sao; lỗi tất định bị chạy lại mãi; NaN có thể lọt vào `train_loss_mean` trước khi phát hiện; lượt "bùng nổ" hữu hạn ghi là ok | đúng, sửa | Vượt giới hạn thì `failed` trước khi huấn luyện; `failed` không tự chạy lại (`--retry-failed`); kiểm trước khi ghi bản ghi; nêu rõ lượt hữu hạn vẫn là `ok` (SPEC §5.3, REQ-02; ADR-008; `contracts.py`) |
| F20 | thấp | US-2 lệch ENG-9; khóa bỏ quên sau crash phải xóa tay trong `results/` | đúng, sửa | US-2 đã sửa; `fcntl.flock` tự nhả; `run` và `sweep` cùng giữ khóa (đã đối chiếu tài liệu Python và flock(2)) |
| F21 | thấp | Kết quả smoke luôn `dirty`, mâu thuẫn test e2e vẽ hình và INT-2 | đúng, sửa | Chế độ `--draft` (nhãn DRAFT, không vào `figures/`); test e2e kiểm cả lời từ chối khi không có `--draft` |
| F22 | thấp–trung bình | Chưa nói kết luận "gợi ý" (3 seed) có được dùng cho khuyến nghị không | đúng, sửa | Mức khuyến nghị: chính / tạm / không (ADR-007; SPEC REQ-13, EXP-4, bảng EXP-5) |
| F23 | thấp–trung bình | Golden và phép so JS có tính vòng tròn; dung sai 0.02 lớn hơn chênh JS giữa IID và α = 100 | đúng, sửa + trade-off | Kiểm thứ tự tăng nghiêm ngặt và dung sai tương đối max(10%, 0.002); nêu rõ độ trung thành với NIID-Bench chỉ dựa trên đọc mã (SPEC REQ-02) |
| F24 | thấp | "Không có điều phối viên" không kiểm được; "ngoại lệ duy nhất" nói quá | đúng, sửa | Tính chất kiểm được: hai lịch tạo độc lập cho cùng đồ thị; liệt kê tri thức chung (ADR-002, ADR-009; SPEC REQ-01, §8, INT-4) |
| F25 | thấp | Luồng ngẫu nhiên của validation, tập con đánh giá, dữ liệu smoke, T3b, B1 chưa định nghĩa | đúng, sửa | Danh sách luồng theo seed của lượt và luồng dữ liệu cố định (`DATA_SEED`) (SPEC §6; ADR-009; `contracts.py`) |
| F26 | thấp | Thứ tự ưu tiên tự mâu thuẫn: cột thiết kế/kiểm chứng của REQ bị ghi đè; G1–G25 xếp dưới KH | đúng, sửa | Cột "Yêu cầu" là gốc; G1–G25 trên KH; cột thiết kế/kiểm chứng của H§2.1 dưới KH (đầu SPEC) |
| F27 | thấp | ER ít nhất 114 message (không phải 108); 0.312 không phải spectral gap | đúng, sửa | Đo lại trên 2,000 đồ thị ER liên thông: trung bình 115 (72–154), gap trung bình 0.20 (95%: 0.09–0.32); 0.312 là tốc độ trộn, gap của E[W] là 0.526 (ADR-005, ADR-009; SPEC REQ-07, §10 dòng 28) |
| F28 | thấp | Biến môi trường số luồng phải đặt trước khi import; start method chưa nêu; fork có thể treo | đúng, sửa | Mỗi lượt một tiến trình con mới (subprocess hoặc spawn, không fork); đặt biến trước khi khởi động; `set_num_threads` trước phép tính torch; ghi `get_num_threads()` (SPEC §5.2; ADR-002; đã đối chiếu tài liệu) |
| F29 | thấp | Không có contract độc lập giữa bốn module lá | nhiễu | Tầng `\|` của contract `layers` trong `pyproject.toml` đã cấm (reviewer không thấy file này); đã ghi rõ ở ENG-7 |
| F30 | thấp | Quantity skew chỉ có một mức | đúng, chấp nhận trade-off | Ghi ở ADR-007 và INT-4; thêm β = 0.1 tốn 6 lượt, chỉ làm nếu nhóm yêu cầu |

Phát hiện thêm khi đối chiếu (không do reviewer nêu):
- Chú thích của `code_version` trong `ManifestRecord` còn tả thiết kế cũ ("hậu tố +dirty"); đã sửa theo ADR-008.
- B3 không được chạy ở Tier 1 dù bảng EXP-5 ghi E0b phục vụ B1–B3; đã thêm B3-IID (xem F18).

Kết quả vòng 2 (30 phát hiện):
- 23 đúng, đã sửa.
- 3 đúng, đã sửa và ghi trade-off (F12, F14, F23).
- 1 đúng, đã sửa sau khi nhóm quyết phần nới đối chứng (F1: SAN-2 so với B1-4k, nhóm chốt 2026-09-30).
- 2 đúng, chấp nhận trade-off (F3, F30).
- 1 nhiễu (F29).

Quyết định của nhóm sau vòng 2 (2026-09-30): SAN-2 so với B1-4k, giữ ngưỡng 1.0 điểm % (QS-3, CONSTRAINTS R7). Nhóm cũng chấp nhận lưới đánh giá 100 bước cho mọi τ (kể cả việc R@90 của ô IID thô hơn) và seed 100 cho E0a.

`make check-full` xanh sau khi sửa (34 test).

## Vòng 3

Reviewer nội bộ (ngữ cảnh mới, nguyên văn prompt đối kháng, bundle sau vòng 2) trả về F1–F26 (đánh số lại). Bundle được tạo trước ba chỉnh sửa nhỏ: nhóm chốt đối chứng SAN-2, câu về tập seed trong INT-2, và câu "100 chia hết cho τ". Các phát hiện được đối chiếu với bản hiện tại. Reviewer cũng tự tính lại các số (gap của ring 0.0326, gap của E[W] 0.526, tốc độ trộn 0.312, 186,920 byte/message, khoảng 17,190 bước của B1, 40/380/20/khoảng 115 message, số lượt 197/81/278, giờ-lõi), và các số đều đúng.

| ID | Mức | Tóm tắt | Phân loại | Xử lý |
|---|---|---|---|---|
| F1 | cao | Không SAN nào kiểm được bước tổng hợp có tác dụng trên code chạy lưới: nếu combine không làm gì, mọi lượt DFL thành B2 mà SAN-1…4 vẫn đạt. QS-3 chưa có hạn chót | đúng, sửa | Thêm SAN-5, là các điều kiện cấu trúc kiểm trên `rounds.jsonl` thật (SPEC §9.3; CONSTRAINTS R7); tín hiệu dừng: std khác 0 trên FC + MH là lỗi. Phần QS-3 đã hết: nhóm chốt B1-4k trước khi có kết quả, sau khi bundle vòng 3 được tạo |
| F2 | cao | Never cho đổi siêu tham số "vì bằng chứng trên validation" mà không giới hạn thời gian, nên có thể thấy kết quả test rồi chạy validation mới để đổi η | đúng, sửa | Từ E0b trở đi chỉ được đổi để sửa lỗi có test tái hiện, không chạy validation mới sau khi đã thấy kết quả test (SPEC §8; CONSTRAINTS R6) |
| F3 | cao | Lưới chạy 4–5.5 ngày từ cây làm việc mà ba người vẫn commit vào; không rõ `code_version` tính ở tiến trình cha hay con | đúng, sửa | Lưới vào `results/` chạy từ worktree riêng check out đúng commit; sweep tính `code_version` một lần, tiến trình con kiểm lại, lệch thì `failed` và sweep dừng (SPEC §5.3, ENG-9; ADR-008; CONSTRAINTS R3; đã đối chiếu tài liệu `git worktree`) |
| F4 | trung bình | B3 được miễn kiểm biến cố định của INT-2 dù là ô so sánh của E1c/H6 và E4; ô size-weighted của E4 dùng hai cách đếm chi phí | đúng, sửa | Chỉ miễn khi vẽ làm đường tham chiếu; ô đường server được kiểm như mọi lượt DFL; mọi lượt size-weighted trên FC đi qua đường server (SPEC INT-2; ADR-005) |
| F5 | trung bình | R@x bị kiểm duyệt hai phía nhưng quy tắc quyết định dùng CI t của trung bình hiệu số | đúng, sửa | Kiểm định dấu trên thứ tự ghép cặp; hai giá trị cùng bị kiểm duyệt là hòa, và hòa tính là không đúng chiều; không dùng CI t cho R@x (SPEC REQ-04, §10 dòng 34; ADR-007) |
| F6 | trung bình | Hash gồm cả comment; chưa nêu đóng băng `src/`; sửa lỗi sau E1 thì η nào có hiệu lực; hash có gồm đường dẫn không | đúng, sửa (hash gồm comment là trade-off đã nhận ở vòng 1) | Đóng băng `src/` từ E0a tới hết Tier 2 (SPEC §8). Sửa lỗi thì chạy lại từ E0a, η chọn lại trên validation, ADR-007 ghi cả hai. Hash gồm đường dẫn tương đối (SPEC §5.3; ADR-007, ADR-008) |
| F7 | trung bình | Máy không nằm trong `run_id` và INT-2 không kiểm máy; mất máy chủ thì trộn máy lặng lẽ | đúng, sửa | `run_id` gồm máy (tên CPU + hash cờ CPU); INT-2 kiểm máy và dataset (SPEC §5.3, INT-2; ADR-008; `contracts.py`) |
| F8 | trung bình | T-INV-5 chỉ thấy đường code mà smoke chạy qua | đúng, sửa | Fixture `autouse` kiểm RNG toàn cục của torch trong mọi test tích hợp và e2e; mỗi đường code có ít nhất một test tích hợp (SPEC §7, T-INV-5, ENG-7) |
| F9 | trung bình | Mẫu `data/` không neo trong `.gitignore` sẽ bỏ qua cả `src/dfl/data/`, mà file bị ignore không hiện trong `git status` | một phần nhiễu, một phần sửa | `.gitignore` thật đã dùng `/data/` có neo (reviewer không thấy file này). Vẫn thêm vào định nghĩa cây sạch: không có file bị ignore trong `src/`, `configs/` (SPEC §5.3; ADR-008) |
| F10 | trung bình | Khóa chung trên `results/` mâu thuẫn với `--workers` và với việc chạy song song E1 và E3a/E4a | đúng, sửa | Khóa theo từng run_id, cộng khóa chỉ mục theo nhóm; lượt đang bị khóa thì bỏ qua (SPEC §5.3, ENG-9; ADR-008) |
| F11 | trung bình | Coi im lặng của GVHD là đồng ý cách hiểu "P2P"; T22 dự phòng không có chủ; bảng §3.1 lệch QS-9 | đúng, sửa + cần nhóm quyết | Viết lại ASSUMPTIONS 1 và ADR-002: im lặng không phải là đồng ý, ghi rủi ro. QS-10 (mặc định: làm T22 ngay sau T13). Bảng §3.1 ghi đề xuất của QS-9. Chủ của T22 đặt trong `tasks/plan.md` |
| F12 | trung bình | Size-weighted trên FC cần n và mọi $n_j$ (đại lượng toàn cục) mà không được liệt kê; siêu dữ liệu đi kèm không được đếm vào byte | đúng, sửa | Liệt kê tri thức chung (bậc láng giềng cho MH, $n_k$ ở server); siêu dữ liệu không được đếm, ghi là hạn chế (SPEC §8, REQ-05, INT-4; ADR-002, ADR-005) |
| F13 | trung bình | Tự chọn một cách ánh xạ bốn loại phân phối của REQ-02 mà không hỏi GVHD; dạng shards chỉ là tùy chọn | đúng, sửa + cần nhóm quyết | Q9 cho GVHD; QS-11 (mặc định: cài sẵn P2' ở T04b, chỉ dùng trong `describe`) (SPEC §1.4, §10 dòng 37, §11) |
| F14 | trung bình | Accuracy chỉ đo sau tổng hợp nên τ lớn được lợi; câu "không phụ thuộc τ" nói quá | đúng, sửa + trade-off | Sửa câu; ghi hạn chế (SPEC REQ-04, INT-4; ADR-006, ADR-007). Không đo accuracy trước tổng hợp, vì việc đó gấp đôi chi phí đánh giá |
| F15 | trung bình | Dấu của một hiệu tương tác trên 1 seed quyết định nhãn "bền/không bền với η" | đúng, sửa | Kiểm độ nhạy chỉ để mô tả, không gắn nhãn (ADR-007; SPEC §10 dòng 26) |
| F16 | trung bình | Nhiều phép so sánh cho mỗi giả thuyết, không kiểm soát bội | đúng, sửa + trade-off | Mỗi giả thuyết một phép so sánh chính khai báo trước. Nêu xác suất sai cho cả họ, tối đa khoảng 20%, trong mục hạn chế (SPEC EXP-4, INT-4, §10 dòng 36; ADR-007) |
| F17 | thấp | RQ2 theo spectral gap chỉ có ba cụm giá trị, lẫn với bậc, message và byte | đúng, chấp nhận trade-off | Ghi rõ phần "giải thích được không" của RQ2 chỉ trả lời ở mức mô tả (ADR-007; SPEC INT-4) |
| F18 | thấp | Golden và phép so JS có tính vòng tròn; sàn 0.002 lớn hơn JS của IID và α = 100 | đã xét (trade-off ở vòng 2, F23) | Phép kiểm thứ tự nghiêm ngặt vẫn phân biệt được IID với α = 100; giữ nguyên |
| F19 | thấp | Kết quả của test e2e phụ thuộc trạng thái git; ngữ nghĩa của `--draft` chưa định nghĩa | đúng, sửa | Không có `--draft` thì chỉ đọc lượt trong `results/`, nên e2e tất định. Định nghĩa đủ các điều `--draft` nới (SPEC INT-1; ADR-008) |
| F20 | thấp | Laptop Windows không có `fcntl`; file khóa ghim bản torch `+cpu`; không có cách cài cho laptop | đúng, sửa | `fcntl` chỉ import khi ghi vào `results/`; lệnh cài cho laptop không dùng file khóa, nên `env_match: false` (SPEC §4, §5.3; ADR-008) |
| F21 | thấp | INT-2 vừa báo lỗi vừa lấy giao các tập seed; lấy giao có thể loại bỏ seed xấu; thiếu dataset trong biến cố định | đúng, sửa | Câu mâu thuẫn đã sửa sau khi bundle được tạo. Thêm: tập seed khai báo trước trong thẻ; lượt thiếu hay phân kỳ không bị bỏ; thêm dataset và máy (SPEC INT-2) |
| F22 | thấp | Gói dev (scipy, joblib) có thể bị code chạy import mà không nằm trong file khóa | đúng, sửa | Contract import-linter `no-dev-deps` cho `dfl` và `analysis`; đã thử phá trên repo và bản sao, đều bắt được (`pyproject.toml`; CONSTRAINTS; SPEC ENG-7) |
| F23 | thấp | Chỉ kết quả ngược chiều giả thuyết mới kích hoạt việc săn lỗi | đúng, sửa | Tín hiệu dừng đối xứng: kết quả đúng chiều nhưng lớn bất thường cũng phải kiểm (SPEC §9.3) |
| F24 | thấp | Lưới 100 bước lệch G12 mà §10 chỉ dẫn KH; thứ tự cắt bỏ E1c lệch G19; G11 chưa đạt | đúng, sửa + trade-off | §10 dòng 18 dẫn thêm G12; dòng 35 ghi lệch G19. Về G11: #C = 1 có kích thước node gần đều (CV 0.05), nên là đối chứng label skew ít quantity skew, nhưng chỉ ở một mức; ghi là trade-off |
| F25 | thấp | Mâu thuẫn nhỏ: thiếu file YAML của E4a; US-2 lệch quy tắc attempt `failed`; T0 không có trong lưới; ngữ nghĩa của chỉ mục; thứ hạng của ADR còn Proposed | đúng, sửa | Sửa cả năm chỗ (SPEC đầu file, §1.2, §5.1, §5.2, §5.3, REQ-07; ADR-008) |
| F26 | thấp | Mẫu test mang nhãn T-INV-9 nhưng không theo quy ước tên; chưa nói tính liên thông được kiểm ở đâu | đúng, sửa | Đổi tên mẫu thành `test_t_inv_9_…` (đã chạy lại ruff và pytest trên bản sao); tính liên thông do `TopologySchedule` kiểm (SPEC §6) |

Kết quả vòng 3 (26 phát hiện):
- 18 đúng, đã sửa.
- 3 đúng, đã sửa và ghi trade-off (F14, F16, F24).
- 2 đúng, đã sửa phần của agent, phần còn lại cần nhóm quyết (F11 → QS-10, F13 → Q9 và QS-11).
- 1 đúng, chấp nhận trade-off (F17).
- 1 đã xét ở vòng 2 (F18).
- 1 một phần nhiễu, một phần sửa (F9).

`make check-full` xanh sau khi sửa (34 test, 5 contract import-linter).

## Kết thúc (STOP)

- Đã chạy đủ 3 vòng, giới hạn của skill và của P1.9. Vòng nào cũng có phát hiện đúng và đã sửa, nên đây không phải "doubt theater".
- Vòng 3 vẫn còn 3 phát hiện mức cao (F1–F3). Cả ba đã được sửa, nhưng chưa có vòng phản biện nào xác nhận bản sửa. Theo skill và theo P1.9, việc này được báo cho nhóm; agent không tự chạy vòng 4.
- Artifact lớn (SPEC khoảng 660 dòng và 5 ADR), nên mỗi vòng phản biện toàn bộ lại tìm ra lớp lỗi mới. Nếu cần phản biện thêm, nên tách artifact và phản biện riêng từng phần đã sửa: quy trình kết quả (§5.3, ADR-008), giao thức đánh giá (ADR-007), rồi sanity và toàn vẹn (§9.3).
- **Cross-model:** vòng 1 có làm. Vòng 2 và 3: file `build/review/P1.9-cross-model-cycle2.md` và `…-cycle3.md` đã gửi nhóm, nhưng nhóm không gửi lại kết quả, nên ghi là **Cross-model skipped (vòng 2, 3)**.

**Quyết định của nhóm sau vòng 3 (2026-09-30):** duyệt các phương án đề xuất. Commit P1.9 như hiện tại, không phản biện thêm. QS-10: T22 làm ngay sau T13 nếu tới hết tuần 5 GVHD chưa trả lời Q1. QS-11: cài sẵn P2' ở T04b, chỉ dùng trong `describe`.
