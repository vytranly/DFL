# Review code ở checkpoint C3 (P3.8 bước 4, câu hỏi P4.1)

- **Ngày:** 2026-10-02
- **Phạm vi:**
  - Review sâu `git diff 319a604..HEAD -- src/`: 15 file, khoảng 3.3k dòng chưa từng được review độc lập, gồm config, datasets, learning, io, run, describe, sweep, torchdist, process_runtime, cli.
  - Canh gác CONSTRAINTS trên toàn diff kể từ Gate G2 (`59ea6a7..HEAD`, 68 file).
- **Cách làm:**
  - Subagent code-reviewer chỉ đọc, review 5 trục cộng các câu hỏi P4.1, kèm vài probe nhỏ.
  - Tiến trình chính kiểm lại các khẳng định nặng nhất trên code (cột "Đã kiểm").
- **Không trùng với** `REVIEW-C3-P3.7.md`; review này xác nhận P0-1 và P1-8 ở đó.

## Tóm tắt

**Critical 0 · Important 6 · Suggestion 12** (thêm 9 mục nhỏ).

- **Đường lưới chính** (inprocess, N = 20): không có lỗi làm sai số liệu. Đã soát dây seed, snapshot và combine, W qua `_check_round`, đếm byte chỉ ở Transport, drift trước tổng hợp, acc^loc và acc^eff, chuẩn hóa ảnh bằng thống kê của phần chia cho node.
- **Các vấn đề nằm ở ba chỗ:** quy tắc ghi `results/`, provenance và chỉ mục, và process runtime (ảnh hưởng E8 và demo).
- **Canh gác CONSTRAINTS:**
  - Không ngưỡng nào bị hạ, không test nào bị skip hay xóa, không có suppression mới, không có stub.
  - Test `test_run_requires_an_output_directory` được gỡ đúng lúc `--out` thành tùy chọn theo SPEC; có test thay thế.
  - Các test chạy lượt thật được rút còn 2 round để vừa ngân sách (đã ghi trong SESSION_LOG).
  - Còn tồn: `MIN_TESTS` = 34 (nâng ở C3) và `check-task` 104–108 s.

## Important

| ID | Vấn đề | Kịch bản | Cách sửa đề xuất | Đã kiểm |
|---|---|---|---|---|
| I1 | `--out` trỏ vào `results/` thì mọi quy tắc ghi results/ bị bỏ qua: không kiểm smoke, cây bẩn, môi trường, không khóa, không từ chối run_id đã xong. Lỗi có ở cả `run` (run.py:593) lẫn `sweep` | `run --config smoke.yaml --out results` đưa lượt smoke vào results/ và tạo attempt `ok` thứ hai cho một run_id | Resolve `out`; nếu nó nằm trong results root thì từ chối, hoặc đi đường results/ đầy đủ. Thêm test cho cả hai lệnh | Có (run.py:593) |
| I2 | Chỗ đặt `results/` tính theo thư mục đang đứng. Sweep không chặn results/ nằm *trong* worktree | Quên tạo symlink `results` ở worktree thì kết quả của lưới nằm trong `<wt>/results`, và `git worktree remove` xóa sạch. Hai sweep từ hai thư mục có hai bộ khóa riêng | Khi ghi vào results/, từ chối nếu thư mục kết quả (sau khi resolve) nằm trong worktree; in đường dẫn đã resolve | Chưa |
| I3 | Chuỗi "máy" hash toàn bộ cờ CPU, gồm cả cờ do kernel, microcode hay hypervisor sinh ra (pti, ibrs…). Chỉ mục giữ một mục mỗi code_version và ghi đè danh sách run_id | Cập nhật kernel giữa E1 và E2 làm mọi run_id đổi: lưới bị chạy lại và lượt cũ biến mất khỏi chỉ mục | Chỉ hash cờ ISA (hoặc mức hỗ trợ CPU của torch), vẫn ghi đủ cờ vào manifest; khóa chỉ mục theo (code_version, máy) và không bao giờ bỏ run_id. **Cần sửa ADR-008 (hỏi trước)** | Một phần (code đúng như mô tả; tác động của reboot là UNVERIFIED) |
| I4 | `deploy: docker` được nhận rồi bị bỏ qua: không code nào ngoài config.py đọc nó | Các ô Docker của E8 thực ra chạy tiến trình cục bộ, trong khi run_id và manifest ghi docker | Từ chối `deploy: docker` cho tới T22c, hoặc làm chế độ observer chờ node từ bên ngoài trước E0a (ADR-010) | Có |
| I5 | ADR-011 đòi ghi tải trung bình của máy chủ vào manifest, nhưng schema chưa có. `--workers` đếm theo lượt, trong khi mỗi lượt processes là N + 1 tiến trình | E8 đo `comm_s` khi 36 tiến trình tranh 8 lõi | Thêm `os.getloadavg()` vào manifest hay timing trước Tier 1; E8 chốt `workers = 1`, hoặc sweep cảnh báo | Chưa |
| I6 | Vòng ATC được viết lại lần hai trong process runtime và thiếu các guard của engine: `check_union_connected` (P0-1), `_check_nodes` (BatchNorm/Dropout, Generator riêng, bộ nhớ riêng, w⁰ hữu hạn, c_i dương), kiểm `graph_at` thuần. Lý do phân kỳ cũng khác nhau giữa hai runtime | Mỗi lần sửa guard của engine, bản sửa chỉ vào đường inprocess | Tách một hàm round dùng chung và một bước preflight dùng chung, chạy trên assembly trước khi spawn; T-INV-13 so cả `summary.json` ở ca phân kỳ | Có (P0-1) |

## Suggestion

| ID | Vấn đề | Đã kiểm |
|---|---|---|
| S1 | **Process runtime ghi summary trước khi đợi node thoát.** Nếu `_wait_exit` raise sau đó, lượt `ok` thành lỗi "finish sau finish" (CLI thoát mã 1). Ở đường phân kỳ, `NodeFailed` thay chỗ `RunDiverged`. Sửa: đợi node thoát rồi mới `finish` | Có |
| S2 | **CodeMismatch ghi attempt failed vào run_id của code mới.** Sweep sau đó với code mới bỏ qua ô này trừ khi có `--retry-failed`. Sửa: sweep truyền `--expect-run-id`, và con thoát mã 5 trước khi chạm results/ (xem lại ADR-008) | Chưa |
| S3 | **Lượt bị OOM-kill không để lại dấu và bị chạy lại ở mọi lần sweep.** Exception trong `launch` không dừng sweep | Chưa |
| S4 | **Ghi file không fsync.** Một `summary.json` hỏng làm `plan()` của cả nhóm thí nghiệm không chạy | Chưa |
| S5 | **Ngữ nghĩa thời gian khác nhau giữa hai runtime.** `combine_s` của processes chứa cả `comm_s`; thời gian chờ ở barrier không vào pha nào; `TorchDistTransport` mã hóa lại snapshot cho mỗi láng giềng (torchdist.py:150), nên `comm_s` của FC tăng theo bậc × số tham số. Ảnh hưởng H8 của E8 | Có (torchdist.py:150) |
| S6 | **Hai barrier toàn cục mỗi round** (mặt phẳng đo): không rò giá trị, nhưng cần ghi vào mục hạn chế (REQ-15) và thẻ E8 | — |
| S7 | **Phụ thuộc vòng `sim.run` ↔ `sim.process_runtime`**, giấu bằng import trong hàm; `run.py` 604 dòng. Đề xuất tách `sim/assembly.py` | Có |
| S8 | **`objective_shift_tv` tính so với phân bố train gộp**, trong khi hợp đồng ghi "phân bố test" (contracts.py:431, heterogeneity.py:80). Chốt định nghĩa trước Tier 1 | Có |
| S9 | **Profile `fast` đánh giá trên tập test.** Dùng nó để chọn gì là rò rỉ test (G8). Đề xuất `eval.split: validation` trong `fast.yaml` (configs/, không đổi code_version) | Có |
| S10 | **Dữ liệu không có dấu vân tay** (torchvision chỉ kiểm file tồn tại). Đề xuất ghi sha256 của 4 file raw vào manifest | Chưa |
| S11 | **File thí nghiệm chỉ cần nằm trong worktree, chưa chắc đã được commit** (ví dụ `build/` bị ignore). Đề xuất đòi file nằm dưới `configs/`, hoặc đã được git track và chưa bị sửa | Có |
| S12 | **`train_loss_mean` gồm cả c_i và số hạng prox**, nên mất mát của DFedProx và size-corrected không so được với DFedAvg. Quyết định schema trước Tier 1 | Có (đúng như contracts.LocalTrainer định nghĩa) |

**Mục nhỏ:**
- ép kiểu số chỉ chạy ở đường YAML, nên `AlgorithmConfig(lr=1)` cho run_id khác `lr=1.0`;
- `local_only` không ghim `mixing`;
- `Popen` lỗi ở rank k thì các node trước đó không bị kill;
- log của node mất cùng thư mục tạm;
- không đặt `GLOO_SOCKET_IFNAME`;
- hằng số `THREAD_VARIABLES` và mã thoát lặp ở nhiều file;
- `describe --seeds 0` làm `fmean` lỗi;
- `--dry-run` không in code_version, root và trạng thái cây;
- `cpu_threads` của processes là số luồng của observer.

## Kích thước commit

| Commit | Task | Dòng | File |
|---|---|---|---|
| 99fc7d5 | T11a | 969 | 6 |
| a973a08 | T11b | 695 | 8 |
| 692a2a8 | T11d | 546 | 10 |
| 7e611a6 | T12 | 1,205 | 15 |
| 8e807f4 | T22b | 818 | 16 |

Các commit này vượt khoảng 300 dòng (SPEC §6) hay 5 file (plan K5); 191290d, c13686f, 48fbff4 vượt ngưỡng dòng.
- **T12** trộn sweep với các sửa ở `run_logger`, `manifest`, `config`, `run`.
- **T22b** trộn tái cấu trúc đường inprocess với runtime mới. P0-1, I6 và S1 đều nằm ở commit này. Lẽ ra nên tách đôi: tái cấu trúc không đổi hành vi, rồi mới tới process runtime.
