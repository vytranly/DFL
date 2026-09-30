# ADR-008: Định dạng kết quả và `code_version` — JSONL + manifest theo attempt, hash nội dung `src/` + file khóa, độ đo suy ra ở `analysis/`, hình chỉ sinh bằng script

## Status
Proposed — chờ Gate G1 (phụ thuộc SPEC QS-1, QS-4, QS-7).

## Date
2026-09-30

## Context
- Kết quả phải tái lập và kiểm toán được: từ một hình lần ra được `run_id`, cấu hình, seed, phiên bản code và máy (O1, SPEC US-6).
- Kết quả được dùng lại giữa các nhóm thí nghiệm (E4 dùng lại các ô của E1, E1c, E2), nhưng chỉ khi chúng do cùng một phiên bản code sinh ra (G20).
- Lưới khoảng 263 lượt chạy trong nhiều đêm (QĐ-2) và có thể bị ngắt. Phải chạy tiếp được mà không làm lại lượt đã xong, và phải giữ lại lượt lỗi.
- Tính bất định, như thời gian chạy, không được lẫn vào dữ liệu cần so khớp (G21, INV-6). PyTorch không đảm bảo kết quả trùng khớp giữa các phiên bản, nền tảng, hay giữa CPU và GPU (https://docs.pytorch.org/docs/2.14/notes/randomness.html), nên chỉ so khớp trên cùng máy (G13).
- Sửa code phân tích (ví dụ cách tính R@x) không được buộc phải chạy lại khoảng 44–57 giờ-CPU thí nghiệm.
- Nguồn: KH §4.7, §5.3, §10; SPEC §5.3, §9.3–§9.4, §11.2; H§2.4 (G13, G20, G21).

## Decision
- **Lưu trữ:** `results/<exp>/<run_id>/attempt-<k>/` gồm `manifest.json`, `rounds.jsonl`, `timing.jsonl`, `summary.json`. Trường của từng file theo SPEC §5.3, gồm cả các trường macro-F1 theo Q3. `results/` không được commit (nằm trong `.gitignore`).
- **`code_version`** = hash nội dung: tree hash của `src/` (`git rev-parse HEAD:src`) cùng hash của file khóa phụ thuộc. File khóa mặc định là `requirements.lock`, sinh bằng `pip freeze --exclude-editable` trên máy chủ, commit vào repo và tạo lại mỗi khi đổi phụ thuộc (SPEC QS-1).
- **`run_id`** = hash của (cấu hình chuẩn hóa + `code_version`).
- **Chỉ ghi vào `results/` khi cây làm việc sạch:** `run` và `sweep` từ chối chạy khi `src/`, `configs/` hay file khóa có thay đổi chưa commit. Ghi ra `--out` ngoài `results/` (ví dụ `build/`) thì được, và manifest ghi `dirty: true` (SPEC QS-7).
- **Attempt và idempotent:** sweep bỏ qua `run_id` đã có một attempt `ok`. Attempt lỗi được giữ lại với `status: failed` và chạy lại thành attempt mới. Khi `code_version` đổi, các lượt cũ thành "stale": không xóa, không dùng lại.
- **Tách thời gian:** thời gian chỉ ghi ở `timing.jsonl`; `rounds.jsonl` là tất định.
- **Độ đo suy ra nằm ở `analysis/`:** R@x, AUC, độ dao động và mức sụt được tính từ `rounds.jsonl` trong `analysis/` (ngoài `src/`; mặc định là `analysis/convergence.py`, SPEC QS-4), nên sửa chúng không làm đổi `code_version`.
- **Hình chỉ sinh bằng script:** `python -m analysis.plots --results results/ --out figures/`. Analysis từ chối trộn nhiều `code_version` trong một hình hay một so sánh (trừ khi được cho phép tường minh), và từ chối so sánh khi các biến cố định khác nhau (SPEC INT-2).

## Alternatives Considered

### Ghi tay vào Excel hoặc notebook
- Ưu: quen thuộc, nhanh lúc đầu.
- Nhược: không kiểm toán được; dễ bị sửa tay; không tái lập.
- Kết luận: không chọn.

### `run_id` chỉ tính theo cấu hình
- Ưu: đơn giản.
- Nhược: không phát hiện được kết quả do code cũ sinh ra; dễ trộn kết quả của hai phiên bản code.
- Kết luận: không chọn (G20).

### `code_version` = HEAD commit
- Ưu: có sẵn.
- Nhược: đổi sau mỗi commit tài liệu, thẻ thí nghiệm hay code phân tích, buộc chạy lại vô ích.
- Kết luận: không chọn.

### Dùng `pyproject.toml` làm file khóa (phương án khác của QS-1)
- Ưu: không phải thêm file.
- Nhược: chỉ ghim phụ thuộc trực tiếp; mọi sửa cấu hình ruff, pytest hay coverage trong file này cũng làm đổi `code_version` và biến toàn bộ kết quả thành stale.
- Kết luận: không chọn làm mặc định; chờ nhóm quyết ở QS-1.

### Đặt độ đo suy ra trong `src/dfl/metrics/` (cây thư mục ban đầu ở KH §4.6)
- Ưu: mọi độ đo nằm ở một chỗ.
- Nhược: sửa cách tính R@x hay AUC làm đổi `code_version`, buộc chạy lại toàn bộ lưới.
- Kết luận: không chọn (QS-4).

### Công cụ theo dõi thí nghiệm (MLflow, Weights & Biases) hoặc cơ sở dữ liệu
- Ưu: có giao diện xem kết quả.
- Nhược: thêm phụ thuộc hoặc dịch vụ ngoài; nặng hơn nhu cầu; dashboard nằm ngoài phạm vi (QĐ-5).
- Kết luận: không chọn.

## Consequences
- Tái lập và kiểm toán được (SPEC REP-1, INT-1…3); chạy lại được lượt lỗi; dùng lại kết quả an toàn giữa các nhóm thí nghiệm.
- Quy trình bắt buộc: commit trước khi chạy sweep. Đổi phụ thuộc thì tạo lại `requirements.lock` và chấp nhận mọi kết quả cũ thành stale.
- Sửa code phân tích rồi vẽ hình lại là rẻ. Sửa `src/` sau khi đã chạy Tier 1 thì phải chạy lại mọi nhóm thí nghiệm bị ảnh hưởng.
- Chỉ so khớp từng byte trên cùng máy, cùng số luồng (REP-1); manifest ghi máy, phiên bản thư viện và số luồng.
- `results/` chỉ nằm trên máy chủ (không commit), nên nhóm cần sao lưu định kỳ, ví dụ nén `results/` sau mỗi nhóm thí nghiệm.
