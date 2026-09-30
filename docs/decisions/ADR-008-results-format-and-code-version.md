# ADR-008: Định dạng kết quả và `code_version` — JSONL + manifest theo attempt, hash nội dung `src/` + file khóa, độ đo suy ra ở `analysis/`, hình chỉ sinh bằng script

## Status
Proposed — chờ Gate G1 (phụ thuộc SPEC QS-1, QS-4, QS-7).

## Date
2026-09-30 (cập nhật theo hai vòng phản biện P1.9 cùng ngày)

## Context
- Kết quả phải tái lập và kiểm toán được: từ một hình lần ra được `run_id`, cấu hình, seed, phiên bản code và máy (O1, SPEC US-6).
- Kết quả được dùng lại giữa các nhóm thí nghiệm (E4 dùng lại các ô của E1, E1c, E2), nhưng chỉ khi chúng do cùng một phiên bản code sinh ra (G20).
- Lưới khoảng 280 lượt (Tier 1 + 2) chạy trong nhiều đêm trên một máy chủ dùng chung (QĐ-2) và có thể bị ngắt. Phải chạy tiếp được mà không làm lại lượt đã xong, và phải giữ lại lượt lỗi.
- Tính bất định, như thời gian chạy, không được lẫn vào dữ liệu cần so khớp (G21, INV-6). PyTorch không đảm bảo kết quả trùng khớp giữa các phiên bản, nền tảng, hay giữa CPU và GPU (https://docs.pytorch.org/docs/2.14/notes/randomness.html), nên chỉ so khớp trên cùng máy (G13).
- Sửa code phân tích (ví dụ cách tính R@x, hay code đọc JSON) không được buộc phải chạy lại lưới.
- Một số lượt có thể phân kỳ (size-corrected ở β = 0.5, 2η ở E0b); phân kỳ là kết quả tất định, không phải lỗi.
- Nguồn: KH §4.7, §5.3, §10; SPEC §5.3, §9.3–§9.4, §11.2; H§2.4 (G13, G20, G21); `docs/decisions/REVIEW-P1.9.md`.

## Decision
- **Lưu trữ:** `results/runs/<run_id>/attempt-<k>/` gồm `manifest.json`, `rounds.jsonl`, `timing.jsonl`, `summary.json`; trường của từng file theo SPEC §5.3 và `src/dfl/contracts.py`. Mỗi nhóm thí nghiệm có chỉ mục `results/experiments/<exp>.json` liệt kê run_id của nó theo từng `code_version`; một `code_version` mới thêm mục mới, không xóa hay ghi đè mục cũ. Một run_id dùng chung được cho nhiều nhóm. `results/` không được commit (nằm trong `.gitignore`).
- **Cấu hình chuẩn hóa:** dataclass cấu hình đầy đủ, đã điền mặc định và kiểm kiểu (kể cả số đọc từ YAML), chỉ gồm tham số ảnh hưởng tới kết quả (gồm số luồng mỗi lượt và tập đánh giá; không gồm tên nhóm thí nghiệm, `--out`, `--workers`), ghi thành JSON khóa sắp xếp, số thực theo `repr`.
- **`code_version`** = sha256 của dãy cặp (đường dẫn tương đối, nội dung) của mọi file `*.py` trong `src/`, sắp theo đường dẫn, cùng `requirements.lock`. Đường dẫn nằm trong hash, nên chuyển code giữa các file cũng làm đổi nó. Trước khi hash, `\r\n` được đổi thành `\n`; các dòng của file khóa được bỏ khoảng trắng thừa và sắp xếp. Tính từ file trên đĩa nên không cần git và phản ánh cả thay đổi chưa commit. `.gitattributes` đặt `* text=auto eol=lf`, vì `eol` ghi đè `core.autocrlf` (https://git-scm.com/docs/gitattributes); nhờ vậy checkout trên Windows không làm cây bẩn.
- **File khóa** (SPEC QS-1): `requirements.lock` chỉ gồm gói *chạy*. Nó được sinh bằng `pip freeze --exclude-editable` trong một venv chỉ cài torch, torchvision và `pip install -e .`, không có `[dev]`. File được commit vào repo, tạo ở T02, và tạo lại mỗi khi đổi phụ thuộc chạy (Ask first). Cài đặt dùng `pip install -r requirements.lock`, rồi `pip install -e ".[dev]" -c requirements.lock`, nên công cụ dev không đổi được phiên bản gói chạy. Constraints file chỉ quyết định phiên bản, không làm gói được cài (https://pip.pypa.io/en/stable/user_guide/#constraints-files). Trên Python 3.12, `pip freeze` chỉ bỏ qua `pip` (https://pip.pypa.io/en/stable/cli/pip_freeze/), và venv 3.12 không cài sẵn setuptools.
- **`run_id`** = hash của (cấu hình chuẩn hóa + `code_version` + máy). "Máy" là tên CPU kèm hash của danh sách cờ CPU. Nhờ vậy, sau khi mất máy chủ hay khi máy ảo bị chuyển sang phần cứng khác, sweep không bỏ qua nhầm các lượt đã chạy ở máy cũ, và `analysis` không trộn chúng (G13). Manifest còn ghi `git_commit` (hoặc `none`) để check out lại khi tái lập.
- **Cây sạch** nghĩa là `git status --porcelain -- src configs requirements.lock` rỗng, tính cả file chưa track, và không có file `*.py` hay `*.yaml` nào trong `src/`, `configs/` bị `.gitignore` bỏ qua (`git status --porcelain --ignored`). File bị ignore không hiện trong `git status` nhưng vẫn vào `code_version`. `.gitignore` dùng mẫu có neo (`/data/`), để không trùng `src/dfl/data/`.
  - **Lưới chạy từ một checkout cố định:** sweep ghi vào `results/` chạy từ một worktree riêng, check out đúng commit của lưới (`git worktree add -d <đường dẫn> <commit>`, https://git-scm.com/docs/git-worktree), không từ cây làm việc hằng ngày. Sweep tính `code_version` một lần lúc bắt đầu. Mỗi tiến trình con tính lại, và nếu khác thì ghi attempt `failed` (lý do: `code_version` đổi giữa chừng); khi đó sweep dừng xếp lượt mới. Lý do: lưới chạy nhiều ngày trong khi nhóm vẫn commit vào cây làm việc.
  - `sweep` luôn đòi cây sạch, kể cả khi ghi ra `--out` (CLAUDE.md).
  - `run` chỉ được ghi vào `results/` khi cây sạch. `run --out` ra ngoài `results/` được phép khi cây bẩn hay không có git, và manifest ghi `dirty: true`.
  - Profile smoke không bao giờ ghi vào `results/`.
- **Môi trường:** trước khi chạy, cả `run` và `sweep` so từng gói trong `requirements.lock` với phiên bản đã cài, đọc bằng `importlib.metadata.version` (https://docs.python.org/3.12/library/importlib.metadata.html). Gói có trong môi trường mà không có trong file khóa, như công cụ dev, được bỏ qua. Ghi vào `results/` thì mọi gói phải khớp, nếu không thì từ chối. Ghi ra `--out` ngoài `results/` (smoke, demo trên laptop) thì chỉ cảnh báo, và manifest ghi `env_match: false`.
- **Attempt:**
  - Mỗi run_id có nhiều nhất một attempt đã xong (`ok` hoặc `diverged`). Sweep bỏ qua run_id đã xong; `run` ghi vào `results/` thì từ chối tạo thêm attempt.
  - Attempt không có `summary.json` (tiến trình bị ngắt) được giữ lại và tự chạy lại thành attempt mới.
  - Attempt `failed` được giữ lại nhưng không tự chạy lại. Với cùng `code_version` và cấu hình, lỗi thường lặp lại y hệt: vượt giới hạn lấy mẫu lại khi chia dữ liệu, cấu hình sai, lỗi code. Sửa code thì đã là run_id mới. Lỗi do môi trường (đầy đĩa, thiếu RAM) thì chạy lại bằng `sweep --retry-failed`.
  - Thư mục attempt được tạo nguyên tử. Tiến trình chạy một lượt giữ khóa `fcntl.flock(LOCK_EX | LOCK_NB)` trên `results/runs/<run_id>/.lock`, từ lúc chọn số attempt tới lúc ghi `summary.json`. Sweep giữ khóa trên `results/experiments/<exp>.lock` khi ghi chỉ mục. Nhờ vậy `--workers` chạy nhiều lượt cùng lúc, và sweep của các nhóm khác nhau (ví dụ E1 và E3a) chạy song song được. Một lượt dùng chung giữa hai sweep chỉ chạy ở một nơi; nơi kia thấy khóa đang giữ thì bỏ qua. Khóa tự nhả khi mọi file descriptor của nó đóng, kể cả khi tiến trình chết, nên không có khóa bỏ quên phải xóa tay trong `results/` (https://docs.python.org/3.12/library/fcntl.html#fcntl.flock, https://man7.org/linux/man-pages/man2/flock.2.html). `fcntl` chỉ có trên Unix, nên chỉ được import khi ghi vào `results/` (máy chủ chạy Linux). Laptop demo chỉ ghi ra `build/`, không khóa, nên chạy được cả trên Windows.
  - Khi `code_version` đổi, các lượt cũ thành "stale": không xóa, không dùng lại. Mọi `code_version` có lượt thuộc lưới được ghi vào `docs/experiments/CODE_VERSIONS.md`, kèm lý do khi bị thay thế (commit sửa lỗi và test tái hiện). `analysis.plots` báo lỗi khi thiếu dòng, và báo cáo có phụ lục liệt kê các `code_version` bị thay thế (SPEC INT-6).
- **Phân kỳ:** mất mát hay tham số không còn hữu hạn thì lượt dừng và ghi `SummaryDiverged` (round phân kỳ, lý do). Việc kiểm làm trước khi ghi bất kỳ bản ghi nào của round đó, nên không có NaN hay vô cực trong bản ghi. JSON ghi với `allow_nan=False`, để lỗi còn sót lại làm lượt `failed` rõ ràng thay vì ghi số sai.
- **Tách thời gian:** thời gian chỉ ghi ở `timing.jsonl`; `rounds.jsonl` là tất định.
- **Độ đo suy ra nằm ở `analysis/`:** R@x, AUC, độ dao động và mức sụt được tính từ `rounds.jsonl` trong `analysis/` (ngoài `src/`; mặc định là `analysis/convergence.py`, SPEC QS-4). `analysis` tự đọc JSON/JSONL và chỉ import `dfl.contracts`, nên sửa code phân tích không làm đổi `code_version`.
- **Hình chỉ sinh bằng script:** `python -m analysis.plots --results results/ --out figures/`, kèm `figures/PROVENANCE.json`. PROVENANCE ghi cho từng hình: lệnh, `code_version`, `analysis_version` (sha256 nội dung `analysis/*.py`, tính như `code_version`), commit, và run_id.
  - Lệnh từ chối ghi vào `figures/` khi `git status --porcelain -- analysis` không rỗng. Độ đo suy ra nằm ngoài `code_version`, nên phải có dấu vết riêng.
  - Analysis từ chối trộn nhiều `code_version` trong một hình hay một so sánh (trừ khi được cho phép tường minh). Nó từ chối lượt `dirty` hay `env_match: false`, so sánh có biến cố định khác nhau, và so sánh ghép cặp khác tập seed (SPEC INT-2).
  - Không có `--draft` thì `analysis.plots` chỉ đọc lượt trong `results/`. Chế độ `--draft` (smoke, demo) nhận lượt `dirty`, lượt lệch môi trường, lượt ngoài `results/` và lịch đánh giá riêng của profile; bỏ qua kiểm `CODE_VERSIONS.md`; gắn nhãn "DRAFT" lên mọi hình; và không được ghi vào `figures/` (SPEC INT-1).
- **Sao lưu:** nén và chép `results/` ra ngoài máy chủ sau mỗi nhóm thí nghiệm; không chạy `git clean -x` trong repo trên máy chủ.

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

### `code_version` = tree hash của commit (`git rev-parse HEAD:src`), bản trước của ADR này
- Ưu: có sẵn trong git, không đổi khi chỉ sửa tài liệu.
- Nhược: không phản ánh thay đổi chưa commit, nên một lượt chạy từ cây bẩn mang nhầm `code_version` của commit sạch; lỗi khi không có `.git` (bản giải nén, laptop demo).
- Kết luận: không chọn; thay bằng sha256 nội dung file trên đĩa.

### Lưu theo nhóm thí nghiệm (`results/<exp>/<run_id>/`)
- Ưu: nhìn thư mục là biết nhóm.
- Nhược: một ô dùng chung giữa hai nhóm phải chạy lại hoặc sao chép; không rõ `run_id` có phụ thuộc tên nhóm không.
- Kết luận: không chọn; lưu theo `run_id` kèm chỉ mục theo nhóm.

### Dùng `pyproject.toml` làm file khóa (phương án khác của QS-1)
- Ưu: không phải thêm file.
- Nhược: chỉ ghim phụ thuộc trực tiếp; mọi sửa cấu hình ruff, pytest hay coverage trong file này cũng làm đổi `code_version` và biến toàn bộ kết quả thành stale.
- Kết luận: không chọn làm mặc định; chờ nhóm quyết ở QS-1.

### `pip freeze` của venv phát triển làm file khóa (bản trước của ADR này)
- Ưu: một lệnh, không cần venv riêng.
- Nhược: gồm cả công cụ dev (pytest, ruff, import-linter, scikit-learn và phụ thuộc của chúng), vốn không ảnh hưởng kết quả. Nâng ruff sẽ đổi file khóa, đổi `code_version` và biến mọi kết quả thành stale, đúng nhược điểm đã loại `pyproject.toml`. So nguyên `pip freeze` với file khóa cũng làm sweep từ chối sau mỗi lần dựng lại venv, vì phụ thuộc bắc cầu không được ghim khi cài.
- Kết luận: không chọn; file khóa chỉ gồm gói chạy, kiểm từng gói bằng `importlib.metadata`.

### `code_version` theo từng module (chỉ hash các file mà lượt đó import)
- Ưu: thêm một file không liên quan (loader Fashion-MNIST, T5, TorchDistTransport) không làm lượt cũ thành stale.
- Nhược: `run_id` phải biết trước khi chạy, lúc chưa biết lượt sẽ import gì. Phần mở rộng hầu như luôn phải sửa file cũ để nối vào (`config.py`, `cli.py`), và khi đó lượt cũ vẫn thành stale. Việc xác định tập import cũng dễ sai.
- Kết luận: không chọn. Muốn so định lượng phần mở rộng với lưới lõi thì viết phần code đó trước E0a (SPEC §1.4).

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
- Sửa code phân tích rồi vẽ hình lại là rẻ.
- **Trade-off đã chấp nhận:** mọi thay đổi trong `src/`, kể cả docstring, làm đổi `code_version`.
  - Vì vậy code của lưới được đóng băng ở một commit từ E0a tới lượt Tier 2 cuối cùng: mọi sweep của lưới chạy từ worktree của commit đó. Nhánh chính vẫn nhận code mới (ví dụ T22), nhưng code đó không được dùng cho lượt nào của lưới.
  - Một bản sửa lỗi phát hiện ở Tier 2 buộc chạy lại mọi ô Tier 1 được dùng lại, và đó là cách làm đúng, vì lỗi có thể đã ảnh hưởng các ô đó. Chạy lại Tier 1 tốn khoảng 64 giờ-lõi, vượt dự phòng 30%; khi đó áp dụng thứ tự cắt Tier 2 (ADR-007).
  - Phần mở rộng sau M4 thêm code vào `src/` (T22, T5, E6) phải tự chạy các ô tham chiếu của mình dưới `code_version` mới, không so trực tiếp với lưới lõi. E5 chỉ đổi N trong cấu hình, nên dùng được `code_version` của lưới lõi.
  - Laptop demo có thể khác phiên bản gói so với máy chủ. Kết quả demo vì vậy mang `env_match: false` và chỉ vẽ được ở chế độ `--draft`.
- Chỉ so khớp từng byte trên cùng máy, cùng số luồng (REP-1); máy nằm trong `run_id`, manifest ghi máy (kèm hash cờ CPU), phiên bản thư viện và số luồng.
- Laptop demo cài không dùng `requirements.lock` (bản torch của nền tảng khác bản `+cpu` của máy chủ), nên mọi lượt demo có `env_match: false` và chỉ vẽ được bằng `--draft` (SPEC §4).
