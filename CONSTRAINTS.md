# CONSTRAINTS — ràng buộc chất lượng của dự án

Lần xem lại gần nhất: 2026-09-30 (P1.9, vòng phản biện 3), nhóm Đề 2; duyệt chính thức ở Gate G1.
File này là nguồn chuẩn. `Makefile` và `pyproject.toml` chỉ cài đặt lại các lệnh ghi ở đây; nếu hai nơi lệch nhau thì file này đúng.
Quy tắc cho mọi agent và thành viên: đọc file này trước khi viết code; **không bao giờ làm yếu nó để một thay đổi qua được kiểm tra** (CLAUDE.md).

## Lệnh theo chi phí

| Lệnh | Gồm | Ngân sách | Khi nào | Khi fail |
|---|---|---|---|---|
| `make check-fast` | sàn (`check-floor`) + `ruff check` + `ruff format --check` + `lint-imports` + unit test | vài giây | sau mỗi lần sửa | chặn |
| `make check-task` | như trên + integration test + coverage | 90 s | trước mỗi commit | chặn; vượt ngân sách thì chỉ cảnh báo |
| `make check-full` | như trên + mọi test (gồm e2e, smoke từ T11) + `pip check` | vài phút | mỗi checkpoint C1–C5 | chặn |
| `make check-inv` | T-INV-1…13 có đủ và xanh (T-INV-13 thêm ngày 2026-10-01, ADR-010) | — | từ C3, và trước mọi lượt Tier 1 | chặn |

Hiện trạng ngày 2026-09-30: `check-fast`, `check-task`, `check-full` xanh; `check-inv` báo thiếu cả 12 test (đúng dự kiến, vì chưa tới C3). Ngày 2026-10-01: thêm T-INV-13; `check-inv` vẫn báo thiếu test của các task chưa làm (đúng dự kiến trước C3).

## Sàn (luôn bắt buộc)

| ID | Ràng buộc | Kiểm bằng |
|---|---|---|
| F1 | Mọi test xanh | pytest trong mọi lệnh `check-*` |
| F2 | Không skip, xfail hay xóa test. Ngoại lệ phải có `lý do:` trên cùng dòng và một dòng trong bảng Ngoại lệ | `check-floor` (grep `pytest.skip`/`xfail`/`importorskip`, `pytest.mark.skip`/`skipif`/`xfail`) + ratchet số test F5 |
| F3 | Không suppression thiếu lý do: `# noqa`, `# type: ignore`, `# pragma: no cover` phải ghi mã luật cụ thể và `lý do:` trên cùng dòng | `check-floor` (grep) + ruff `PGH003`, `PGH004` (phải có mã luật), `RUF100` (noqa thừa) |
| F4 | Không để stub hay việc dở: không `NotImplementedError`; thân hàm chỉ có `...` chỉ được nằm ở `src/dfl/contracts.py`; không TODO/FIXME/XXX/HACK; không nuốt lỗi bằng except rỗng; không để code bị comment-out | `check-floor` (grep) + ruff `FIX`, `S110`, `S112`, `ERA001` |
| F5 | Số test thu thập được không giảm (ratchet, xem "Đo và giữ") | `check-floor`: so với `MIN_TESTS` trong Makefile |
| F6 | An toàn khi nạp dữ liệu: chỉ `yaml.safe_load`; không unpickle dữ liệu ngoài; không `torch.load(..., weights_only=False)` | ruff `S506`, `S301` + `check-floor` (grep `weights_only=False`) |
| F7 | Ngẫu nhiên tường minh, không dùng RNG toàn cục; chỉ `src/dfl/seeding.py` được tạo `Generator`/`SeedSequence`, nơi khác nhận chúng qua tham số (SPEC §6) | ruff `NPY002` (numpy.random kiểu cũ), `S311` (module `random`) + `check-floor` (grep `torch.manual_seed` trong `src/`; grep `default_rng(`, `SeedSequence(`, `Generator(` ngoài `seeding.py`) + T-INV-5 từ T11: `torch.get_rng_state()` không đổi sau một lượt, để bắt lời gọi RNG của torch thiếu `generator=` mà grep không thấy |
| F8 | Không làm yếu ngưỡng: không hạ số trong file này, không bỏ check khỏi Makefile, không nới cấu hình ruff/coverage/import-linter trong `pyproject.toml` | review: `git diff <commit trước task> -- CONSTRAINTS.md Makefile pyproject.toml` (xem "Canh gác khi review") |
| F9 | Không commit dữ liệu MNIST, `results/` hay secret/token | `.gitignore` + review (chưa có công cụ quét secret; dự án không dùng dịch vụ cần khóa) |

Luật ruff `S311` vốn cảnh báo về mật mã; ở đây nó được dùng để cấm module `random` chuẩn, vì mọi nguồn ngẫu nhiên phải là `Generator` có seed con (G9, INV-6).

## Ngưỡng có số

| Chiều | Luật | Kiểm bằng | Chạy ở | Vì sao chọn số này |
|---|---|---|---|---|
| Lint + format | 0 lỗi ruff, đúng định dạng `ruff format` | `ruff check .` và `ruff format --check .` | check-fast, check-task, check-full | Bộ luật E, F, W, I, B, UP cộng các luật sàn ở trên; 0 lỗi là mức đạt được ngay hôm nay |
| Ranh giới module | 0 contract import bị phá: `layers`, `no-analysis-in-src`, `analysis-reads-only`, `pure-no-torch`, `no-dev-deps` | `lint-imports` (import-linter 2.15, cấu hình `[tool.importlinter]` trong `pyproject.toml`) | check-fast, check-task, check-full | Capability map và hướng phụ thuộc một chiều của SPEC §3.2; ENG-6, ENG-7. `no-dev-deps` cấm `dfl` và `analysis` import scipy, sklearn, joblib, pytest, vì các gói này không nằm trong `requirements.lock` (P1.9). Đã thử phá từng contract, cả 5 đều bắt được |
| Coverage | ≥ 80% dòng cho `src/dfl` (trừ `cli.py`) và `analysis/` | `pytest --cov` với `fail_under = 80` trong `[tool.coverage.report]` | check-task, check-full | 80% đủ cao để buộc có test, vẫn chừa chỗ cho dòng cấu hình (KH §5.1). `analysis/` tính vào vì R@x, AUC, CI, kiểm định H2 quyết định kết luận (nhóm chốt QS-5 ở P1.8) |
| Test bất biến | T-INV-1…13 có đủ (tên `test_t_inv_<k>_<hành vi>`) và xanh | `make check-inv` | từ C3 (T11); trước mọi lượt Tier 1 | Lỗi mô phỏng làm sai mọi kết luận (KH §9); SPEC §9.2 |
| Smoke e2e | `tests/e2e/test_smoke_run.py` chạy `python -m dfl run` với profile smoke và kiểm đủ 4 file đúng schema | pytest trong check-full | từ T11 | ENG-4; bắt lỗi ghép nối mà unit test không thấy |
| Phụ thuộc | Không có phụ thuộc hỏng | `python -m pip check` | check-full | Kiểm chứng từ metadata bên ngoài dự án |

**Ngân sách thời gian (T1 — chỉ cảnh báo, không chặn):** check-task dưới 90 s; smoke e2e dưới 60 s trên máy chủ. `check-task` in thời gian chạy và cảnh báo khi vượt; `check-full` in 5 test chậm nhất (`--durations=5`). Chỉ cảnh báo vì SPEC §7 cấm test phụ thuộc thời gian thực, và laptop chậm hơn máy chủ. Xem lại ở mỗi checkpoint trên máy chủ.

## Toàn vẹn nghiên cứu

| ID | Ràng buộc | Kiểm bằng | Có hiệu lực từ |
|---|---|---|---|
| R1 | Hình và bảng chỉ sinh bằng `python -m analysis.plots --results results/ --out figures/`; không sửa tay `figures/`. Lệnh từ chối ghi vào `figures/` khi `analysis/` có thay đổi chưa commit; hình `--draft` (smoke, demo) không bao giờ vào `figures/` | `analysis.plots` ghi `figures/PROVENANCE.json` (lệnh, `code_version`, `analysis_version`, commit, run_id nguồn của từng hình) + unit test của lệnh từ chối + review | T16 |
| R2 | Không sửa tay `results/` | REP-1: chạy lại 2 run_id ngẫu nhiên phải trùng từng byte + review | T11 |
| R3 | Mọi lượt có manifest (`code_version`, `git_commit`, `dirty`, `env_match`, cấu hình, seed, máy, số luồng); `run` chỉ ghi vào `results/` khi cây sạch và mọi gói trong `requirements.lock` khớp phiên bản đã cài; `sweep` luôn đòi cây sạch, và khi ghi vào `results/` thì chạy từ worktree riêng check out đúng commit; tiến trình con kiểm lại `code_version` | Test integration của sweep và `run` (SPEC ENG-9, ENG-10) | T11, T12 |
| R4 | Không xóa lượt lỗi; lượt phân kỳ (`SummaryDiverged`) được giữ và báo cáo "k/n seed phân kỳ", không bị lọc khỏi thống kê (không dùng `skipna` mặc định của pandas) | `analysis` kiểm số attempt của mỗi run_id liên tục, báo số attempt `failed`/`diverged`, và báo lỗi khi gặp NaN (SPEC INT-3) | T15 |
| R5 | Không trộn nhiều `code_version`, hay các lượt có biến cố định khác nhau, trong một hình hoặc một so sánh | `analysis` báo lỗi, có unit test (SPEC INT-2) | T15 |
| R6 | Siêu tham số, lịch đánh giá và ngưỡng đã chốt chỉ đổi qua ADR mới, và phải chạy lại cả nhóm thí nghiệm bị ảnh hưởng. Không bao giờ đổi chúng vì kết quả trên tập test, kể cả kết quả kiểm độ nhạy η/2, 2η của E0b. Trước E0b, ADR mới được dựa trên bằng chứng trên validation; từ E0b trở đi chỉ được dựa trên lỗi có test tái hiện (không chạy validation mới sau khi đã thấy kết quả test) | Review diff của `configs/`, `docs/decisions/`, file này | ngay |
| R7 | Sanity trên kết quả thật phải đạt trước khi chạy E1 (SAN-1…3) và E2 (SAN-4): SAN-1 B1 tập trung ≥ 98.5%; SAN-2 DFedAvg-FC-IID kém B1-4k (B1 với η chung, 4,000 bước) ≤ 1.0 điểm %; SAN-3 B2 ở α = 0.1 thấp hơn B2 ở IID ≥ 20 điểm %; SAN-4 chênh α = 100 với IID ≤ 1.0 điểm %; SAN-5 (cấu trúc, kiểm trên `rounds.jsonl` của mọi lượt): FC + MH và đường server có `consensus_distance` ≤ 1e-6 × `drift_pre_combine` và `acc_std` = 0, lượt dùng W đối xứng ngẫu nhiên kép có `consensus_distance` ≤ `drift_pre_combine`, lượt DFL có message, B1 và B2 không có. Không đạt thì chỉ được sửa lỗi có test tái hiện; không đổi siêu tham số, mô hình hay ngưỡng để qua SAN (tránh chọn trên tập test). Nếu debug không tìm ra lỗi (mọi T-INV xanh, không có test tái hiện nào fail): dừng và báo nhóm; chỉ nhóm được quyết định chạy tiếp, và báo cáo phải ghi SAN không đạt kèm số đo | Lệnh kiểm sanity trong `analysis`, đọc `results/` và trả mã lỗi khác 0 nếu không đạt (tạo ở T14/T15) | T14 |
| R8 | Thẻ tiền đăng ký `docs/experiments/E<k>.md` được commit trước khi chạy nhóm thí nghiệm đó | Review (SPEC EXP-1) | T14 |
| R9 | Mọi `code_version` có lượt trong `results/runs/` có một dòng trong `docs/experiments/CODE_VERSIONS.md` (ngày, commit, nhóm thí nghiệm; nếu bị thay thế thì lý do, commit sửa lỗi và test tái hiện) | `analysis.plots` báo lỗi khi thiếu dòng, có unit test (SPEC INT-6) | T15 |

Vì sao chọn các số của R7 (nhóm chốt QS-3 ở P1.8): SmallCNN trên MNIST khi huấn luyện tập trung thường đạt khoảng 99%; sai số của tập 10k ảnh ở mức 99% chỉ khoảng 0.1 điểm %, nên 1.0 điểm % đủ rộng để không dừng oan vì nhiễu seed. Local-only ở α = 0.1 (khoảng 2.5 lớp mỗi node) phải kém xa IID; nếu chênh chưa tới 20 điểm % thì nhiều khả năng dữ liệu rò rỉ giữa các node hoặc chia dữ liệu sai.

**Đối chứng của SAN-2 (nhóm chốt ngày 2026-09-30, P1.9, QS-3):** B1-4k thay cho B1, giữ ngưỡng 1.0 điểm %. B1 có η riêng và ngân sách gấp khoảng 4 lần DFedAvg-FC-IID, nên khoảng cách tới B1 có thể vượt 1.0 điểm % mà không có lỗi nào. B1-4k cùng η chung, cùng B = 32 và cùng 4,000 bước, nên kém nó hơn 1.0 điểm % mới là dấu hiệu lỗi. Khoảng cách tới B1 vẫn được báo cáo, nhưng không phải điều kiện dừng. Đây là một lần nới đối chứng có quyết định của nhóm, ghi trước khi có bất kỳ kết quả nào (SPEC §8, Ask first).

## Ràng buộc có ý kiến bên ngoài

Các kiểm tra sau không do chính test của dự án tự chấm:
- R7 (SAN-1…4): so với mức đã biết của MNIST.
- T-INV-10: macro-F1 so với scikit-learn.
- T-INV-8: kết quả chia dữ liệu so với hàm tham chiếu sao từ thuật toán NIID-Bench.
- T-INV-9: spectral gap so với công thức giải tích.
- Docker (tùy chọn, T22c, ADR-010): số byte mạng của container theo `docker stats` so với số byte của CommLedger, ở `make check-docker` và 4 ô Docker của E8.
- `pip check`: metadata của các phụ thuộc. Đây là kiểm tra duy nhất chạy được ngay hôm nay; các kiểm tra còn lại có hiệu lực theo task của chúng.

## Đo và giữ (ratchet)

| Chỉ số | Giá trị đo ở C3 (2026-10-02; P1.9 ngày 2026-09-30 là 34 test, 100%, 1 s) | Hướng | Kiểm bằng |
|---|---|---|---|
| Số test thu thập được | 611 | Không giảm. `MIN_TESTS` trong Makefile chỉ được tăng, cập nhật ở mỗi checkpoint | `check-floor` (F5) |
| Coverage tổng (`src/dfl` + `analysis/`) | 95.2% dòng (94.6% dòng + nhánh). Hai module chạy trong tiến trình con không được đo: `process_runtime.py` 65%, `torchdist.py` 61% | Không dưới 80%. Ghi lại giá trị ở mỗi checkpoint | `check-task` |
| Thời gian check-task | 66–134 s tùy tải máy chủ (tải 11–18 trên 8 lõi); khoảng 65 s là test chạy tiến trình con thật | ≤ 90 s (chỉ cảnh báo) | `check-task` |

## Kích hoạt theo task

Chỉ được thêm hoặc thắt chặt, không được nới.

| Khi | Thay đổi |
|---|---|
| T02 | Tạo `requirements.lock` (chỉ gói chạy, SPEC §4) và `.gitattributes` (`* text=auto eol=lf`); cập nhật lệnh cài đặt dùng `-r` và `-c requirements.lock` |
| T04 | Thêm `dfl.data.partition`, `dfl.data.heterogeneity` vào `source_modules` của contract `pure-no-torch`. Không thêm sớm được, vì import-linter báo lỗi khi `source_modules` có module chưa tồn tại |
| T11 (C3) | Tạo `tests/e2e/test_smoke_run.py`; đổi `check-full: check-static` thành `check-full: check-static check-inv` |
| T12 | `run` và `sweep` kiểm môi trường theo `requirements.lock`; khóa `flock` theo từng run_id; sweep vào `results/` chạy từ worktree riêng và tiến trình con kiểm lại `code_version` (R3) |
| T14 | Có lệnh kiểm sanity R7 (SAN-1…5); chạy `make check-inv` trước E0a |
| T15 | `analysis` kiểm R4, R5, R9 |
| T16 | `figures/PROVENANCE.json` có `analysis_version`; lệnh từ chối khi `analysis/` bẩn; chế độ `--draft` (R1) |
| T22b | Có `test_t_inv_13_…` (process runtime trùng từng byte với bộ mô phỏng); `make check-inv` đếm tới 13 (Makefile đã đổi ngày 2026-10-01) |
| T22c (tùy chọn) | Thêm `make check-docker` vào bảng lệnh ở đầu file: chạy tay, ghi kết quả vào `docs/DOCKER.md` |
| Mỗi checkpoint | Tăng `MIN_TESTS` lên bằng số test hiện có; ghi lại coverage |

## Ngoại lệ

Mỗi ngoại lệ cần người chịu trách nhiệm và ngày hết hạn, tối đa 90 ngày.

| ID | Luật | Đường dẫn | Lý do | Người chịu trách nhiệm | Hết hạn |
|---|---|---|---|---|---|
| — | (chưa có) | | | | |

## Canh gác khi review

Mỗi review (và mỗi lần một agent báo "xong") phải xem `git diff` của task để tìm năm kiểu "đi đường tắt cho xanh":
1. **Ngưỡng bị dời:** số trong file này, `fail_under`, `MIN_TESTS`, luật ruff hay contract import bị hạ hoặc bị xóa.
2. **Test bị làm dễ đi:** thêm skip/xfail, xóa file test, rút bớt assert trong test còn giữ lại.
3. **Checker bị tắt:** thêm `noqa`, `type: ignore`, `pragma: no cover`, `ignore_imports` trong cấu hình import-linter.
4. **Việc dở:** stub `NotImplementedError`, `...`, except rỗng, TODO.
5. **Ngoại lệ mới xuất hiện:** một dòng mới trong bảng Ngoại lệ mà nhóm chưa bàn.

Thắt chặt thì có thể làm lặng lẽ; nới lỏng thì phải nói rõ và có quyết định của nhóm (SPEC §8, Ask first).
