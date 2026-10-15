# PERF — đo hiệu năng và chốt giao thức (T13)

- **Ngày đo:** 2026-10-15, trên máy chủ thí nghiệm: Intel Xeon Platinum 8176, 8 nhân, 31 GB RAM, không GPU.
- **Code:** nhánh `binh`, sau commit `7bac5b0` (T11e). torch 2.14.0+cpu, Python 3.12.3. Mọi lượt dùng 1 luồng torch, trừ bảng số luồng.
- **Tải:**
  - Load average lúc đo là 12–41, nhưng phần lớn do tiến trình chờ đĩa (`du`, `rsync` ở trạng thái D). `vmstat` cho thấy CPU người dùng chỉ 2–3%, iowait 38–61%.
  - Số lõi rảnh cho tính toán, đo qua `/proc/stat` (idle + iowait) mỗi lần 30–60 s, là **6.7–7.8 / 8** suốt chiến dịch.
  - Ước tính ngày 2026-09-30 (khoảng 1 lõi rảnh) là ở một thời điểm khác. Tải của máy chủ dùng chung thay đổi, nên trước mỗi sweep lớn phải đo lại (lệnh ở cuối).
- Mọi lượt ghi ra `build/perf/`, không lượt nào vào `results/`.

## 1. Một bước huấn luyện (B = 32), ms

Trung vị của 5 × 50 bước, dữ liệu tổng hợp cùng shape (thời gian không phụ thuộc giá trị điểm ảnh). Tải lúc đo 14.7–17.3.

| Model | 1 luồng | 2 luồng | 4 luồng | 8 luồng |
|---|---|---|---|---|
| logreg (M0) | 0.52 | 0.49 | 0.50 | 1.96 |
| smallcnn (M1, lõi) | 9.16 | 5.91 | 4.73 | 3.79 |
| mlp2nn (M2) | 1.36 | 1.12 | 1.04 | 1.02 |
| cnn_mcmahan (M3) | 43.77 | 23.68 | 17.37 | 88.48 |
| cnn_niidbench + crop_flip (CIFAR-10) | 9.43 | 5.95 | 4.99 | 46.25 |

- **Số luồng:** tính theo lõi, 1 luồng hiệu quả nhất. SmallCNN tốn 9.2 lõi·ms mỗi bước với 1 luồng, nhưng 30.3 lõi·ms với 8 luồng; M3 và CIFAR-10 với 8 luồng còn chậm hơn 1 luồng vì tranh lõi.
- **Kết luận:** giữ `threads: 1` mỗi lượt và tăng số lượt song song (worker).

## 2. Đánh giá, s

SmallCNN, MNIST thật, 1 luồng.

| Phép đánh giá | N = 20 (21 model) | E8, N = 8 (9 model) |
|---|---|---|
| Một mốc trên tập con 2,000 ảnh test | 9.04 | M0 0.01; M1 4.29; M2 0.12; M3 19.32 |
| Đánh giá cuối trên 10,000 ảnh | 46.14 | M1 khoảng 21.5; M3 khoảng 96.6 (5 lần một mốc) |

## 3. Một lượt và số lượt song song (worker)

Profile `fast` (N = 20, 1,000 bước, τ = 10, ring, IID), seed khác nhau, chạy cùng lúc bằng `python -m dfl run`, mỗi lượt 1 luồng.

| Worker | Thời gian thực của lô | Thời gian trong lượt (timing.jsonl) | Huấn luyện | Đánh giá |
|---|---|---|---|---|
| 1 | 343 s | 261 s | 169 s | 89 s |
| 4 | 363 s | 278 s (+7%) | 185 s | 90 s |
| 7 | 393 s | khoảng 300 s (+15%) | khoảng 199 s | khoảng 96 s |

- **Thông lượng:** 7 worker cho khoảng 6.1 lần thông lượng của 1 worker.
- **Phần ngoài timing:** khoảng 80 s mỗi lượt, gồm nạp torch và MNIST, ghi manifest và đánh giá cuối 46 s.
- **RAM:** một lượt N = 20 inprocess cần 0.86 GB, nên 7 worker cần khoảng 6 GB.

**Lượt full** (`configs/profiles/full.yaml`: MNIST, N = 20, 4,000 bước, τ = 20, ring, IID, η = 0.05 giữ chỗ), chạy một mình:

| Thời gian thực | Trong lượt | Huấn luyện | Đánh giá (40 mốc) | Tổng hợp | RAM đỉnh |
|---|---|---|---|---|---|
| 1,153 s = **19.2 phút-lõi** | 1,102 s | 724 s | 366 s | 12 s | 0.86 GB |

Con số này khớp ước tính 20 phút-lõi của ADR-007. Acc(w̄) cuối là 99.03%, nhưng chỉ là số đo phát triển, không phải kết quả.

**τ = 1 so với τ = 20** (N = 20, 200 bước, 2 mốc):

| τ | Huấn luyện | Tổng hợp | Thời gian trong lượt |
|---|---|---|---|
| 1 (200 round) | 38.3 s | 6.4 s | 63.7 s |
| 20 (10 round) | 35.0 s | 1.3 s | 53.9 s |

Mỗi round tốn thêm khoảng 32 ms tổng hợp, cộng chi phí gọi trainer. Vì vậy một lượt D-PSGD đủ 4,000 round tốn khoảng **22.5 phút-lõi** (+17%).

## 4. Process runtime (ADR-010), N = 8

Profile `fast` với N = 8, 200 bước, τ = 10.

- **Thời gian:** processes 44 s, inprocess 48 s.
- **Trùng từng byte:** `rounds.jsonl` của hai runtime trùng nhau (T-INV-13 trên MNIST thật).
- **RAM:** RAM đỉnh cộng của 8 tiến trình node là 3,953 MiB, khoảng 0.49 GB mỗi node. Từ T11e, node không nạp ảnh test.
- **N = 20:** cần khoảng 10 GB, vừa máy chủ (khoảng 20 GB trống). Với CIFAR-10 thì cần khoảng 21 × 1.17 GB (ADR-012), không vừa.

**E8** (R = 50, τ = 20, tức 1,000 bước; 10 mốc). Ước tính từ §1 và §2; khởi động khoảng 6 s mỗi tiến trình.

| Model | Phút-lõi một lượt | Thời gian thực một lượt | 18 lượt |
|---|---|---|---|
| M0 | khoảng 1 | khoảng 0.3 phút | 18 phút-lõi |
| M1 | khoảng 3.2 | khoảng 1.5 phút | 58 |
| M2 | khoảng 1.1 | khoảng 0.3 phút | 20 |
| M3 | khoảng 11.6 | khoảng 5.7 phút (đánh giá chiếm khoảng 5 phút) | 209 |

E8 tổng cộng khoảng **5 giờ-lõi**, khoảng 2.3 giờ thực nếu chạy lần lượt, khớp ADR-011 (1.5–3.5 giờ). **Giữ R = 50.**

## 5. Ngân sách Tier 1 + 2 theo số đo mới

Phút-lõi mỗi lượt:
- DFL τ = 20 (cả B2, B3, đường server): 19.2.
- τ = 1: 22.5.
- τ = 5: khoảng 19.7.
- τ = 50: khoảng 19.
- DFedProx: khoảng 21 (thêm số hạng prox; ước tính).
- B1: khoảng 4.4 (17,190 bước, 172 mốc của một model).
- B1-4k: khoảng 1.4.

| Nhóm | Lượt | Phút-lõi | Giờ-lõi |
|---|---|---|---|
| E0a | 24 (9 τ = 20, 6 τ = 1, 6 τ = 50, 3 B1) | 435 | 7.3 |
| E0b | 43 | 728 | 12.1 |
| E1 | 70 | 1,344 | 22.4 |
| E2 | 30 | 576 | 9.6 |
| E3a | 24 (12 D-PSGD, 12 DFedProx) | 522 | 8.7 |
| E4a | 6 | 115 | 1.9 |
| **Tier 1, lưới lõi** | **197** | **3,720** | **62.0** |
| E8 | 72 | 305 | 5.1 |
| Tier 2: E1c 15, E2b 9, E3b 42, E4 15 | 81 | 1,591 | 26.5 |
| Có điều kiện (12 τ = 1, 12 τ = 50) | ≤ 24 | 498 | 8.3 |
| **Tổng** | **≤ 374** | **6,206** | **103.4** (kể cả 30% dự phòng: khoảng 134) |

**Thời gian thực** = giờ-lõi / (worker / 1.15), với 1.15 là độ chậm thêm đo được khi 7 worker chạy cùng lúc. E8 chạy riêng, vì mỗi lượt dùng tới 8 tiến trình.

| Lõi rảnh | Worker | Tier 1 (lõi) + E8 | Tier 1 + 2 + có điều kiện | Kể cả 30% dự phòng |
|---|---|---|---|---|
| khoảng 7 (đo hôm nay) | 6 | khoảng 12 h + 2.3 h | khoảng 20 h | khoảng 26 h |
| khoảng 1 (đo 2026-09-30) | 1 | khoảng 2.6 ngày + 2.3 h | khoảng 4.4 ngày | khoảng 5.7 ngày |

- **Kết luận:** Tier 1 + 2 vừa 3 tuần (tuần 8–10) ở cả hai mức tải. Với khoảng 7 lõi rảnh, còn đủ chỗ chạy lại toàn bộ Tier 1 một lần nếu phải sửa lỗi sau E0b.
- **Không cần cắt theo ADR-007:** giữ lịch đánh giá mỗi 100 bước, tập con 2,000 ảnh và `threads: 1` như `configs/base.yaml`. Không đổi gì nên không cần ADR mới.
- **Khuyến nghị cho sweep:** `--workers 6`, chừa một lõi cho hệ thống và cho các thành viên khác. Đo lõi rảnh ngay trước sweep, rồi đặt worker bằng số lõi rảnh trừ 1.

## 6. Laptop demo (QS-8)

Câu hỏi O6 vẫn mở: laptop demo chạy hệ điều hành gì? Việc cài thử và chạy smoke trên laptop phải làm trên chính laptop đó. Các bước dưới đây không cần Makefile, nên chạy được trên cả Windows:

```bash
git clone https://github.com/vytranly/DFL.git && cd DFL && git checkout binh
python3.12 -m venv .venv
. .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install torch==2.14.0 torchvision==0.29.0 && pip install -e .   # laptop không dùng requirements.lock (SPEC §4)
python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke   # không cần mạng
python -m dfl download mnist                                               # cần mạng, một lần
python scripts/perf_t13.py laptop
```

`laptop` in ra máy và phiên bản, ms một bước, giây một mốc đánh giá, và số bước demo lớn nhất để 2 lượt N = 20 (ring và FC) cộng 60 s cố định vừa 5 phút. Kết quả khi chạy trên máy chủ:

| Một bước | Một mốc (2,000 ảnh) | Đánh giá cuối (10,000 ảnh) | Số bước demo | Ước tính tổng |
|---|---|---|---|---|
| 10.80 ms | 9.67 s | 48.35 s | **200** | 282 s |

**Phát hiện cần nhóm quyết ở T26:** QS-8 dự kiến khoảng 400 bước, nhưng ngay trên máy chủ, 5 phút chỉ đủ khoảng 200 bước. Nguyên nhân là đánh giá cuối trên đủ 10,000 ảnh test mất khoảng 48 s mỗi lượt. Các phương án:
- demo 200 bước;
- demo dài hơn 5 phút;
- demo đánh giá cuối trên tập nhỏ hơn: đổi hành vi REQ-03 cho demo, thuộc Ask first.

Chốt sau khi có số đo trên laptop thật.

## 7. Lệnh chạy lại

```bash
.venv/bin/python scripts/perf_t13.py cores --seconds 60    # lõi rảnh (Linux)
.venv/bin/python scripts/perf_t13.py step                  # bảng §1
.venv/bin/python scripts/perf_t13.py eval                  # bảng §2, cần MNIST
mkdir -p build/perf && bash scripts/perf_t13_campaign.sh > build/perf/campaign.log 2>&1   # §3, §4 (khoảng 45 phút)
```

`scripts/perf_t13_campaign.sh` chạy profile `fast` với 1, 4, 7 lượt song song, một lượt `full`, rồi `fast` với N = 8 ở hai runtime (kèm `cmp` rounds.jsonl và RAM đỉnh của các tiến trình node). Thời gian τ = 1 và τ = 20 ở §3 đo bằng hai lượt `fast` với `local_steps: 200` và `algorithm.tau` tương ứng.
