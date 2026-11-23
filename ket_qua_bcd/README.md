# Kết quả EXP B, C, D (MNIST và CIFAR-10)

Thư mục này chứa kết quả của EXP B (quantity skew), EXP C (label + quantity skew) và EXP D (topology), chạy trên cả MNIST và CIFAR-10 với 10 client và 50 round. Có hai lần chạy, khác nhau ở cách khởi tạo trọng số. Nhận xét chi tiết ở [NHAN_XET.md](NHAN_XET.md).

| Thư mục | Code | Khởi tạo | Máy chạy | Ngày |
|---|---|---|---|---|
| `khoi_tao_chung/cifar` | `86b067d` + 2 chỗ sửa tại chỗ ([patch](khoi_tao_chung/code_86b067d_sua_tai_cho.patch)) | mọi client cùng một `w0` | b1: RTX 3060 desktop, chạy chung GPU với một việc khác; d5: RTX 3060 desktop, chạy một mình; còn lại: RTX 3060 Laptop | 13–14/11/2026 |
| `khoi_tao_chung/mnist` | như trên | cùng `w0` | CPU, cả 13 thí nghiệm trên một máy (torch 2.14.0+cpu) | 13–14/11/2026 |
| `khoi_tao_chung/mnist_gpu_chay_lai` | như trên | cùng `w0` | B, C: RTX 3060 desktop; D: RTX 3060 Laptop | 14/11/2026 |
| `khoi_tao_rieng_nghia/cifar` | `bf72c59` (nhánh `nghia`) + sửa lỗi EXP C ([VERSION](khoi_tao_rieng_nghia/VERSION)) | mỗi client khởi tạo ngẫu nhiên riêng | B: RTX 3060 desktop; C, D: RTX 3060 Laptop | 20–21/11/2026 |
| `khoi_tao_rieng_nghia/mnist` | như trên | khởi tạo riêng | RTX 3060 desktop | 20/11/2026 |

**Hai chỗ sửa tại chỗ của bản `khoi_tao_chung`** (so với `86b067d`, xem file patch):
1. `simulator.py`: sau khi tạo mô hình, mọi client nạp cùng trọng số của client 0 (`w0`).
2. `main.py`: sửa cờ `--dataset`. Bản gốc gán `_DATASETS` vào module `main` được import lại, nên không có tác dụng khi chạy `python main.py`.

**Bản `khoi_tao_rieng_nghia`** dùng đúng code `bf72c59`, chỉ sửa lỗi EXP C (commit `fix(main): EXP C crashed with TypeError…` ngay trên nhánh này). Không có lỗi đó thì cả 4 thí nghiệm `exp_c*` dừng ngay ở bước tạo cấu hình.

## Cấu trúc mỗi thư mục

- `exp_<x>_<dataset>_result.csv`: một dòng cho mỗi round. Gồm `mean_f1`, `std_f1`, `mean_acc`, F1 và accuracy từng client, `comm_cost_mb` (cộng dồn), `round_time_s`, `cum_time_s`.
- `exp?_combined_<dataset>_*.png`: hình tổng hợp từng nhóm (F1 theo round, chi phí truyền thông, hub/leaf, ...). Hình riêng của từng thí nghiệm không đưa lên; có thể vẽ lại bằng code.
- `logs/`: log từng thí nghiệm (có topology, danh sách hub/leaf, spectral gap, F1 theo round) và log tổng `run_*.log`. Tên máy và đường dẫn thư mục nhà trong log đã được thay bằng tên chung (`desktop-gpu`, `laptop-gpu`, `office-cpu`, `~`); số liệu giữ nguyên.

Ở bản `khoi_tao_chung`, `exp_c2` là bản chép của `exp_b4`, vì ở code `86b067d` hai thí nghiệm này cùng cấu hình (giống cách `run_bcd.sh` làm). Ở code `bf72c59`, C2 dùng topology ring nên được chạy riêng.

## Tổng hợp số liệu

`python phan_tich.py` đọc mọi CSV và log, rồi ghi `tong_hop.csv`: F1 cuối, độ lệch giữa các client, F1 của client kém nhất, round đạt F1 0,75, round đạt 95 % F1 cuối, chi phí truyền thông, thời gian mỗi round, F1 của hub/leaf. Script chỉ cần thư viện chuẩn của Python.

## Lưu ý khi dùng số liệu

- **Mỗi cấu hình chỉ chạy 1 seed (42).** Hai lần chạy MNIST cùng code, một trên CPU, một trên GPU, cho F1 cuối lệch tối đa 0,005. Chênh lệch nhỏ hơn mức đó nên coi là nhiễu.
- **`round_time_s` và `cum_time_s` phụ thuộc máy chạy**, nên chỉ so được giữa các thí nghiệm chạy trên cùng một máy (xem bảng trên). Ở `khoi_tao_rieng_nghia/cifar`, laptop khởi động lại 3 lần; d1 và d2 có round mất 120–243 s, nên thời gian của hai thí nghiệm này không dùng được. F1, accuracy và chi phí truyền thông không phụ thuộc máy.
- Thí nghiệm nào bị ngắt giữa chừng đều đã chạy tiếp từ checkpoint (code `bf72c59`) hoặc chạy lại từ đầu (code `86b067d`); mọi CSV đều đủ 50 round.
