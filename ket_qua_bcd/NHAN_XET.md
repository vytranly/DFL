# Nhận xét kết quả EXP B, C, D

Thiết lập chung: 10 client, 50 round, 5 local epoch, batch 32, lr 0,01, FedProx (`mu_local = 0`, `mu_agg = 0,1`), seed 42. Thông số chính là F1 trung bình giữa các client ở round 50. Nguồn gốc từng bộ số liệu xem ở [README.md](README.md); toàn bộ con số dưới đây tính bằng `phan_tich.py` (kết quả trong `tong_hop.csv`).

## Tóm tắt

1. **Khởi tạo trọng số ảnh hưởng tới mọi kết quả.** Với cùng cấu hình, khởi tạo riêng từng client (code `bf72c59` hiện tại) cho F1 cuối thấp hơn khởi tạo chung `w0` ở 10/10 cặp CIFAR. Mức giảm tới 0,07 ở topology thưa. Round 1 của CIFAR b1 sụp về đoán một lớp (accuracy 0,100). Nhóm nên dùng khởi tạo chung.
2. **Label skew là yếu tố chính.** Khi mỗi client chỉ có 2–3 lớp (α = 0,05 hoặc 0,1), F1 trên CIFAR rơi xuống 0,21–0,66, trừ fully connected (0,79). Quantity skew khi nhãn vẫn IID gần như không ảnh hưởng.
3. **Kết quả B4/C2/C4 bị lẫn hai yếu tố.** Giới hạn 8000 mẫu mỗi client làm tổng dữ liệu chỉ còn 17–34 %. Vì vậy chưa thể nói F1 giảm là do lệch kích thước hay do thiếu dữ liệu.
4. **Topology chỉ quan trọng khi dữ liệu non-IID.** Ở α = 0,1, CIFAR xếp hạng fully connected (0,787) > lattice (0,657) ≈ scale-free (0,647) > random (0,620) >> ring (0,388). Khi nhãn IID, ring chỉ kém scale-free 0,015.
5. **Hub học tốt hơn leaf khi non-IID.** Trên CIFAR scale-free α = 0,1, hub đạt 0,742 còn leaf 0,606. Khi IID thì hub và leaf gần như bằng nhau.
6. **MNIST gần bão hòa.** Mọi cấu hình nhãn IID đều đạt F1 0,99, nên các so sánh chủ yếu nên dựa vào CIFAR-10.

Mỗi cấu hình chỉ chạy 1 seed. Hai lần chạy MNIST cùng code (CPU và GPU) lệch tối đa 0,005 F1, nên chênh lệch nhỏ hơn khoảng 0,01 chưa đủ để kết luận.

---

## 1. Khởi tạo trọng số: chung `w0` hay riêng từng client

Hai lần chạy dùng cùng cách chia dữ liệu và cùng seed. Với các thí nghiệm dưới đây, khác biệt duy nhất ảnh hưởng tới kết quả là cách khởi tạo: `data_generator.py` và `trainer.py` giống hệt; phần `simulator.py` mà Nghĩa xóa là nhánh `elif fedprox` thứ hai, vốn không bao giờ chạy tới.

F1 cuối (cũ = khởi tạo chung, mới = khởi tạo riêng). Cột "round ≥ 0,75" là round đầu tiên F1 đạt 0,75 (−1 = không đạt):

| | Topology | CIFAR cũ | CIFAR mới | Δ | round ≥ 0,75 | MNIST cũ (GPU) | MNIST mới | Δ | round ≥ 0,75 |
|---|---|---|---|---|---|---|---|---|---|
| b1 | scale-free | 0,863 | 0,849 | −0,014 | 5 → 10 | 0,994 | 0,994 | 0,000 | 1 → 2 |
| b2 | scale-free | 0,859 | 0,852 | −0,007 | 5 → 8 | 0,994 | 0,993 | 0,000 | 1 → 2 |
| b3 | scale-free | 0,866 | 0,845 | −0,021 | 5 → 11 | 0,994 | 0,993 | 0,000 | 1 → 2 |
| b4 | scale-free | 0,798 | 0,776 | −0,023 | 8 → 16 | 0,988 | 0,982 | −0,006 | 1 → 1 |
| c4 | scale-free | 0,207 | 0,165 | −0,042 | −1 → −1 | 0,775 | 0,754 | −0,021 | 29 → 42 |
| d1 | ring | 0,388 | 0,337 | −0,050 | −1 → −1 | 0,828 | 0,776 | −0,053 | 9 → 26 |
| d2 | fully connected | 0,787 | 0,780 | −0,007 | 21 → 26 | 0,989 | 0,988 | 0,000 | 1 → 3 |
| d3 | random | 0,620 | 0,550 | −0,070 | −1 → −1 | 0,958 | 0,953 | −0,004 | 3 → 6 |
| d4 | lattice | 0,657 | 0,651 | −0,006 | −1 → −1 | 0,973 | 0,971 | −0,001 | 3 → 6 |
| d5 | scale-free | 0,647 | 0,578 | −0,068 | −1 → −1 | 0,963 | 0,957 | −0,006 | 4 → 6 |

Với d5 MNIST, round đạt 95 % F1 cuối tăng từ 10 lên 22.

**Nhận xét:**
- **CIFAR:** khởi tạo riêng thấp hơn ở cả 10 cặp. Mức giảm lớn nhất rơi vào topology thưa hoặc không đều: ring −0,050, random −0,070, scale-free −0,068. Fully connected và lattice gần như không đổi.
- **MNIST:** F1 cuối phần lớn nằm trong mức nhiễu, nhưng thí nghiệm nào cũng **hội tụ chậm hơn**: số round cần để đạt 0,75 thường tăng gấp đôi, riêng ring tăng từ 9 lên 26.
- **Round 1 bị sụp:** CIFAR b1 và d5 khởi tạo riêng có F1 = 0,018 và accuracy = 0,100 ở round 1, tức mọi client cùng đoán một lớp. Khởi tạo chung cho F1 0,113 (b1) và 0,035 (d5).
- **Lý giải:** lấy trung bình các mạng được khởi tạo khác nhau làm hỏng trọng số, vì các neuron của mỗi mạng không tương ứng với nhau. FedAvg (McMahan và cộng sự, 2017, Hình 1) và D-PSGD (Lian và cộng sự, 2017) đều cho mọi node bắt đầu từ cùng một mô hình. Topology càng thưa thì các node càng mất nhiều round mới "kéo" được về cùng một mô hình, nên khởi tạo riêng **làm lệch chính phép so sánh topology** ở EXP D.
- **Đề xuất:** thêm 3 dòng sau vào `simulator.py`, ngay sau khi tạo `self.models` (đã có trong patch của bản cũ), rồi chạy lại các thí nghiệm của code mới:

  ```python
  w0 = copy.deepcopy(self.models[0].state_dict())
  for cid in range(1, self.num_clients):
      self.models[cid].load_state_dict(w0)
  ```

Các mục 2–4 dưới đây dùng bản khởi tạo chung (CIFAR và MNIST chạy CPU). Riêng phần C mới dùng bản của Nghĩa vì cấu hình C đã thay đổi.

## 2. EXP B: quantity skew (nhãn IID, α = 100, scale-free)

Kích thước dữ liệu từng client, tính lại bằng đúng hàm chia dữ liệu và seed của code:

| | Cấu hình | CIFAR: mẫu/client | Tổng | F1 cũ | F1 mới | MNIST F1 cũ |
|---|---|---|---|---|---|---|
| b1 | chia đều | 4718–5334 | 50.000 | 0,863 | 0,849 | 0,994 |
| b2 | bounded, max_diff 1500 | 4138–5475 (lệch thật 1337) | 49.940 | 0,859 | 0,852 | 0,993 |
| b3 | bounded, max_diff 3000 | 4159–5992 (lệch thật 1833) | 49.549 | 0,866 | 0,845 | 0,994 |
| b4 | imbalanced, size α 0,1, tối đa 8000 | 20–5334 | **17.051 (34 %)** | 0,799 | 0,776 | 0,988 |

**Nhận xét:**
- **b1, b2, b3 cho kết quả như nhau** (trên CIFAR chênh tối đa 0,007 ở cùng một cách khởi tạo). Lý do là "bounded" thực tế chỉ lệch 1337 và 1833 mẫu giữa client lớn nhất và nhỏ nhất, khoảng ±15 % quanh 5000. Mức lệch đó quá nhỏ để thấy ảnh hưởng. Muốn khảo sát quantity skew thì cần mức lệch lớn hơn nhiều.
- **b4 giảm 0,064 F1 trên CIFAR, nhưng kết luận đang bị lẫn hai yếu tố.** Giới hạn 8000 mẫu mỗi client kết hợp phân phối kích thước rất lệch làm tổng dữ liệu chỉ còn 34 % (MNIST: 32 %). Vì thế mỗi round của b4 nhanh gấp đôi, và F1 giảm có thể do thiếu dữ liệu chứ không riêng do lệch kích thước. Nên chạy thêm một biến thể giữ nguyên tổng 50.000 mẫu, chỉ thay đổi cách phân bổ, để tách hai yếu tố.
- **Khi nhãn IID, độ lệch giữa các client rất nhỏ** (std 0,003–0,013) và hub với leaf gần như bằng nhau (b1: 0,866 so với 0,862). Mọi client đều học tốt.

## 3. EXP C: label skew + quantity skew

### Bản cũ (scale-free cho cả nhóm)

| | Cấu hình | Lớp/client (CIFAR) | Tổng dữ liệu | CIFAR F1 | F1 client kém nhất | hub / leaf | MNIST F1 |
|---|---|---|---|---|---|---|---|
| c1 | chỉ label skew, α 0,05 | 1–5 (tb 2,7) | 50.000, nhưng 127–13.128/client | 0,448 | 0,323 | 0,549 / 0,404 | 0,688 |
| c2 | = b4 | 9–10 | 17.051 | 0,799 | 0,788 | 0,805 / 0,796 | 0,988 |
| c3 | α 0,5, bounded 2500 | 4–7 (tb 5,5) | 48.376 | 0,828 | 0,795 | 0,842 / 0,822 | 0,992 |
| c4 | α 0,05, imbalanced | 1–5 (tb 2,7) | **8.639 (17 %)** | 0,207 | 0,073 | 0,274 / 0,179 | 0,770 |

**Nhận xét:**
- **Label skew áp đảo quantity skew.** c2 (chỉ quantity skew) vẫn đạt 0,799, còn c1 (chỉ label skew) chỉ đạt 0,448 trên CIFAR. c4 có cả hai và thiếu dữ liệu nên kém nhất (0,207), có client chỉ đạt 0,07.
- **C1 "chỉ label skew" thật ra cũng lệch kích thước rất mạnh** (127–13.128 mẫu mỗi client), vì chia Dirichlet theo từng lớp ở α nhỏ đồng thời làm lệch kích thước. Khi viết báo cáo cần nói rõ điểm này.
- **Khi non-IID, các client chênh nhau rất xa.** Ở c1 MNIST, độ lệch giữa các client là 0,224 và client kém nhất chỉ đạt 0,241. Hub (0,549) vượt leaf (0,404) trên CIFAR, vì hub nhận trọng số từ nhiều hàng xóm có dữ liệu khác nhau.
- **c3 (α 0,5) gần như không giảm** (0,828 so với 0,863 của b1). Với mỗi client có khoảng 5–6 lớp, FedProx vẫn xử lý được.

### Bản mới của Nghĩa (C1, C2 dùng ring; C3, C4 dùng scale-free; khởi tạo riêng)

| | Cấu hình | Lớp/client (CIFAR) | CIFAR F1 | MNIST F1 |
|---|---|---|---|---|
| c1 | ring, α 0,05 | 1–5 (tb 2,7) | 0,238 | 0,735 |
| c2 | ring, α 100, imbalanced | 9–10 | 0,761 | 0,986 |
| c3 | scale-free, α 10, bounded 2500 | **9–10 (tb 9,9)** | 0,826 | 0,994 |
| c4 | scale-free, α 0,05, imbalanced | 1–5 (tb 2,7) | 0,165 | 0,754 |

**Nhận xét về thiết kế mới:**
- **Topology bị lẫn vào nhóm C.** C1 và C2 dùng ring, còn C3 và C4 dùng scale-free. Khi so C1 với C3/C4, ta không tách được phần chênh nào do skew và phần nào do topology. Nên dùng cùng một topology cho cả nhóm C; nếu muốn xét topology thì chạy cả nhóm trên hai topology.
- **C3 mới (α = 10) gần như IID:** mỗi client có trung bình 9,9 lớp, nên F1 0,826 không thể hiện mức skew "trung bình" như tên gọi. α khoảng 0,5–1 hợp với ý "trung bình" hơn.
- **So sánh có ích ngoài dự kiến:** C2 mới (ring) và B4 mới (scale-free) có cùng cách chia dữ liệu. Khi nhãn IID, ring chỉ kém 0,015 (0,761 so với 0,776). Trong khi đó ở α = 0,1 (EXP D, cùng code mới), ring kém scale-free tới 0,241 (0,337 so với 0,578). Tức là **topology chỉ thực sự quan trọng khi dữ liệu non-IID.**

## 4. EXP D: topology (α = 0,1, mỗi client 1–5 lớp, trung bình 2,6)

| | Topology | Số cạnh | Spectral gap | CIFAR F1 | client kém nhất | Chi phí truyền (GB) | round đạt 95 % F1 cuối | hub / leaf | MNIST F1 |
|---|---|---|---|---|---|---|---|---|---|
| d1 | ring | 10 | 0,127 | 0,388 | 0,245 | 9,9 | 30 | (mọi node bậc 2) | 0,825 |
| d5 | scale-free | 16 | 0,167 | 0,647 | 0,438 | 15,9 | 26 | 0,742 / 0,606 | 0,963 |
| d3 | random, p = 0,4 | 20 | 0,259 | 0,620 | 0,457 | 19,8 | 34 | 0,693 / 0,571 | 0,956 |
| d4 | lattice, k = 4 | 20 | 0,353 | 0,657 | 0,565 | 19,8 | 26 | (mọi node bậc 4) | 0,974 |
| d2 | fully connected | 45 | 1,000 | 0,787 | 0,787 | 44,6 | 20 | (mọi node bậc 9) | 0,989 |

**Nhận xét:**
- **Thứ hạng giống nhau trên cả hai dataset:** fully connected > lattice ≈ scale-free > random >> ring. F1 nhìn chung tăng theo spectral gap. Riêng scale-free (gap 0,167) đạt F1 ngang lattice (gap 0,353) với ít cạnh hơn: hub đóng vai trò "trạm trung chuyển" trộn mô hình rất hiệu quả.
- **Cùng chi phí truyền thông, cấu trúc đều tốt hơn.** Random và lattice có cùng 20 cạnh và cùng 19,8 GB, nhưng lattice hơn 0,037 F1 trên CIFAR và client kém nhất của lattice cao hơn hẳn (0,565 so với 0,457). Random có node gần như bị cô lập, kéo client kém nhất xuống.
- **Đánh đổi chi phí truyền và F1:**
  - Fully connected tốn 44,6 GB, gấp 2,8 lần scale-free (15,9 GB), để hơn 0,14 F1.
  - Ring rẻ nhất (9,9 GB) nhưng chỉ đạt 0,388.
  - Scale-free có tỉ lệ F1 trên chi phí tốt nhất nhóm.
  - Điều này khớp với nhận xét buổi 2 của cô: fully connected gần như mô hình tập trung nhưng chi phí truyền tăng theo bình phương số node.
- **Hub và leaf:** trên scale-free, hub đạt 0,742 còn leaf 0,606; trên random là 0,693 và 0,571. Leaf có ít hàng xóm nên chậm "nhận" được tri thức về các lớp mình không có. Đây là số liệu trả lời yêu cầu "theo dõi riêng hub và leaf" của cô.
- **Fully connected cho mọi client cùng F1 (độ lệch = 0),** vì sau mỗi round mọi node có cùng mô hình trung bình. Đây đúng là hành vi của FedAvg tập trung.

### Về "thời gian thực" mà cô yêu cầu
Cột `round_time_s` đo thời gian chạy mô phỏng **tuần tự trong một process**: mỗi round gồm huấn luyện 10 client lần lượt rồi tổng hợp. Thời gian đó chủ yếu là huấn luyện, nên gần như không đổi theo topology (117–121 s/round cho cả 4 topology chạy trên cùng laptop: ring, fully connected, random, lattice). Vì vậy số liệu hiện tại **không thể hiện được** ý của cô: "mạng thưa thì mỗi round nhanh nhưng cần nhiều round". Thước đo thay thế hiện có là chi phí truyền thông (GB) kết hợp số round hội tụ. Muốn đo thời gian thật cần chạy chế độ đa process (`p2p/`), nơi thời gian truyền phụ thuộc số hàng xóm.

## 5. Các vấn đề phát hiện trong code và thiết kế

1. **Lỗi EXP C ở `bf72c59`:** `dict(**_EXP_C_BASE, **extra)` báo TypeError vì `topology` bị truyền hai lần, nên cả 4 thí nghiệm C dừng ngay khi bắt đầu. Đã sửa trong commit đầu của nhánh này.
2. **Khởi tạo riêng từng client:** xem mục 1.
3. **Giới hạn 8000 mẫu (b4, c2, c4) làm giảm tổng dữ liệu xuống còn 17–34 %**, lẫn với yếu tố quantity skew. Xem mục 2.
4. **"Bounded" chỉ lệch khoảng ±15 %**, nên b2 và b3 không khác b1. Xem mục 2.
5. **`target_f1 = 0,75` cố định:** nhiều cấu hình CIFAR không bao giờ đạt, nên "Convergence round" = −1 và không so sánh được tốc độ. Nên dùng thêm ngưỡng tương đối, ví dụ 95 % F1 cuối (đã có trong `tong_hop.csv`, cột `r95`).
6. **Hub được định nghĩa là node có bậc lớn hơn bậc trung bình, với 10 node.** Cô đề nghị mô tả hub (khoảng ≥ 6 kết nối) và leaf (1–2 kết nối) trên quy mô khoảng 20 node; hiện chưa có thí nghiệm nào thay đổi số node.
7. **Chỉ 1 seed:** chênh lệch dưới khoảng 0,01 (ví dụ b1, b2, b3) chưa đủ để kết luận. Nên chạy 3 seed cho những cặp quan trọng.

## 6. Việc nên làm tiếp

1. Chốt khởi tạo chung `w0` trong code của nhóm, rồi chạy EXP A và E bằng code đó.
2. Nếu giữ thiết kế C mới: đưa C1–C4 về cùng một topology, và chọn lại α cho C3.
3. Thêm biến thể B4 giữ nguyên tổng dữ liệu để tách hai yếu tố.
4. Chạy thêm seed cho EXP D, vì đây là phần trả lời trực tiếp nhận xét của cô về topology.
