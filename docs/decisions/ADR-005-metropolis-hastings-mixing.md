# ADR-005: Trọng số trộn Metropolis–Hastings mặc định; uniform và size-corrected là ablation; size-weighted chỉ dùng trên FC

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30

## Context
- Mỗi node tổng hợp theo công thức $w_i^{t+1}=\sum_j W_{ij}\,w_j^{t+\frac12}$. W phải không âm, tổng mỗi hàng bằng 1 và khớp với cạnh của đồ thị (INV-4). Nếu W còn ngẫu nhiên kép thì trung bình các model được bảo toàn và mạng đồng thuận về đúng trung bình đó (KH §3.1).
- Trọng số phải tính được cục bộ: node chỉ biết láng giềng của mình (REQ-01).
- Với W ngẫu nhiên kép, decentralized SGD tối ưu $\frac1N\sum_i F_i$ (mọi node có trọng số bằng nhau), còn FedAvg tối ưu $\sum_i \frac{n_i}{n}F_i$. Hai mục tiêu này khác nhau khi kích thước node lệch nhau, như ở cách chia Dirichlet ("lệch mục tiêu", H6, G19).
- RQ4 hỏi cách chọn trọng số trộn có quan trọng không.
- Nguồn: KH §3.1, §3.3, §3.6 ($\mathrm{Acc}^{eff}$), §10; Xiao & Boyd 2004; SPEC §3.1 (`network`).

## Decision
- **Mặc định — Metropolis–Hastings (MH):** $W_{ij}=\frac{1}{1+\max(d_i,d_j)}$ trên mỗi cạnh $(i,j)$ và $W_{ii}=1-\sum_{j\ne i}W_{ij}$. Ma trận đối xứng, ngẫu nhiên kép, và mỗi node tính được hàng của mình chỉ từ bậc của láng giềng.
- **Ablation cho E4:**
  - *Uniform:* $W_{ij}=\frac{1}{d_i+1}$. Chỉ ngẫu nhiên theo hàng; trên đồ thị không đều, mạng đồng thuận về trung bình có trọng số tỉ lệ với $d_j+1$.
  - *Size-corrected:* giữ MH, nhân mất mát cục bộ của node $i$ với $c_i=N n_i/n$, để điểm đồng thuận tối ưu $\sum_i\frac{n_i}{n}F_i$. Bước học hiệu dụng của node lớn tăng tới khoảng 2.2 lần (α = 0.1) hoặc 4 lần (β = 0.5); nếu phân kỳ thì báo cáo như một hạn chế của cách sửa này, không tinh chỉnh riêng.
  - *Size-weighted trên FC:* $W_{ij}=n_j/n$, đúng bằng FedAvg (baseline B3, các lượt E1c).
- Manifest ghi trọng số đồng thuận π (MH: 1/N; size-weighted và size-corrected: $n_j/n$; uniform: tỉ lệ với $d_j+1$) và độ lệch mục tiêu của từng lượt.

## Alternatives Considered

### Uniform làm mặc định
- Ưu: đơn giản nhất.
- Nhược: không ngẫu nhiên kép trên đồ thị không đều (ER), nên điểm đồng thuận lệch theo bậc (G5).
- Kết luận: chỉ dùng làm ablation.

### $W_{ij}\propto n_j$ chuẩn hóa theo hàng trên đồ thị không đầy đủ
- Ưu: trông giống "FedAvg cục bộ".
- Nhược: hội tụ về trọng số tỉ lệ với $n_j s_j$ ($s_j$ là tổng dữ liệu trong lân cận của $j$), không phải $n_j$, nên không sửa được lệch mục tiêu (G19).
- Kết luận: không chọn.

### Ma trận trộn tối ưu tốc độ hội tụ (fastest mixing, Xiao & Boyd 2004)
- Ưu: spectral gap lớn nhất có thể trên một đồ thị cho trước.
- Nhược: phải giải một bài toán tối ưu lồi dùng thông tin của toàn đồ thị, không tính cục bộ đơn giản như MH; làm thay đổi chính đại lượng giải thích (spectral gap) mà dự án dùng để so sánh topology.
- Kết luận: không chọn.

## Consequences
- Spectral gap với N = 20 và MH: ring 0.033; ER p = 0.3 trung bình 0.198 (0.085–0.310 tùy seed); gossip trên $K_N$ 0.312 (kỳ vọng); FC 1.000 (KH §3.3). T-INV-9 kiểm giá trị của ring và FC.
- Trên FC, MH cho $W=\frac1N\mathbf 1\mathbf 1^\top$: sau tổng hợp mọi node trùng model, nên std giữa node và consensus distance bằng 0 theo cấu trúc (G15). Vì vậy phải đo thêm acc^loc và drift trước tổng hợp (ADR-007, T-INV-11).
- Mọi kết quả Dirichlet phải đi kèm độ lệch mục tiêu (TV) và kích thước node. E1c và E4 giúp ước lượng phần suy giảm do lệch mục tiêu so với phần do drift, nhưng không tách được hoàn toàn (H6).
- `network/mixing.py` không được import torch (SPEC §3.2).
