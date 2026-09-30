# ADR-005: Trọng số trộn Metropolis–Hastings mặc định; uniform và size-corrected là ablation; size-weighted chỉ dùng trên FC

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30 (cập nhật theo hai vòng phản biện P1.9 cùng ngày)

## Context
- Mỗi node tổng hợp theo công thức $w_i^{t+1}=\sum_j W_{ij}\,w_j^{t+\frac12}$. W phải không âm, tổng mỗi hàng bằng 1 và khớp với cạnh của đồ thị (INV-4). Nếu W còn ngẫu nhiên kép thì trung bình các model được bảo toàn và mạng đồng thuận về đúng trung bình đó (KH §3.1).
- Trọng số phải tính được cục bộ: node chỉ biết láng giềng của mình (REQ-01).
- Với W ngẫu nhiên kép, decentralized SGD tối ưu $\frac1N\sum_i F_i$ (mọi node có trọng số bằng nhau), còn FedAvg tối ưu $\sum_i \frac{n_i}{n}F_i$. Hai mục tiêu này khác nhau khi kích thước node lệch nhau, như ở cách chia Dirichlet và #C = k ("lệch mục tiêu", H6, G19).
- RQ4 hỏi cách chọn trọng số trộn có quan trọng không.
- Nguồn: KH §3.1, §3.3, §3.6 ($\mathrm{Acc}^{eff}$), §10; Xiao & Boyd 2004; SPEC §3.1 (`network`); `docs/decisions/REVIEW-P1.9.md`.

## Decision
- **Mặc định — Metropolis–Hastings (MH):** $W_{ij}=\frac{1}{1+\max(d_i,d_j)}$ trên mỗi cạnh $(i,j)$ và $W_{ii}=1-\sum_{j\ne i}W_{ij}$. Ma trận đối xứng, ngẫu nhiên kép, và mỗi node tính được hàng của mình chỉ từ bậc của láng giềng.
- **Ablation cho E4:**
  - *Uniform:* $W_{ij}=\frac{1}{d_i+1}$. Chỉ ngẫu nhiên theo hàng; trên đồ thị không đều, mạng đồng thuận về trung bình có trọng số tỉ lệ với $d_j+1$. Trên đồ thị đều (ring, FC, matching của gossip) uniform trùng hẳn MH, nên ablation uniform chỉ chạy trên ER (T2).
  - *Size-corrected:* giữ MH, nhân mất mát cục bộ của node $i$ với $c_i=n_i/\bar n$, trong đó $\bar n$ là kích thước node trung bình (tức $c_i=Nn_i/n$), để điểm đồng thuận xấp xỉ tối ưu $\sum_i\frac{n_i}{n}F_i$.
    - $\bar n$ là đại lượng toàn cục, được tính một lần trước khi huấn luyện. Đây là ngoại lệ duy nhất của INV-1, và được ghi là hạn chế. Một cách lấy phi tập trung là chạy average consensus trên $n_j$ trước khi huấn luyện; dự án không cài cách này.
    - Chỉ dùng với μ = 0 (DFedAvg). Nếu kết hợp với prox thì $c_i$ làm yếu tương đối số hạng prox ở node lớn, nên không kết hợp.
    - Với SGD thường (momentum 0, μ = 0), nhân mất mát với $c_i$ đúng bằng nhân bước học của node $i$ với $c_i$: $\eta_i=c_i\eta$. Size-corrected vì vậy là một can thiệp về *bước học theo node*, nhằm tái trọng số mục tiêu. Nó không phải một cách chọn trọng số trộn, và không được dùng làm bằng chứng cho khuyến nghị về trọng số trộn. Khuyến nghị đó chỉ dựa trên MH so với uniform (SPEC REQ-13). Báo cáo size-corrected riêng, kèm dải $c_i$ của từng lượt.
    - Tương đương với mục tiêu $\sum_i\frac{n_i}{n}F_i$ chỉ đúng gần đúng, khi η nhỏ. Với τ = 20 bước cục bộ và bước học hiệu dụng lệch nhau giữa các node (node lớn tới khoảng 2.2 lần ở α = 0.1, 4 lần ở β = 0.5), điểm hội tụ có thể lệch khỏi mục tiêu đó ("objective inconsistency", như FedNova của Wang et al. 2020 đã chỉ ra cho FL). Kết quả của E4 về size-corrected vì vậy lẫn hai hiệu ứng: tái trọng số mục tiêu, và bước học lớn hơn ở node lớn.
    - Nếu phân kỳ, lượt được ghi `diverged` và báo cáo như một hạn chế của cách sửa này; không tinh chỉnh riêng.
  - *Size-weighted trên FC:* $W_{ij}=n_j/n$, đúng bằng FedAvg. Mọi lượt size-weighted trên FC (B3, các lượt E1c, và ô size-weighted của E4 ở cả β = 0.5 lẫn α = 0.1) không chạy như FC trong Transport (sẽ đếm 380 message/round), mà qua *đường server* của engine. Nhờ vậy mọi lượt FedAvg có cùng một cách đếm chi phí. Server ảo có chỉ số N, không có dữ liệu. Mỗi round nó nhận N snapshot, tổng hợp theo $n_k/n$, rồi gửi lại N snapshot. Mọi message đi qua Transport, nên CommLedger đếm đúng 2N message/round, và tải lớn nhất là của server (INV-2). T-INV-3 kiểm đường server cho cùng model với FC + size-weighted (allclose).
- **Trọng số mục tiêu π** (`objective_weights` trong manifest) là trọng số của mục tiêu mà lượt chạy thực sự tối ưu: MH là 1/N; uniform tỉ lệ với $d_j+1$; size-weighted và size-corrected là $n_j/n$. Manifest ghi π và độ lệch mục tiêu $\mathrm{TV}(\sum_j\pi_jP_j, P)$ của từng lượt; $\mathrm{Acc}^{eff}$ dùng π này.

## Alternatives Considered

### Uniform làm mặc định
- Ưu: đơn giản nhất.
- Nhược: không ngẫu nhiên kép trên đồ thị không đều (ER), nên điểm đồng thuận lệch theo bậc (G5).
- Kết luận: chỉ dùng làm ablation, trên ER.

### $W_{ij}\propto n_j$ chuẩn hóa theo hàng trên đồ thị không đầy đủ
- Ưu: trông giống "FedAvg cục bộ".
- Nhược: hội tụ về trọng số tỉ lệ với $n_j s_j$ ($s_j$ là tổng dữ liệu trong lân cận của $j$), không phải $n_j$, nên không sửa được lệch mục tiêu (G19).
- Kết luận: không chọn.

### Ma trận trộn tối ưu tốc độ hội tụ (fastest mixing, Xiao & Boyd 2004)
- Ưu: spectral gap lớn nhất có thể trên một đồ thị cho trước.
- Nhược: phải giải một bài toán tối ưu lồi dùng thông tin của toàn đồ thị, không tính cục bộ đơn giản như MH; làm thay đổi chính đại lượng giải thích (spectral gap) mà dự án dùng để so sánh topology.
- Kết luận: không chọn.

### Sửa lệch mục tiêu bằng số bước cục bộ tỉ lệ với $n_i$ (kiểu FedNova)
- Ưu: không phóng đại bước học.
- Nhược: đổi tổng số bước cục bộ giữa các node, phá thiết kế τ·R cố định (ADR-006); tốn thêm công cài đặt.
- Kết luận: không chọn; size-corrected được giữ như một ablation có hạn chế đã nêu.

## Consequences
- Spectral gap với N = 20 và MH: ring 0.033; FC 1.000; ER p = 0.3 trung bình khoảng 0.20. Đo lại trên 2,000 đồ thị ER liên thông, 95% số đồ thị nằm trong khoảng 0.09–0.32; KH §3.3 ghi 0.198 (0.085–0.310) trên 20 seed. T-INV-9 kiểm giá trị của ring và FC. Spectral gap của W không đối xứng (uniform) phải tính bằng `np.linalg.eigvals`, không dùng `eigvalsh`.
- Gossip trên $K_N$ không có spectral gap cố định, vì W đổi mỗi round. Nó được mô tả bằng tốc độ trộn kỳ vọng $1-\sqrt{\lambda_2(\mathbb E[W])}\approx 0.312$ (ADR-009). Con số này không phải spectral gap: spectral gap của $\mathbb E[W]$ là $1-\lambda_2(\mathbb E[W])\approx 0.526$. Manifest ghi `spectral_gap = null` cho gossip, và P8 vẽ gossip riêng.
- Trên FC, MH cho $W=\frac1N\mathbf 1\mathbf 1^\top$: sau tổng hợp mọi node trùng model, nên std giữa node và consensus distance bằng 0 theo cấu trúc (G15). Vì vậy phải đo thêm acc^loc và drift trước tổng hợp (ADR-007, T-INV-11).
- Mọi kết quả Non-IID (Dirichlet, #C = k, quantity) phải đi kèm độ lệch mục tiêu (TV) và kích thước node. E1c và E4 giúp ước lượng phần suy giảm do lệch mục tiêu so với phần do drift, nhưng không tách được hoàn toàn (H6).
- **Trade-off đã chấp nhận:** size-corrected dùng một đại lượng toàn cục, chỉ gần đúng, và thực chất là bước học theo node; kết luận của E4 về nó phải nêu ba hạn chế này. $n_k$ gửi kèm lên server ở đường server không được đếm vào byte (1 số nguyên, dưới 0.005% một message). Cách không lẫn bước học (số bước cục bộ tỉ lệ $n_i$, kiểu FedNova) phá thiết kế τ·R cố định, nên không chọn (Alternatives).
- `network/mixing.py` không được import torch (SPEC §3.2).
