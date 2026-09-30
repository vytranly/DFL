# ADR-006: Một vòng Gossip-SGD kiểu ATC tham số (τ, μ, $c_i$) cho cả ba thuật toán; τ·R = 4,000 cố định; khởi tạo chung

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30

## Context
- REQ-08 yêu cầu các thuật toán tổng hợp FedAvg (hoặc Gossip Learning), FedSGD và FedProx, ở đây trong bối cảnh phi tập trung.
- RQ3 hỏi tần suất giao tiếp (τ) và số hạng prox (μ) đánh đổi accuracy với chi phí giao tiếp ra sao ở cùng tổng số bước cục bộ. So D-PSGD với DFedAvg theo round là sai (G10).
- Ba cài đặt riêng cho ba thuật toán dễ lệch nhau ở chi tiết (cách lấy minibatch, thứ tự tổng hợp), làm hỏng phép so sánh.
- μ quá nhỏ làm FedProx gần như không khác DFedAvg (G16); cường độ hiệu dụng của số hạng prox là ημτ.
- Koloskova et al. 2020: tốc độ hội tụ phụ thuộc vào tỉ lệ p/τ giữa tham số trộn và số bước cục bộ (DIGEST §2.5).
- Bản nháp của nhóm mô tả thứ tự tổng hợp rồi mới huấn luyện (combine-then-adapt), xem comment C9502 trong `docs/đath1_revised.docx`.
- A3: mọi node cùng $w^0$ theo seed chung. Badie-Modiri et al. gọi đây là "coordinated initialisation" và lấy khởi tạo không phối hợp làm tiền đề (DIGEST §2.8).
- Nguồn: KH §3.4, §3.5, §10; SPEC §9.1 (REQ-08), §10 (điểm 12), §11.2 (QS-2); OQ A3.

## Decision
- **Một vòng lặp Adapt-Then-Combine (ATC)** cho mọi thuật toán:
  1. *Adapt:* node $i$ xuất phát từ $w_i^t$, chạy τ bước SGD với mất mát $c_i\,\ell(w)+\frac{\mu}{2}\lVert w-w_i^t\rVert^2$. Anchor $w_i^t$ là bản clone + detach, không có gradient (G6). Mặc định $c_i=1$.
  2. *Gửi:* mỗi node tạo đúng một snapshot sở hữu riêng của $w_i^{t+\frac12}$ và gửi tới các láng giềng qua Transport (ADR-002).
  3. *Combine:* $w_i^{t+1}=\sum_j W^{(t)}_{ij}\,w_j^{t+\frac12}$, cộng theo id người gửi tăng dần; chỉ ghi sau khi mọi node đã tổng hợp xong.
  4. *Đo:* Observer đo drift trước tổng hợp và các độ đo còn lại sau tổng hợp (ADR-007).
- **Các thuật toán là những điểm trong không gian (τ, μ):**
  - D-PSGD (Lian et al. 2017): τ = 1, μ = 0, R = 4,000 — tương tự phi tập trung gần nhất của FedSGD.
  - DFedAvg (Sun et al. 2022, không momentum): τ = 20, μ = 0, R = 200 — thuật toán chính của E1 và E2.
  - DFedProx (Li et al. 2020): τ = 20 với μ ∈ {0.1, 1}, tức ημτ ∈ {0.1, 1} ở η = 0.05; thêm μ = 1 ở τ = 50 (ημτ = 2.5).
  - Quét τ ∈ {1, 5, 20, 50} với μ = 0: τ = 5 và 50 chạy ở E3b; τ = 1 lấy từ E3a (D-PSGD); τ = 20 lấy từ E0b/E1 (DFedAvg).
- **Ngân sách:** tổng số bước cục bộ τ·R = 4,000 cố định cho mọi điều kiện; η dùng chung cho mọi τ (ADR-007).
- **Khởi tạo:** mọi node dùng cùng $w^0$, sinh từ seed chung (A3).
- **Nếu E0a chọn η ≠ 0.05** (SPEC QS-2, đang dùng mặc định): giữ cường độ ημτ ∈ {0.1, 1} ở τ = 20, tức μ = {0.1, 1} × 0.05/η, và ghi μ vào ADR-007 cùng với η.

## Alternatives Considered

### Ba cài đặt riêng cho ba thuật toán
- Ưu: sát với mô tả gốc của từng bài.
- Nhược: trùng code; dễ lệch nhau ở chi tiết, làm sai phép so sánh.
- Kết luận: không chọn.

### Thứ tự combine-then-adapt (như bản nháp), hoặc dạng gốc của Lian 2017, $x^{t+1}=Wx^t-\eta\,g(x^t)$
- Ưu: khớp bản nháp và công thức gốc của D-PSGD.
- Nhược: DFedAvg là huấn luyện cục bộ rồi mới trung bình; dùng hai thứ tự khác nhau cho hai thuật toán sẽ làm lệch phép so sánh, trong khi cần một thứ tự duy nhất cho mọi τ.
- Kết luận: không chọn. Báo cáo ghi rõ D-PSGD ở đây là dạng ATC (KH §3.4); bản nháp docx cần sửa theo ATC (C9502).

### μ = 0.01
- Ưu: giá trị hay gặp trong tài liệu FedProx.
- Nhược: ở η = 0.05, τ = 20 thì ημτ = 0.01, gần như không khác DFedAvg (G16), dễ dẫn tới kết luận sai "FedProx vô dụng".
- Kết luận: không chọn.

### Mỗi node tự khởi tạo độc lập
- Ưu: sát tiền đề "không phối hợp" của Badie-Modiri et al.
- Nhược: gây một bình nguyên loss kéo dài theo N (Badie-Modiri, DIGEST §2.8); thêm một biến gây nhiễu cho câu hỏi chính.
- Kết luận: không chọn làm mặc định. Khởi tạo chung được ghi là hạn chế trong báo cáo; ablation khởi tạo độc lập (dùng hệ số hiệu chỉnh của bài) là phần mở rộng.

## Consequences
- Ít code, một đường đi cho mọi thuật toán. Kiểm bằng T-INV-3 (FC + size-weighted tương đương FedAvg), T-INV-7 (μ = 0 tương đương SGD; gradient prox đúng) và T-INV-4 (thứ tự duyệt không ảnh hưởng kết quả).
- Mọi so sánh giữa thuật toán dùng trục số bước cục bộ và trục byte, không dùng round (G10). D-PSGD cần 4,000 round và gửi nhiều message gấp 20 lần DFedAvg ở τ = 20.
- η dùng chung có thể bất lợi cho D-PSGD. E0a quét thêm η cho τ = 1; nếu η tốt nhất của τ = 1 khác η chung từ 2 lần trở lên thì chạy thêm 12 lượt D-PSGD với η riêng (Tier 2).
- Hạn chế cần nêu trong báo cáo: khởi tạo chung (A3); D-PSGD ở dạng ATC; lr dùng chung cho τ ≠ 20.
- Phụ thuộc SPEC QS-2 (μ khi η ≠ 0.05).
