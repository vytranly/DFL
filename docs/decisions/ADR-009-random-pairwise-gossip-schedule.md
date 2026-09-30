# ADR-009: Random gossip là perfect matching đều trên $K_N$ theo lịch sinh từ seed chung; thêm T3b giới hạn trên cạnh ER

## Status
Proposed — chờ Gate G1.

## Date
2026-09-30 (cập nhật theo hai vòng phản biện P1.9 cùng ngày)

## Context
- Đề nêu "random gossip" trong các topology cần so sánh (REQ-07); topology này thay đổi theo từng round.
- Không được có điều phối viên trung tâm (REQ-01). Lịch gossip phải tất định và là hàm thuần của (seed, round), để việc gọi `graph_at` thêm lần không làm đổi quỹ đạo huấn luyện (G18, T-INV-12).
- Gossip trên $K_N$ cho phép ghép cặp hai node bất kỳ, một lợi thế mà các topology thưa cố định không có. Muốn so sánh chi phí theo byte (H4) cho công bằng thì cần một đối chứng.
- Nguồn: KH §3.3, §10; Boyd et al. 2006 (randomized gossip); SPEC §3.1 (`network`), §6; H§2.4 (G18); `docs/decisions/REVIEW-P1.9.md`.

## Decision
- **T3 — gossip trên $K_N$:** mỗi round một perfect matching đều ngẫu nhiên trên $K_N$, sinh từ lịch chung. Mọi node tự tính đối tác của mình từ luồng gossip của seed chung, tức `SeedSequence(entropy=seed, spawn_key=(ID_GOSSIP, round_idx))` trong `seeding.py`. Luồng này không trùng entropy với luồng nào khác (SPEC §6), và không cần điều phối viên. Mỗi cặp $(i,j)$ dùng $W_{ii}=W_{jj}=W_{ij}=W_{ji}=\frac12$ (chính là MH trên đồ thị matching). Nếu N lẻ, mỗi round có một node không được ghép.
- **T3b — gossip giới hạn trên cạnh ER:** mỗi round một maximal matching ngẫu nhiên trên cạnh của đồ thị T2, cũng sinh từ lịch chung, với luồng `SeedSequence(entropy=seed, spawn_key=(ID_GOSSIP_ER, round_idx))`. T3b là đối chứng công bằng cho H4 (E2b).
- **Node không được ghép** (N lẻ ở T3, hoặc bị bỏ ra ở T3b) vẫn chạy bước Adapt như mọi node khác. Nó không gửi, không nhận message nào, và ở bước Combine dùng $W_{ii}=1$, tức giữ nguyên $w^{t+\frac12}$ của chính nó (ADR-006). Nhờ vậy mọi node có cùng τ·R bước cục bộ và luồng batch của node không bị lệch.
- **Liên thông:** kiểm theo hợp của các round, không theo từng round (INV-4).
- **Tốc độ trộn kỳ vọng:** với T3 là $1-\sqrt{\lambda_2(\mathbb E[W])}$, trong đó $\lambda_2(\mathbb E[W])=\frac12-\frac{1}{2(N-1)}$ khi N chẵn và $=\frac12$ khi N lẻ; giá trị bằng 0.312 với N = 20. Với T3b thì ước lượng bằng cách lấy mẫu $10^4$ matching. Đại lượng này không phải spectral gap (spectral gap của $\mathbb E[W]$ là 0.526 với N = 20), nên không được đặt chung trục với spectral gap của topology tĩnh, và manifest không ghi nó vào `spectral_gap` (ADR-005).
- **"Không có điều phối viên" nghĩa là gì trong mô phỏng:** engine gọi `graph_at(t)` một lần mỗi round rồi đưa W cho các node, vì mọi node chạy trong một tiến trình. Điều kiện kiểm được là lịch chỉ phụ thuộc (seed chung, t, N), không phụ thuộc trạng thái nào của mô phỏng. Test: hai lịch tạo độc lập với cùng (seed, N) cho cùng đồ thị ở mọi round, gọi theo bất kỳ thứ tự nào (SPEC REQ-01). Khi đó mỗi node thật tự tính được toàn bộ matching và đọc ra đối tác của mình. Tri thức chung mà việc này cần (id, N, đồng hồ round, seed chung) được liệt kê ở ADR-002.

## Alternatives Considered

### Một điều phối viên trung tâm chọn matching mỗi round
- Ưu: dễ cài.
- Nhược: là một server trá hình, trái REQ-01.
- Kết luận: không chọn.

### Lịch ngẫu nhiên có trạng thái (một RNG dùng chung bị "rút" dần sau mỗi lần gọi)
- Ưu: đơn giản.
- Nhược: gọi `graph_at` thêm lần (ví dụ từ Observer) làm đổi lịch, tức đổi quỹ đạo huấn luyện; không tái lập được.
- Kết luận: không chọn (T-INV-12).

### Seed thô `[seed, round_idx]` truyền thẳng vào `default_rng`
- Ưu: ngắn gọn.
- Nhược: có thể trùng entropy với luồng khác cũng dùng dạng `[seed, k]` (ví dụ luồng batch của node k), làm hai luồng ngẫu nhiên tương quan với nhau.
- Kết luận: không chọn; dùng spawn_key có ID luồng cố định.

### Gossip bất đồng bộ theo đồng hồ Poisson (Boyd et al. 2006)
- Ưu: đúng mô hình randomized gossip gốc.
- Nhược: bất đồng bộ; khó tất định và khó so sánh với các thuật toán đồng bộ.
- Kết luận: phần mở rộng sau M4 (SPEC §1.4).

### Chỉ gossip trên $K_N$, không có T3b
- Ưu: ít lượt chạy hơn.
- Nhược: so sánh chi phí theo byte với ring và ER không công bằng, vì $K_N$ được chọn đối tác bất kỳ.
- Kết luận: không chọn; T3b ở Tier 2 (E2b).

## Consequences
- Phi tập trung thật (mỗi node tự tính đối tác), tất định và tái lập được; `graph_at` là hàm thuần (T-INV-12).
- Mỗi W của gossip đối xứng, ngẫu nhiên kép và $W^2=W$. Cận $\rho^t$ chỉ áp dụng cho topology tĩnh; gossip chỉ được kiểm theo kỳ vọng (T-INV-1).
- Chi phí với N = 20: T3 gửi đúng 20 message mỗi round (3.74 MB), ít hơn cả ba topology chính còn lại: ring 40, FC 380, ER trung bình khoảng 115. Với ER, KH §3.3 ghi 108 vì đó là đồ thị của seed 0; đo lại trên 2,000 đồ thị ER(20, 0.3) liên thông cho 72–154, tức 2·|E| của từng đồ thị. T3b gửi tối đa 20 (T-INV-6).
- Hạn chế cần nêu trong báo cáo: T3 giả định mọi node biết địa chỉ của mọi node khác.
- Hình P8 vẽ gossip riêng theo tốc độ trộn kỳ vọng, có ghi chú, không đặt chung trục spectral gap với các topology tĩnh.
