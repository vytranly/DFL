# Demo trên laptop (T26; SPEC ENG-11, QS-8)

Hai lượt DFedAvg trên MNIST, N = 20, α = 0.1: ring so với FC, 400 bước cục bộ mỗi node (τ = 20, 20 round). Mục tiêu: chạy và vẽ hình xong trong ≤ 5 phút, không cần mạng. Đây là trình diễn, không phải kết quả cho báo cáo; hình mang nhãn DRAFT.

## Chuẩn bị (một lần, cần mạng)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0   # bản của nền tảng; laptop không bắt buộc dùng requirements.lock
pip install -e .
python -m dfl download mnist                     # tải MNIST vào data/ trước buổi demo
```

Không cài từ `requirements.lock` thì manifest ghi `env_match: false`; điều này đúng như thiết kế, vì demo ghi ra `build/demo`, không ghi vào `results/` (SPEC §5.3).

## Chạy (không cần mạng)

Cây làm việc phải sạch (`git status` trống ở `src/`, `configs/`), vì sweep luôn đòi cây sạch.

```bash
time python -m dfl sweep --experiment configs/experiments/demo.yaml --out build/demo --workers 2
python -m analysis.plots --results build/demo --out build/demo/figures --draft
```

Hình nên mở khi trình bày: `build/demo/figures/P1.png` (learning curve ring so với FC), `P7.png` (phân bố lớp theo node ở α = 0.1), `P12.png` và `P13.png` (chi phí giao tiếp theo loại và theo node: FC gửi 380 message mỗi round, ring 40).

## Thời gian đo

| Máy | Ngày | Sweep 2 lượt | Vẽ hình | Ghi chú |
|---|---|---|---|---|
| i5-12400, 6 nhân (máy chạy lưới, đang chạy 4 lượt lưới song song) | 2026-10-21 | 150 s | 5 s | Tổng 2.6 phút ≤ 5 phút (ENG-11). FC đạt khoảng 96%, ring khoảng 73% ở 400 bước. Không phải laptop demo; laptop thật phải đo lại (QS-8) |
