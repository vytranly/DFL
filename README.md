# DFL -- Decentralized Federated Learning Experiments

## Cai dat

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn networkx tqdm
```

## Chay thi nghiem

```bash
cd code_dfl_main/DFL
python main.py --exp <ten_exp> --dataset <both|mnist|cifar10>
```

## EXP A -- Non-IID alpha sweep (label skew)

| Lenh | Mo ta | Thoi gian |
|------|-------|-----------|
| `python main.py --exp exp_a1 --dataset mnist` | alpha=0.05 Non-IID cuc manh | ~1.5h |
| `python main.py --exp exp_a2 --dataset mnist` | alpha=0.1 Non-IID manh | ~1.5h |
| `python main.py --exp exp_a3 --dataset mnist` | alpha=1.0 Non-IID trung binh | ~1.5h |
| `python main.py --exp exp_a4 --dataset mnist` | alpha=50.0 Gan IID | ~1.5h |
| `python main.py --exp exp_a5 --dataset mnist` | alpha=100.0 IID hoan toan | ~1.5h |
| `python main.py --exp exp_a_combined` | Tong hop sau khi a1-a5 xong | ~30s |

## EXP B -- Quantity skew (size imbalance)

| Lenh | Mo ta | Thoi gian |
|------|-------|-----------|
| `python main.py --exp exp_b1 --dataset mnist` | Size bang nhau | ~1.5h |
| `python main.py --exp exp_b2 --dataset mnist` | Size lech nhe (diff<=1500) | ~1.5h |
| `python main.py --exp exp_b3 --dataset mnist` | Size lech trung binh (diff<=3000) | ~1.5h |
| `python main.py --exp exp_b4 --dataset mnist` | Size lech manh (size_alpha=0.1) | ~1.5h |
| `python main.py --exp exp_b_combined` | Tong hop sau khi b1-b4 xong | ~30s |

## EXP C -- Label + Quantity skew ket hop

| Lenh | Mo ta | Thoi gian |
|------|-------|-----------|
| `python main.py --exp exp_c1 --dataset mnist` | Chi label skew | ~1.5h |
| `python main.py --exp exp_c2 --dataset mnist` | Chi quantity skew | ~1.5h |
| `python main.py --exp exp_c3 --dataset mnist` | Ca 2 muc trung binh | ~1.5h |
| `python main.py --exp exp_c4 --dataset mnist` | Ca 2 muc nang nhat | ~1.5h |
| `python main.py --exp exp_c_combined` | Tong hop sau khi c1-c4 xong | ~30s |

## EXP D -- Topology sweep

| Lenh | Mo ta | Thoi gian |
|------|-------|-----------|
| `python main.py --exp exp_d1 --dataset mnist` | ring | ~1.5h |
| `python main.py --exp exp_d2 --dataset mnist` | fully_connected | ~1.5h |
| `python main.py --exp exp_d3 --dataset mnist` | random | ~1.5h |
| `python main.py --exp exp_d4 --dataset mnist` | lattice | ~1.5h |
| `python main.py --exp exp_d5 --dataset mnist` | scale_free | ~1.5h |
| `python main.py --exp exp_d_combined` | Tong hop sau khi d1-d5 xong | ~30s |

## EXP E -- Algorithm sweep (Gossip vs FedAvg vs FedProx)

| Lenh | Mo ta | Thoi gian |
|------|-------|-----------|
| `python main.py --exp exp_e1 --dataset mnist` | Gossip | ~1.5h |
| `python main.py --exp exp_e2 --dataset mnist` | FedAvg | ~1.5h |
| `python main.py --exp exp_e3 --dataset mnist` | FedProx (mu=0.1) | ~1.5h |
| `python main.py --exp exp_e_combined` | Tong hop sau khi e1-e3 xong | ~30s |

## Chay song song (khuyen nghi)

Mo nhieu terminal, moi terminal 1 sub-exp:

```bash
# Terminal 1
python main.py --exp exp_a1 --dataset mnist

# Terminal 2
python main.py --exp exp_a2 --dataset mnist

# Terminal 3
python main.py --exp exp_a3 --dataset mnist
...
```

## Output

Moi sub-exp sinh ra **10 bieu do** luu tai `results/`:
1. F1/round curves
2. Accuracy/round curves
3. Variance (std F1)/round
4. Comm cost tich luy
5. Final F1 bar + convergence round
6. Client heatmap
7. Hub vs Leaf F1
8. Per-node cost (comm + train)
9. Round time/round
10. Convergence speed

Combined sinh them **7 bieu do so sanh** + **3 bieu do MNIST vs CIFAR-10**.

## Cau truc thu muc

```
DFL/
  main.py          -- Dieu phoi thi nghiem
  trainer.py       -- CNN model + aggregation algorithms
  simulator.py     -- Vong lap DFL
  topology.py      -- Tao graph topology
  data_generator.py -- Phan vung du lieu Non-IID
  metrics.py       -- Tracking metrics
  visualizer.py    -- Ve bieu do
  config.py        -- Config mac dinh
  results/         -- Ket qua (CSV + PNG)
  data/            -- MNIST/CIFAR-10 (tu dong tai xuong)
```
