"""Tong hop cac bo ket qua EXP B/C/D trong thu muc nay thanh bang so lieu (Markdown + CSV).

Chay:  python phan_tich.py            (chi can thu vien chuan cua Python)
Ra:    tong_hop.csv  va  in bang Markdown ra man hinh.

Chi so:
  f1_cuoi      mean F1 giua cac client o round cuoi (round 50)
  std_cuoi     do lech chuan F1 giua cac client o round cuoi
  f1_min       F1 cua client kem nhat o round cuoi
  r075         round dau tien mean F1 >= 0.75 (dinh nghia "Convergence round" cua code, target_f1 = 0.75); -1 = khong dat
  r95          round dau tien mean F1 >= 95% F1 cuoi (de so toc do hoi tu giua moi cau hinh)
  comm_gb      tong chi phi truyen thong sau 50 round (GB)
  s_round      thoi gian trung binh moi round (s) -- PHU THUOC MAY CHAY, xem README
  hub, leaf    mean F1 cuoi cua hub node / leaf node (hub = degree > degree trung binh, theo log cua tung run)
"""
import csv, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = [
    ("khoi_tao_chung/cifar", "cu", "cifar"),
    ("khoi_tao_chung/mnist", "cu", "mnist"),
    ("khoi_tao_chung/mnist_gpu_chay_lai", "cu_gpu", "mnist"),
    ("khoi_tao_rieng_nghia/cifar", "nghia", "cifar"),
    ("khoi_tao_rieng_nghia/mnist", "nghia", "mnist"),
]


def hubs_from_log(path):
    if not os.path.exists(path):
        return None, None, None
    txt = open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    m = re.search(r"\[Topology: (\w+)\]", txt)
    h = re.search(r"Hub\s+nodes \([^)]*\): \[([^\]]*)\]", txt)
    g = re.search(r"Spectral gap = ([0-9.]+)", txt)
    hubs = [int(x) for x in h.group(1).split(",") if x.strip()] if h else []
    return (m.group(1).lower() if m else None), hubs, (float(g.group(1)) if g else None)


def summarize(folder, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, folder, f"exp_*_{tag}_result.csv"))):
        exp = os.path.basename(f)[:6]
        rows = list(csv.DictReader(open(f)))
        last = rows[-1]
        f1s = [float(r["mean_f1"]) for r in rows]
        final = f1s[-1]
        client_f1 = [float(v) for k, v in last.items() if k.startswith("f1_client_")]
        log_exp = "exp_b4" if (exp == "exp_c2" and not os.path.exists(os.path.join(HERE, folder, "logs", f"exp_c2_{tag}.log"))) else exp
        topo, hubs, gap = hubs_from_log(os.path.join(HERE, folder, "logs", f"{log_exp}_{tag}.log"))
        hub = leaf = None
        if hubs:
            hv = [client_f1[i] for i in hubs]
            lv = [v for i, v in enumerate(client_f1) if i not in hubs]
            hub, leaf = sum(hv) / len(hv), (sum(lv) / len(lv) if lv else None)
        out.append(dict(
            exp=exp, topology=topo, spectral_gap=gap, rounds=len(rows),
            f1_cuoi=round(final, 4), std_cuoi=round(float(last["std_f1"]), 4),
            f1_min=round(min(client_f1), 4) if client_f1 else None,
            acc_cuoi=round(float(last["mean_acc"]), 4),
            r075=next((int(r["round"]) for r in rows if float(r["mean_f1"]) >= 0.75), -1),
            r95=next(int(r["round"]) for r in rows if float(r["mean_f1"]) >= 0.95 * final),
            comm_gb=round(float(last["comm_cost_mb"]) / 1024, 2),
            s_round=round(sum(float(r["round_time_s"]) for r in rows) / len(rows), 1),
            hub=None if hub is None else round(hub, 4), leaf=None if leaf is None else round(leaf, 4),
        ))
    return out


def main():
    allrows = []
    for folder, run, tag in SETS:
        for r in summarize(folder, tag):
            allrows.append(dict(bo=folder, run=run, dataset=tag, **r))
    keys = list(allrows[0].keys())
    with open(os.path.join(HERE, "tong_hop.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    for folder, _, tag in SETS:
        print(f"\n### {folder}\n")
        print("| exp | topology | F1 cuoi | std | F1 min | r>=0.75 | r95 | comm GB | s/round | hub | leaf |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in allrows:
            if r["bo"] == folder:
                print(f"| {r['exp']} | {r['topology']} | {r['f1_cuoi']:.4f} | {r['std_cuoi']:.4f} | {r['f1_min']:.4f} | "
                      f"{r['r075']} | {r['r95']} | {r['comm_gb']} | {r['s_round']} | {r['hub']} | {r['leaf']} |")


if __name__ == "__main__":
    main()
