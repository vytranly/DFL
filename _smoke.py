"""Smoke test: exp1-4 default path (size_alpha=None, 1 round) khong loi."""
import copy, torch, numpy as np
from config import DEFAULT_CONFIG
from data_generator import build_dataloaders
from topology import create_topology
from simulator import DFLSimulator


def mk(**o):
    c = copy.deepcopy(DEFAULT_CONFIG); c.update(o); c['save_dir'] = './results'; return c


cases = [
    dict(topology='ring',            algorithm='gossip',  mu=0.0),
    dict(topology='fully_connected', algorithm='fedavg',  mu=0.0),
    dict(topology='scale_free',      algorithm='fedprox', mu=0.1),
    dict(topology='lattice',         algorithm='gossip',  mu=0.0, lattice_k=4),
]

for i, c in enumerate(cases):
    torch.manual_seed(42); np.random.seed(42)
    # size_alpha KHONG truyen -> None -> dirichlet() path (exp1-4 mac dinh)
    cfg = mk(num_clients=5, num_rounds=1, local_epochs=1, eval_every=1, **c)
    cl, tl, ci, _ = build_dataloaders('mnist', 5, 0.5, batch_size=32,
                                      data_dir='./data')
    g = create_topology(cfg['topology'], 5, lattice_k=cfg.get('lattice_k', 4))
    sim = DFLSimulator(cfg, cl, tl, g, ci)
    tr = sim.run()
    s = tr.summary()
    print(f"SMOKOK case{i}: {cfg['topology']}/{cfg['algorithm']} "
          f"f1={s.get('final_f1','NA')} size_alpha={cfg.get('size_alpha')}")

print("SMOKE_ALL_OK")
