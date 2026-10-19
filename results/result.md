PS C:\Users\User\Videos\252\261\dath\code_dfl_main\DFL> python main.py --exp all                            
>> 

============================================================
EXP 1 — Baseline: IID (alpha=100) + FC + Gossip
============================================================

=======================================================
  DFL Simulator — FULLY_CONNECTED | GOSSIP | alpha=100.0 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 32940.8 KB
  Num edges  : 10
=======================================================

  Round    5 | F1=0.9937 ± 0.0000 | Acc=0.9938 | Comm=160.8 MB                                                 
  Round   10 | F1=0.9944 ± 0.0000 | Acc=0.9944 | Comm=321.7 MB                                                 
  Round   15 | F1=0.9944 ± 0.0000 | Acc=0.9945 | Comm=482.5 MB                                                 
  Round   20 | F1=0.9948 ± 0.0000 | Acc=0.9948 | Comm=643.4 MB                                                 
  Round   25 | F1=0.9947 ± 0.0000 | Acc=0.9947 | Comm=804.2 MB                                                 
  Round   30 | F1=0.9945 ± 0.0000 | Acc=0.9946 | Comm=965.1 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 30/30 [2:53:47<00:00, 347.60s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9945
  Final Accuracy     : 0.9946
  Final Std (F1)     : 0.0
  Convergence round  : 5
  Total comm cost    : 965.06 MB
  Best F1            : 0.9948
==================================================
[Metrics] Saved -> ./results/exp1_baseline.csv
[Visualizer] Saved -> ./results/exp1_f1_curve.png
[Visualizer] Saved -> ./results/exp1_comm_cost.png

============================================================
EXP 2 — Non-IID Sweep: vary alpha (Ring + Gossip)
============================================================

  alpha = 0.1

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=0.1 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.7955 ± 0.0630 | Acc=0.8269 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9179 ± 0.0365 | Acc=0.9247 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9413 ± 0.0245 | Acc=0.9436 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9471 ± 0.0245 | Acc=0.9492 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9546 ± 0.0194 | Acc=0.9557 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9628 ± 0.0150 | Acc=0.9633 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9603 ± 0.0163 | Acc=0.9612 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9639 ± 0.0138 | Acc=0.9644 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9666 ± 0.0121 | Acc=0.9671 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9655 ± 0.0140 | Acc=0.9659 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:39:42<00:00, 263.64s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9655
  Final Accuracy     : 0.9659
  Final Std (F1)     : 0.014
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9666
==================================================
[Metrics] Saved -> ./results/exp2_alpha0.1.csv

  alpha = 0.5

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9834 ± 0.0020 | Acc=0.9836 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9893 ± 0.0010 | Acc=0.9894 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9907 ± 0.0014 | Acc=0.9908 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9913 ± 0.0013 | Acc=0.9914 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9918 ± 0.0013 | Acc=0.9919 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9914 ± 0.0013 | Acc=0.9915 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9924 ± 0.0010 | Acc=0.9925 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9924 ± 0.0008 | Acc=0.9925 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9915 ± 0.0010 | Acc=0.9917 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9919 ± 0.0008 | Acc=0.9920 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:47:42<00:00, 273.26s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9919
  Final Accuracy     : 0.992
  Final Std (F1)     : 0.0008
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9924
==================================================
[Metrics] Saved -> ./results/exp2_alpha0.5.csv

  alpha = 1.0

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=1.0 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9874 ± 0.0024 | Acc=0.9876 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9908 ± 0.0008 | Acc=0.9909 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9922 ± 0.0004 | Acc=0.9922 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9928 ± 0.0005 | Acc=0.9929 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9931 ± 0.0003 | Acc=0.9931 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9928 ± 0.0006 | Acc=0.9928 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9930 ± 0.0004 | Acc=0.9931 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9928 ± 0.0006 | Acc=0.9929 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9930 ± 0.0004 | Acc=0.9931 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9931 ± 0.0003 | Acc=0.9931 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [2:40:22<00:00, 192.45s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9931
  Final Accuracy     : 0.9931
  Final Std (F1)     : 0.0003
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9931
==================================================
[Metrics] Saved -> ./results/exp2_alpha1.0.csv

  alpha = 5.0

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=5.0 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9892 ± 0.0005 | Acc=0.9893 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9929 ± 0.0005 | Acc=0.9930 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9939 ± 0.0003 | Acc=0.9940 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9936 ± 0.0002 | Acc=0.9936 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9937 ± 0.0003 | Acc=0.9938 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9938 ± 0.0001 | Acc=0.9939 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9939 ± 0.0004 | Acc=0.9939 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9940 ± 0.0003 | Acc=0.9940 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9942 ± 0.0004 | Acc=0.9942 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9941 ± 0.0001 | Acc=0.9942 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:14:24<00:00, 233.28s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9941
  Final Accuracy     : 0.9942
  Final Std (F1)     : 0.0001
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9942
==================================================
[Metrics] Saved -> ./results/exp2_alpha5.0.csv

  alpha = 100.0

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=100.0 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9905 ± 0.0009 | Acc=0.9905 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9935 ± 0.0002 | Acc=0.9935 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9941 ± 0.0004 | Acc=0.9941 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9943 ± 0.0003 | Acc=0.9943 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9935 ± 0.0004 | Acc=0.9936 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9939 ± 0.0002 | Acc=0.9940 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9939 ± 0.0001 | Acc=0.9940 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9946 ± 0.0001 | Acc=0.9947 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9946 ± 0.0002 | Acc=0.9946 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9945 ± 0.0003 | Acc=0.9945 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:54:08<00:00, 280.98s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9945
  Final Accuracy     : 0.9945
  Final Std (F1)     : 0.0003
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9946
==================================================
[Metrics] Saved -> ./results/exp2_alpha100.0.csv
[Visualizer] Saved -> ./results/exp2_f1_curves.png
[Visualizer] Saved -> ./results/exp2_comm_cost.png
[Visualizer] Saved -> ./results/exp2_variance.png
[Visualizer] Saved -> ./results/exp2_final_bar.png

============================================================
EXP 3 — Topology Sweep (alpha=0.5, Gossip)
============================================================

  topology = ring

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9834 ± 0.0020 | Acc=0.9836 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9893 ± 0.0010 | Acc=0.9894 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9907 ± 0.0014 | Acc=0.9908 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9913 ± 0.0013 | Acc=0.9914 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9918 ± 0.0013 | Acc=0.9919 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9914 ± 0.0013 | Acc=0.9915 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9924 ± 0.0010 | Acc=0.9925 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9924 ± 0.0008 | Acc=0.9925 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9915 ± 0.0010 | Acc=0.9917 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9919 ± 0.0008 | Acc=0.9920 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:18:07<00:00, 237.76s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9919
  Final Accuracy     : 0.992
  Final Std (F1)     : 0.0008
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9924
==================================================
[Metrics] Saved -> ./results/exp3_topo_ring.csv

  topology = fully_connected

=======================================================
  DFL Simulator — FULLY_CONNECTED | GOSSIP | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 32940.8 KB
  Num edges  : 10
=======================================================

  Round    5 | F1=0.9899 ± 0.0000 | Acc=0.9900 | Comm=160.8 MB                                                 
  Round   10 | F1=0.9926 ± 0.0000 | Acc=0.9927 | Comm=321.7 MB                                                 
  Round   15 | F1=0.9931 ± 0.0000 | Acc=0.9932 | Comm=482.5 MB                                                 
  Round   20 | F1=0.9931 ± 0.0000 | Acc=0.9932 | Comm=643.4 MB                                                 
  Round   25 | F1=0.9931 ± 0.0000 | Acc=0.9932 | Comm=804.2 MB                                                 
  Round   30 | F1=0.9926 ± 0.0000 | Acc=0.9927 | Comm=965.1 MB                                                 
  Round   35 | F1=0.9941 ± 0.0000 | Acc=0.9941 | Comm=1125.9 MB                                                
  Round   40 | F1=0.9935 ± 0.0000 | Acc=0.9936 | Comm=1286.7 MB                                                
  Round   45 | F1=0.9935 ± 0.0000 | Acc=0.9936 | Comm=1447.6 MB                                                
  Round   50 | F1=0.9932 ± 0.0000 | Acc=0.9933 | Comm=1608.4 MB                                                
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:37:17<00:00, 260.74s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9932
  Final Accuracy     : 0.9933
  Final Std (F1)     : 0.0
  Convergence round  : 5
  Total comm cost    : 1608.44 MB
  Best F1            : 0.9941
==================================================
[Metrics] Saved -> ./results/exp3_topo_fully_connected.csv

  topology = random

=======================================================
  DFL Simulator — RANDOM | GOSSIP | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 23058.5 KB
  Num edges  : 7
=======================================================

  Round    5 | F1=0.9830 ± 0.0055 | Acc=0.9832 | Comm=112.6 MB                                                 
  Round   10 | F1=0.9890 ± 0.0026 | Acc=0.9891 | Comm=225.2 MB                                                 
  Round   15 | F1=0.9904 ± 0.0023 | Acc=0.9905 | Comm=337.8 MB                                                 
  Round   20 | F1=0.9911 ± 0.0017 | Acc=0.9913 | Comm=450.4 MB                                                 
  Round   25 | F1=0.9906 ± 0.0020 | Acc=0.9907 | Comm=563.0 MB                                                 
  Round   30 | F1=0.9910 ± 0.0019 | Acc=0.9912 | Comm=675.5 MB                                                 
  Round   35 | F1=0.9920 ± 0.0017 | Acc=0.9921 | Comm=788.1 MB                                                 
  Round   40 | F1=0.9919 ± 0.0019 | Acc=0.9920 | Comm=900.7 MB                                                 
  Round   45 | F1=0.9910 ± 0.0018 | Acc=0.9911 | Comm=1013.3 MB                                                
  Round   50 | F1=0.9917 ± 0.0016 | Acc=0.9918 | Comm=1125.9 MB                                                
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:14:46<00:00, 233.73s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9917
  Final Accuracy     : 0.9918
  Final Std (F1)     : 0.0016
  Convergence round  : 5
  Total comm cost    : 1125.91 MB
  Best F1            : 0.992
==================================================
[Metrics] Saved -> ./results/exp3_topo_random.csv
[Visualizer] Saved -> ./results/exp3_f1_curves.png
[Visualizer] Saved -> ./results/exp3_comm_cost.png
[Visualizer] Saved -> ./results/exp3_final_bar.png

============================================================
EXP 4 — Algorithm Sweep (alpha=0.5, Ring)
============================================================

  Gossip

=======================================================
  DFL Simulator — RING | GOSSIP | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9834 ± 0.0020 | Acc=0.9836 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9893 ± 0.0010 | Acc=0.9894 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9907 ± 0.0014 | Acc=0.9908 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9913 ± 0.0013 | Acc=0.9914 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9918 ± 0.0013 | Acc=0.9919 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9914 ± 0.0013 | Acc=0.9915 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9924 ± 0.0010 | Acc=0.9925 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9924 ± 0.0008 | Acc=0.9925 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9915 ± 0.0010 | Acc=0.9917 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9919 ± 0.0008 | Acc=0.9920 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [3:10:42<00:00, 228.85s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9919
  Final Accuracy     : 0.992
  Final Std (F1)     : 0.0008
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9924
==================================================
[Metrics] Saved -> ./results/exp4_Gossip.csv

  FedAvg

=======================================================
  DFL Simulator — RING | FEDAVG | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9816 ± 0.0036 | Acc=0.9819 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9888 ± 0.0014 | Acc=0.9889 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9900 ± 0.0013 | Acc=0.9901 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9914 ± 0.0013 | Acc=0.9915 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9919 ± 0.0011 | Acc=0.9920 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9916 ± 0.0013 | Acc=0.9917 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9918 ± 0.0012 | Acc=0.9919 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9923 ± 0.0011 | Acc=0.9924 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9916 ± 0.0010 | Acc=0.9917 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9923 ± 0.0011 | Acc=0.9924 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [2:46:44<00:00, 200.09s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9923
  Final Accuracy     : 0.9924
  Final Std (F1)     : 0.0011
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9923
==================================================
[Metrics] Saved -> ./results/exp4_FedAvg.csv

  FedProx μ=0.01

=======================================================
  DFL Simulator — RING | FEDPROX | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9782 ± 0.0030 | Acc=0.9784 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9878 ± 0.0010 | Acc=0.9879 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9894 ± 0.0015 | Acc=0.9895 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9909 ± 0.0012 | Acc=0.9910 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9912 ± 0.0013 | Acc=0.9912 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9908 ± 0.0009 | Acc=0.9909 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9919 ± 0.0013 | Acc=0.9920 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9916 ± 0.0008 | Acc=0.9916 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9916 ± 0.0010 | Acc=0.9917 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9918 ± 0.0006 | Acc=0.9919 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [1:52:25<00:00, 134.91s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9918
  Final Accuracy     : 0.9919
  Final Std (F1)     : 0.0006
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9919
==================================================
[Metrics] Saved -> ./results/exp4_FedProx_μ=0.01.csv

  FedProx μ=0.1

=======================================================
  DFL Simulator — RING | FEDPROX | alpha=0.5 | 5 clients
  Model size : 1647.0 KB
  Comm/round : 16470.4 KB
  Num edges  : 5
=======================================================

  Round    5 | F1=0.9492 ± 0.0082 | Acc=0.9498 | Comm=80.4 MB                                                  
  Round   10 | F1=0.9710 ± 0.0040 | Acc=0.9713 | Comm=160.8 MB                                                 
  Round   15 | F1=0.9795 ± 0.0012 | Acc=0.9796 | Comm=241.3 MB                                                 
  Round   20 | F1=0.9798 ± 0.0022 | Acc=0.9799 | Comm=321.7 MB                                                 
  Round   25 | F1=0.9858 ± 0.0019 | Acc=0.9859 | Comm=402.1 MB                                                 
  Round   30 | F1=0.9852 ± 0.0025 | Acc=0.9853 | Comm=482.5 MB                                                 
  Round   35 | F1=0.9875 ± 0.0014 | Acc=0.9876 | Comm=563.0 MB                                                 
  Round   40 | F1=0.9871 ± 0.0021 | Acc=0.9872 | Comm=643.4 MB                                                 
  Round   45 | F1=0.9874 ± 0.0016 | Acc=0.9876 | Comm=723.8 MB                                                 
  Round   50 | F1=0.9893 ± 0.0014 | Acc=0.9893 | Comm=804.2 MB                                                 
Rounds: 100%|███████████████████████████████████████████████████████████████| 50/50 [1:52:04<00:00, 134.49s/it]

==================================================
  KẾT QUẢ TỔNG HỢP
==================================================
  Final F1-macro     : 0.9893
  Final Accuracy     : 0.9893
  Final Std (F1)     : 0.0014
  Convergence round  : 5
  Total comm cost    : 804.22 MB
  Best F1            : 0.9893
==================================================
[Metrics] Saved -> ./results/exp4_FedProx_μ=0.1.csv
[Visualizer] Saved -> ./results/exp4_f1_curves.png
[Visualizer] Saved -> ./results/exp4_variance.png
[Visualizer] Saved -> ./results/exp4_final_bar.png
[Visualizer] Saved -> ./results/exp4_gossip_heatmap.png

[DONE] Kết quả lưu tại: ./results/