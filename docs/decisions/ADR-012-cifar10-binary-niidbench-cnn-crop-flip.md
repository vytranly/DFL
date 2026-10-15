# ADR-012: CIFAR-10 đọc từ bản binary bằng numpy, model CNN của NIID-Bench (62,006 tham số), augmentation crop + flip lấy ngẫu nhiên từ luồng batch của node

## Status
Proposed — soạn ngày 2026-10-15 theo yêu cầu làm code CIFAR-10 trước M4 (nhánh `binh`). Đây là việc thuộc Ask first của SPEC §8 ("phần mở rộng trước M4"); ba lựa chọn thiết kế (model, augmentation, phạm vi) được hỏi và chốt trong phiên ngày 2026-10-15. Chờ nhóm duyệt và Gate G1. Bổ sung ADR-003 và ADR-004, không thay thế chúng. ADR này chỉ quyết phần code; lưới thu gọn, ngân sách, η và ngưỡng sanity của CIFAR-10 chưa quyết (xem cuối mục Decision).

## Date
2026-10-15

## Context
- ADR-003 xếp CIFAR-10 là phần mở rộng ưu tiên sau M4, và yêu cầu có ADR mới (model cho ảnh 3×32×32 không BatchNorm, lưới thu gọn, ngân sách) trước khi bắt đầu. ADR-004 ghi CIFAR-10 cần model khác SmallCNN.
- Ràng buộc phải giữ:
  - F6: không unpickle dữ liệu ngoài. SPEC §6: dữ liệu chỉ tải qua torchvision.
  - F7, T-INV-5: mọi nguồn ngẫu nhiên đi qua Generator của `seeding.py`; RNG toàn cục của torch không đổi sau một lượt.
  - INV-6, T-INV-13: cùng cấu hình thì cùng `rounds.jsonl`; hai runtime trùng từng bit.
  - G3: không BatchNorm.
  - Lượt MNIST phải cho kết quả như cũ, từng bit.
- `torchvision.datasets.CIFAR10` đọc bản python của CIFAR-10 bằng `pickle.load` (mã nguồn torchvision 0.29 đã cài, `datasets/cifar.py` dòng 2, 83, 100), nên dùng nó là trái F6.
- Trang chính thức https://www.cs.toronto.edu/~kriz/cifar.html (đọc ngày 2026-10-15):
  - Bản binary nặng 162 MB, md5 `c32a1d4ab5d03f1284b67883e8d87530`.
  - Gồm `data_batch_1.bin` … `data_batch_5.bin` và `test_batch.bin`. Mỗi file "should be exactly 30730000 bytes long", tức 10,000 bản ghi 3,073 byte.
  - Mỗi bản ghi có 1 byte nhãn 0–9, rồi 1,024 byte kênh đỏ, 1,024 byte kênh xanh lá, 1,024 byte kênh xanh dương, theo thứ tự hàng ("row-major order").
- NIID-Bench (Li et al., ICDE 2022, arXiv 2102.02079v4), bài cung cấp các cách chia P0–P3 của dự án:
  - Model cho mọi dataset ảnh: "two 5x5 convolution layers followed by 2x2 max pooling (the first with 6 channels and the second with 16 channels) and two fully connected layers with ReLU activation (the first with 120 units and the second with 84 units)" [p.7].
  - Bảng III, FedAvg trên CIFAR-10 [p.8]: homogeneous 70.4% ± 0.2%; p_k ∼ Dir(0.5) 68.2% ± 0.7%; #C = 1 10.0% ± 0.0%; #C = 2 49.8% ± 3.3%; #C = 3 58.3% ± 1.2%; quantity q ∼ Dir(0.5) 72.0% ± 0.3%.
  - Augmentation cho CIFAR-10 trong mã nguồn (`utils.py`, nhánh main, dòng 711–721, đọc ngày 2026-10-15; bài báo không nêu): ToTensor → đệm phản chiếu 4 điểm ảnh (`F.pad(..., (4, 4, 4, 4), mode='reflect')`) → RandomCrop(32) → RandomHorizontalFlip, không có Normalize.
  - Thiết lập của bài khác dự án: 10 party có server, SGD momentum 0.9, batch 64, 10 epoch cục bộ, 50 round (DIGEST §2.4); đệm phản chiếu thay vì đệm đen; không chuẩn hóa ảnh. Vì vậy bảng chỉ là mốc tham chiếu thô.
- Tutorial CIFAR-10 chính thức của PyTorch dùng đúng mạng này, với ReLU sau mỗi conv và sau hai lớp FC đầu: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Nhóm chọn có augmentation (2026-10-15). Không có augmentation, mỗi node ở N = 20, IID chỉ có khoảng 2,250 ảnh và chạy khoảng 57 epoch cục bộ (4,000 × 32 / 2,250), nên quá khớp nặng.

## Decision

### Dữ liệu (`dataset: cifar10`)
- **Tải:** `python -m dfl download cifar10`.
  - Lệnh gọi `torchvision.datasets.utils.download_and_extract_archive` với URL bản binary và md5 của trang chính thức, giải nén vào `data/cifar-10-batches-bin/` (`/data/` đã nằm trong `.gitignore`).
  - UNVERIFIED: hàm này không có trong trang tài liệu torchvision 0.29. Theo mã nguồn đã cài (`datasets/utils.py`), nó kiểm md5 rồi mới giải nén, và không tải lại khi file đã có với md5 đúng.
  - Lớp `CIFAR10` cũng tải bằng chính hàm này (`datasets/cifar.py` dòng 139), nên quy tắc "chỉ tải qua torchvision" vẫn giữ.
- **Đọc:** bằng numpy (`np.fromfile`), không pickle. Mỗi file phải đúng 30,730,000 byte, sai thì raise. Ảnh là uint8 (n, 3, 32, 32) đúng thứ tự kênh của file; nhãn là byte đầu của mỗi bản ghi.
- **Chia:** 50k ảnh train thành 45k cho node và 5k validation phân tầng (500 ảnh mỗi lớp), cùng thủ tục và cùng luồng `VAL_SPLIT` như MNIST. 10k ảnh test là tập test chung. Tập con đánh giá 2,000 ảnh lấy như cũ (luồng `EVAL_SUBSET`).
- **Chuẩn hóa:** mean/std **từng kênh**, tính trên phần 45k của node (không gồm validation), áp cho cả ba tập. MNIST là trường hợp một kênh, nên ra đúng các giá trị cũ.
- **Dữ liệu tổng hợp `synthetic_cifar10`:** cùng shape 3×32×32, cùng cỡ với `synthetic` (2,000 ảnh train, 200 trong đó làm validation; 500 ảnh test). Dùng cho test và profile `smoke_cifar10`. Như `synthetic`, nó không bao giờ được ghi vào `results/`.

### Model `cnn_niidbench`
- Kiến trúc: Conv 5×5 (3→6) → ReLU → MaxPool 2 → Conv 5×5 (6→16) → ReLU → MaxPool 2 → FC 400 → 120 → ReLU → FC 120 → 84 → ReLU → FC 84 → 10.
- **62,006 tham số**, tức 248,024 byte mỗi message (float32).
- Không BatchNorm, không dropout. w⁰ lấy từ luồng init theo đúng quy tắc của ADR-004 và T09a.
- Cấu hình kiểm model khớp shape ảnh của dataset. `smallcnn`, `logreg`, `mlp2nn`, `cnn_mcmahan` chỉ nhận 1×28×28 (MNIST, `synthetic`); `cnn_niidbench` chỉ nhận 3×32×32 (`cifar10`, `synthetic_cifar10`).

### Augmentation `augmentation: crop_flip`
- **Phép biến đổi:** mỗi bước cục bộ, mỗi ảnh của minibatch được xử lý ba bước:
  1. đệm 4 điểm ảnh mỗi cạnh bằng điểm ảnh đen (giá trị 0 trước chuẩn hóa);
  2. cắt ngẫu nhiên một vùng 32×32, độ dời hàng và cột đều trong 0..8;
  3. lật ngang với xác suất 1/2.

  Ba bước này tương đương `RandomCrop(32, padding=4)` (mặc định `fill=0`, `padding_mode='constant'`) rồi `RandomHorizontalFlip(p=0.5)` của torchvision, áp trước ToTensor/Normalize. Test so với `torchvision.transforms.v2.functional`.
  - https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.RandomCrop.html
  - https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.RandomHorizontalFlip.html
- **Nguồn ngẫu nhiên:** Generator batch của node (luồng `BATCH`, khóa id node; B1 và B1-4k dùng `B1_BATCH`). Trong mỗi bước, các số được rút ngay sau B chỉ số minibatch: trước là 2B độ dời, sau là B cờ lật.
  - Không cần luồng mới và không đổi contract `LocalTrainer`, vì trainer vẫn chỉ dùng `model`, `data`, `generator`.
  - RNG toàn cục không bị động tới (T-INV-5). Hai runtime vẫn trùng từng bit (T-INV-13), vì cả hai dựng trainer bằng `Assembly.trainer()`.
- **Phạm vi áp dụng:** chỉ khi huấn luyện; đánh giá dùng ảnh gốc. Chỉ nhận ảnh 3×32×32, vì lật ngang làm sai nghĩa chữ số MNIST.
- **Giá trị cấu hình:** augmentation do dataset quyết định.
  - `crop_flip`: ảnh 3×32×32 (`cifar10`, `synthetic_cifar10`).
  - `none`: ảnh 1×28×28 (`mnist`, `synthetic`); đường code của MNIST giữ nguyên.
  - `null` (mặc định của `configs/base.yaml`): điền giá trị của dataset. Cấu hình chuẩn hóa ghi giá trị đã điền, như `topology.p` của ER.

  Ghi giá trị khác của dataset thì cấu hình báo lỗi. Nhờ vậy một lượt CIFAR-10 không thể lặng lẽ chạy thiếu augmentation, kể cả khi `dataclasses.replace` đổi dataset mà quên đặt lại trường này thành `None` (review T28, R1). Muốn chạy CIFAR-10 không augmentation (ablation) thì cần ADR mới (xem Alternatives).

### Profile
- `configs/profiles/smoke_cifar10.yaml`: như `smoke`, với `synthetic_cifar10` và `cnn_niidbench`.
- `configs/profiles/cifar10_fast.yaml`: như `fast`, với `cifar10` và `cnn_niidbench`.
- η = 0.05 của `configs/base.yaml` chỉ là giá trị giữ chỗ.

### Chưa quyết ở ADR này
- **η của CIFAR-10:** phải chọn trên validation 5k của CIFAR-10 theo quy tắc của ADR-007, trong một nhóm riêng kiểu E0a. Không dùng η của MNIST.
- **Lưới thu gọn và ngân sách:** dựa trên số đo của T28f và T13.
- **Ngưỡng sanity:** SAN-1 ≥ 98.5% chỉ dành cho MNIST. Mốc ngoài gần nhất là bảng NIID-Bench ở trên, với thiết lập khác.
- Thí nghiệm CIFAR-10 vẫn là phần mở rộng sau M4 (SPEC §1.4) và cần thẻ tiền đăng ký (R8) cùng ADR riêng cho lưới.

## Alternatives Considered

### `torchvision.datasets.CIFAR10` (bản python)
- Ưu: một dòng code, có tài liệu.
- Nhược: unpickle file tải về (F6).
- Kết luận: không chọn.

### Tự viết downloader bằng `urllib` + `tarfile(filter="data")`
- Ưu: mọi API đều có tài liệu thư viện chuẩn.
- Nhược: trái quy tắc "dữ liệu chỉ tải qua torchvision" (SPEC §6); nhiều code và test hơn để làm lại phần torchvision đã có.
- Kết luận: không chọn.

### SmallCNN 3 kênh (Conv 3→16→32, FC 800 → 64 → 10; 65,962 tham số)
- Ưu: cùng họ với model lõi của MNIST, dễ so chéo.
- Nhược: không có mốc tham chiếu ngoài.
- Kết luận: không chọn. Thêm sau được nếu cần so chéo.

### Model lớn (CNN CIFAR của McMahan et al. 2017, VGG, ResNet với GroupNorm)
- Ưu: accuracy cao hơn nhiều.
- Nhược: từ khoảng 10⁶ tham số trở lên, không vừa ngân sách CPU (ADR-003, ADR-011).
- Kết luận: không chọn.

### Không augmentation
- Ưu: pipeline giống hệt MNIST, không đụng `local_train`.
- Nhược: quá khớp nặng ở node nhỏ (xem Context).
- Kết luận: không chọn (nhóm chọn ngày 2026-10-15).

### Augmentation với luồng ngẫu nhiên riêng (luồng mới, khóa id node)
- Ưu: minibatch giống hệt giữa lượt có và không augmentation, nên ablation được ghép cặp.
- Nhược: phải đổi contract `LocalTrainer`, `Node`, engine và process runtime. Ablation augmentation không nằm trong phạm vi.
- Kết luận: không chọn. Cần ablation thì ghi ADR mới.

### Đệm phản chiếu (reflect) như mã nguồn NIID-Bench
- Ưu: giống cách làm của bài dùng làm mốc tham chiếu; không cần giá trị đệm theo kênh, vì chuẩn hóa là phép biến đổi theo từng điểm ảnh nên đệm phản chiếu trước hay sau chuẩn hóa cho cùng kết quả.
- Nhược: khác `RandomCrop(32, padding=4)` mặc định của torchvision, phép biến đổi dùng làm chuẩn để kiểm.
- Kết luận: không chọn, và ghi đây là một điểm lệch so với NIID-Bench (review T28, R5). Nhóm có thể đổi trước lượt CIFAR-10 đầu tiên vào `results/`, vì ADR còn Proposed.

### Đệm bằng 0 sau chuẩn hóa (tức màu trung bình)
- Ưu: không cần giá trị đệm theo kênh.
- Nhược: khác `RandomCrop(padding=4)` của torchvision, phép biến đổi dùng làm chuẩn để kiểm.
- Kết luận: không chọn.

### Một mean/std chung cho ba kênh
- Ưu: giữ kiểu `float` của `Datasets.mean`/`std`.
- Nhược: ba kênh của CIFAR-10 có thống kê khác nhau; chuẩn hóa theo kênh là cách làm thông thường.
- Kết luận: không chọn.

## Consequences
- **Code:**
  - `data/datasets.py`: đọc và tải CIFAR-10, chuẩn hóa theo kênh, `synthetic_cifar10`.
  - `learning/models.py`: `cnn_niidbench`.
  - `learning/augment.py` (file mới): `crop_flip`.
  - `learning/local_train.py`: móc augmentation; `None` giữ nguyên đường MNIST.
  - `config.py`: dataset, kiểm model khớp shape, trường `augmentation`.
  - `cli.py`: `download cifar10`.
  - `sim/run.py`, `sim/sweep.py`: truyền augmentation và giá trị đệm; chặn mọi dữ liệu tổng hợp khỏi `results/`.
- **run_id:** cấu hình chuẩn hóa có thêm trường `augmentation`, nên `run_id` của mọi cấu hình đổi. Chưa có lượt nào trong `results/`, nên không mất gì.
- **MNIST giữ nguyên:** `rounds.jsonl` không chứa `run_id` hay `code_version`, nên lượt MNIST phải cho cùng file như trước, từng byte. Kiểm bằng `cmp` giữa lượt chạy trước và sau thay đổi, với profile smoke và fast; kết quả ghi ở T28f.
- **`Datasets.mean`/`std`:** thành tuple theo kênh. Với MNIST tuple có một phần tử, cùng giá trị cũ. Hai trường này không nằm trong manifest.
- **Byte:**
  - 248,024 B mỗi message.
  - Với N = 20 mỗi round: ring 40 message = 9.92 MB; FC 380 message = 94.25 MB; gossip trên $K_N$ 20 message = 4.96 MB.
- **Tính toán:** khoảng 0.65 triệu phép nhân-cộng mỗi ảnh khi chạy xuôi (conv1 352,800; conv2 240,000; FC 58,920). SmallCNN trên MNIST là khoảng 1.08 triệu. Đo được 9.40 ms một bước có `crop_flip`, so với 10.36 ms của SmallCNN trên MNIST (mục Số đo).
- **RAM:**
  - Mỗi tiến trình giữ ba tập float32: 45k × 3,072 × 4 B = 553 MB (train), 61 MB (validation), 123 MB (test), tổng khoảng 737 MB. MNIST chỉ khoảng 220 MB.
  - Runtime inprocess còn giữ bản sao dữ liệu của các node (tổng 553 MB). Đo được RAM đỉnh 1.69 GB cho một lượt `cifar10_fast`.
  - Một tiến trình node của process runtime đạt RAM đỉnh 1,291 MiB (MNIST 764 MiB). Với N = 20 cần khoảng 21 × 1.29 GB ≈ 27 GB, quá RAM còn trống của máy chủ dùng chung (31 GB tổng, khoảng 20 GB trống ngày 2026-10-15). N = 8 cần khoảng 11.6 GB.
- **Augmentation:** mọi lượt CIFAR-10 dùng `crop_flip`, nên mọi so sánh trong một nhóm CIFAR-10 vẫn ghép cặp (G9): cùng seed thì cùng chuỗi số ở mọi topology và thuật toán. Một ablation không augmentation sẽ không ghép cặp được minibatch với lượt có augmentation, vì hai lượt rút số xen kẽ khác nhau.
- **SPEC** (sửa cùng ADR này, theo lựa chọn phạm vi ngày 2026-10-15): §1.4, §4, §5.1, §5.2, §6, REQ-14, §10 dòng 41.
- **Task:** T28a–T28f trong `tasks/todo.md`.

## Số đo (T28f, 2026-10-15)
Máy chủ CPU 8 nhân dùng chung, 1 luồng mỗi lượt, load average 14–18 trong lúc đo, nên thời gian chỉ là mức tham khảo. Mọi lượt ghi ra `build/` hay thư mục tạm, không vào `results/`.
- **MNIST giữ nguyên:**
  - Chạy profile `smoke` và `fast` (MNIST thật, N = 20, 1,000 bước) ở commit `1c10f2b` (trước T28b) và `8aa7f3b` (sau T28e).
  - `cmp` cho thấy `rounds.jsonl` và `summary.json` trùng từng byte ở cả hai profile.
  - `run_id` đổi (smoke: `ac88cc7953197c61` → `3c57d0a30cae3883`), đúng dự kiến vì cấu hình chuẩn hóa có thêm `augmentation`.
- **Dữ liệu thật:**
  - Archive có md5 `c32a1d4ab5d03f1284b67883e8d87530`, đúng như trang chính thức. Giải nén ra thư mục `cifar-10-batches-bin`, sáu file đều 30,730,000 byte. `python -m dfl download cifar10` chạy lại thì bỏ qua bước tải.
  - Sau khi tách: 45,000 / 5,000 / 10,000 ảnh. Nạp mất 1.9 s.
  - mean theo kênh của phần node là (0.4914, 0.4821, 0.4465), std (0.2471, 0.2437, 0.2616). Các giá trị này khớp mức thường dùng cho CIFAR-10, nên thứ tự kênh R, G, B được đọc đúng.
- **describe** (20 seed, `configs/profiles/cifar10_fast.yaml`):
  - `--check-req02` đạt trên nhãn CIFAR-10.
  - JS: Dir(0.5) 0.287; Dir(0.1) 0.554; #C = 2 0.614; #C = 1 0.758.
  - Node nhỏ nhất: 145 ảnh ở α = 0.1; 134 ở α = 0.05; 83 ở quantity β = 0.5.
- **Thời gian một bước** (B = 32, trung vị của 5 × 100 bước):
  - SmallCNN trên MNIST: 10.36 ms.
  - `cnn_niidbench` không augmentation: 8.29 ms; có `crop_flip`: 9.40 ms.
  - Riêng `crop_flip`: 0.63 ms mỗi batch.
- **Một lượt `cifar10_fast`** (ring, IID, N = 20, 1,000 bước, τ = 10, η = 0.05 giữ chỗ):
  - 4 phút 56 giây, gồm huấn luyện 190 s và đánh giá 62 s.
  - 4,000 message, 992,096,000 byte (= 4,000 × 248,024).
  - Trên 10k ảnh test: acc(w̄) 43.9%, accuracy trung bình của node 42.5%. Mất mát train vẫn đang giảm (2.30 → 1.69).
  - Đây chỉ là số đo phát triển, không phải kết quả: η chưa được chọn, và ngân sách chỉ bằng 1/4 của lưới.
- **RAM đỉnh:**
  - Một tiến trình node (nạp dữ liệu, lắp lượt, dựng node): 1,291 MiB với CIFAR-10, 764 MiB với MNIST.
  - Runtime inprocess với `cifar10_fast`: 1.69 GB.
