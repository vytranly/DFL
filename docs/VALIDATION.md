# Thẩm định cuối (T25)

Ngày 2026-11-03, máy chạy lưới (12th Gen Intel Core i5-12400, Fedora). Tài liệu ghi lại phép kiểm tái lập REP-1, ba lượt review độc lập và các sửa đổi sau review. Gate G3 do nhóm duyệt.

## 1. REP-1: chạy lại hai run_id ngẫu nhiên từ clone sạch

- **Chọn lượt:** `random.Random(20261103).sample` trên các lượt `ok` của E1, E2, E3a, E3b, ra hai lượt của E1:
  - `5be35b5f02e669b3`: ring, #C = 2, seed 4;
  - `998c7676d9db1e02`: FC, Dirichlet α = 1, seed 1.
- **Lệnh:**
  ```bash
  git clone /home/tun/ecaep/DFL clean && git -C clean checkout --detach 542612134196   # git_commit trong manifest
  # cấu hình: trường "config" của manifest, ghi lại thành YAML
  PYTHONPATH=clean/src .venv/bin/python -m dfl run --config <run_id>.yaml --out out-<run_id> --data-root data
  cmp out-<run_id>/runs/<run_id>/attempt-1/rounds.jsonl results/runs/<run_id>/attempt-1/rounds.jsonl
  cmp out-<run_id>/runs/<run_id>/attempt-1/summary.json results/runs/<run_id>/attempt-1/summary.json
  ```
- **Kết quả:** **đạt** ở cả hai lượt.
  - Lượt chạy lại cho đúng run_id cũ.
  - `rounds.jsonl` và `summary.json` trùng từng byte.
  - Manifest mới có `code_version` `eff6fa19…`, `dirty: false`, `env_match: true`, cùng máy và cùng cấu hình.
- **Lệch so với mô tả của T25:** dùng lại môi trường `.venv` của máy (`env_match: true`: mọi gói đúng phiên bản trong `requirements.lock`) thay vì cài lại từ file khóa. Mạng của máy chỉ khoảng 70 KB/s (đo khi tải CIFAR-10), nên tải lại torch bản CPU mất gần một giờ. Code đọc từ clone sạch ở đúng `git_commit` (`PYTHONPATH=clean/src`), nên phép kiểm vẫn xác nhận code và cấu hình đã ghi tái lập được kết quả.

## 2. Review độc lập

Ba agent review trên nhánh `binh` ở `9e53893` (sau khi gộp `grid`); chỉ đọc code và test, không sửa.

### 2.1 Code (code-reviewer)
- **Lõi mô phỏng mới:** không có lỗi đúng/sai. Phần này gồm BA, lattice, rare_classes, vị trí theo bậc, `graph_edges`, describe, thời gian ước tính.
- **Important, đã sửa ở `5d06216`:** `hypotheses.Cell` không so runtime, nên khi có kết quả E8 (N = 8, process runtime), ô smallcnn của E8 trùng ô lưới lõi. Khi đó H1–H6 và H8 sẽ báo "hai lượt ở seed 0". Bản sửa thêm trường `runtime` vào `Cell`, kèm test có cả E1 lẫn E8.
- **Minor, đã sửa:**
  - H9, H10 chưa có đường kết luận: đã thêm ở `0658c71`.
  - Một hình lỗi giữa chừng để lại `PROVENANCE.json` cũ: giờ PROVENANCE được ghi sau mỗi hình (`5d06216`).
  - Lượt phân kỳ bị bỏ khỏi trung vị R@95 của T1 và P3: giờ được tính là +∞ (`5d06216`).

### 2.2 Test (test-engineer)
- **Phần `src/` mới:** test tốt. Coverage theo dòng của analysis: aggregate 95%, comm_cost 99%, hypotheses 93%, plots 98%.
- **Đã sửa ở `0658c71`:**
  - H9 và H10 có test theo quy tắc ADR-007, kể cả trường hợp có lượt phân kỳ.
  - Mỗi phép so chỉ đọc nhóm thí nghiệm của nó. Lý do: E9 chạy lại các ô ring, FC, ER của lưới lõi với cùng cấu hình, nên đọc chung sẽ cho hai lượt mỗi seed.
  - P14, P16 ghi số lượt phân kỳ.
  - Tên topology BA kèm m.
  - Lượt ghi trước ADR-014 có vị trí node mặc định là random.
  - Thêm test: ba bản cài JS trùng nhau; CV lý thuyết của demo khớp Monte Carlo (test cũ so một giá trị với chính nó); số hub/leaf ghi trong thẻ E9; `graph_edges` của ba vị trí không rỗng.
- **Ghi nhận, chưa sửa (thấp):**
  - Test P14–P17 chủ yếu kiểm nhãn, chưa kiểm dữ liệu vẽ.
  - `describe_shapes` chưa có test cho trường hợp không có đồ thị ER liên thông nào.
  - `degree_class(0)` trả "giữa" (chỉ gặp với `local_only`, vốn không vào P16).
  - s@x khi R@x là "≤ 100" không được đánh dấu là kiểm duyệt (cùng quy ước với bytes@x).

### 2.3 Bảo mật, phạm vi hẹp (security-auditor)
- **Không có vấn đề Critical hay High.** Không có pickle, `torch.load(weights_only=False)` hay `yaml.load` thường. CIFAR-10 đọc bằng `np.fromfile` và kiểm cỡ. Tiến trình con chạy bằng list argv, không qua shell.
- **Gloo nghe trên địa chỉ link-local của card mạng thật** (firewalld mở 1025–65535/tcp). Máy cùng mạng có thể chen vào lúc dựng mesh, làm lượt chạy hỏng hay bơm byte vào snapshot.
  - Lưới (E8) chạy với biến môi trường `GLOO_SOCKET_IFNAME=lo`, không đổi `src/` hay `code_version`. Đã kiểm trên smoke processes: chỉ nghe 127.0.0.1, `rounds.jsonl` trùng từng byte với khi không đặt biến.
  - Trên `binh`, process runtime tự đặt interface loopback cho node con và observer (`014fbeb`, `674adaa`).
- **Low, ghi vào hạn chế:**
  - Giải nén CIFAR-10 bằng `tarfile.extractall` không có `filter`. Rủi ro thấp vì md5 được kiểm trước khi giải nén.
  - `requirements.lock` ghim phiên bản nhưng không ghim hash.
  - `setuptools==78.1.0` nằm trong dải của CVE-2025-47273. Dự án không dùng `PackageIndex`, nên không khai thác được.
  - Chưa chạy `pip-audit`, vì cần mạng.

## 3. Kiểm tra tự động
- `make check-task` trên `binh` sau các sửa đổi: 1111 test xanh.
- `make check-full` (cả e2e, `check-inv`, `pip check`): chạy lại sau khi lưới xong, ghi ở mục 5.

## 4. Không làm được trên máy này
- **Docker (T22c, tùy chọn):** máy không cài Docker.
- **Demo trên laptop (T26):** đã đo trên máy chạy lưới (khoảng 155 s cho hai lượt cộng hình); chưa đo trên laptop thật của nhóm.

## 5. Kết quả cuối
(điền khi lưới xong: số lượt mỗi nhóm, SAN-5, `make check-full`)
