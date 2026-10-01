# Quyết định kiến trúc (ADR)

Tách từ bản tóm tắt ở `docs/KE_HOACH_DE_TAI_2.md` §10 (P1.6), cập nhật theo `SPEC.md` và [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). Mỗi ADR theo template của skill `documentation-and-adrs`: Status / Date / Context / Decision / Alternatives Considered / Consequences.

| ADR | Quyết định | Status | Phụ thuộc câu hỏi / quyết định |
|---|---|---|---|
| [ADR-001](ADR-001-custom-pytorch-networkx-simulator.md) | Tự viết mô phỏng trên PyTorch + NetworkX, không dùng framework FL | Proposed | Q7 (đã trả lời) |
| [ADR-002](ADR-002-single-process-synchronous-transport.md) | Đơn tiến trình, đồng bộ, message-passing qua Transport | Proposed | Q1 (GVHD trả lời 2026-10-01; bổ sung bởi ADR-010) |
| [ADR-003](ADR-003-mnist-primary-dataset.md) | MNIST chính; CIFAR-10 là mở rộng ưu tiên; Fashion-MNIST là phương án kiểm chứng rẻ | Proposed | Q5, QĐ-4 |
| [ADR-004](ADR-004-smallcnn-no-batchnorm-plain-sgd.md) | SmallCNN 46,730 tham số, không BatchNorm, SGD thuần, minibatch có hoàn lại | Proposed | — |
| [ADR-005](ADR-005-metropolis-hastings-mixing.md) | Trọng số MH; ablation uniform và size-corrected; size-weighted chỉ trên FC | Proposed | — |
| [ADR-006](ADR-006-single-atc-gossip-sgd-loop.md) | Một vòng ATC tham số (τ, μ, $c_i$); τ·R = 4,000; khởi tạo chung | Proposed | A3, QS-2 |
| [ADR-007](ADR-007-evaluation-protocol.md) | Giao thức đánh giá, chọn lr, seed, thống kê, ngân sách | Proposed | Q3, QĐ-2, QS-2; η điền sau E0a |
| [ADR-008](ADR-008-results-format-and-code-version.md) | Định dạng kết quả, `code_version`, độ đo suy ra ở `analysis/` | Proposed | QS-1, QS-4, QS-7 |
| [ADR-009](ADR-009-random-pairwise-gossip-schedule.md) | Random gossip theo lịch chung trên $K_N$; thêm T3b trên cạnh ER | Proposed | — |
| [ADR-010](ADR-010-multiprocess-node-runtime.md) | Mỗi node một tiến trình (tiến trình cục bộ hoặc container Docker) qua gloo; bộ mô phỏng một tiến trình giữ lại và phải trùng từng bit | Proposed | Q1 (đã trả lời), Q10, Q12, Q13 |
| [ADR-011](ADR-011-model-complexity-ladder.md) | Thang bốn model (7,850 → 1,663,370 tham số) cho E8; SmallCNN vẫn là model lõi | Proposed | Q1 (đã trả lời), Q11 |

Q1–Q13, A1–A6, QĐ-1…5: [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). QS-1…8: `SPEC.md` §11.2.

## Quy tắc vòng đời

- **Proposed → Accepted** tại Gate G1, khi nhóm và GVHD duyệt; ghi ngày duyệt vào mục Status.
- **Khi còn Proposed:** sửa trực tiếp theo góp ý và theo câu trả lời của các câu hỏi đang mở (Q1, QS-k); lịch sử nằm trong git.
- **Khi đã Accepted:** không sửa nội dung quyết định. Tạo ADR mới (số tiếp theo, bắt đầu từ ADR-010) và đánh dấu ADR cũ "Superseded by ADR-00M". Không xóa ADR nào.
- **Ngoại lệ đã đăng ký trước:** điền η và μ vào ADR-007 sau E0a, kèm ngày và run ID của E0a.
- Đổi siêu tham số hay ngưỡng đã chốt thì phải có ADR mới và chạy lại cả nhóm thí nghiệm bị ảnh hưởng (SPEC §8).
