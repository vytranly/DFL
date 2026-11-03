# Sổ code_version của lưới (SPEC INT-6; ADR-008)

Mọi `code_version` có lượt thuộc lưới trong `results/` có một dòng ở đây. Khi một `code_version` bị thay thế, ghi lý do (commit sửa lỗi và test tái hiện) ở cột cuối; lượt cũ thành "stale", không xóa, không trộn vào hình.

| Ngày | code_version | Commit lưới | Nhóm | Ghi chú |
|---|---|---|---|---|
| 2026-10-16 | 00fe7f198c9a156bfac710d4d2edfcea9ea1830143a43b752a9d3a2904d46461 | `50fef80` (nhánh `binh`; worktree /root/ecaep/dfl-grid) | E0a (24 lượt ok) | Bị thay ngày 2026-10-21 bằng `eff6fa19…`. Không phải sửa lỗi: người dùng duyệt đưa ADR-013 (ghi tải và thời gian theo node, commit `4dece43`, không đổi huấn luyện) vào lưới, và chạy Tier 1 + 2 trên máy khác. Lượt E0a ở máy chủ giữ nguyên, không trộn vào hình |
| 2026-10-21 | eff6fa19fd0d4f799c45d6af9c023229b69aab3b08b2f9f28aaff2460b33b41a | nhánh `grid` từ `2598ba4` (worktree /home/tun/ecaep/dfl-grid) | E0a (chạy lại) và mọi nhóm Tier 1 + 2 | Đang dùng. Máy `12th Gen Intel(R) Core(TM) i5-12400 (flags c7483486fdc8)`, 6 nhân, 12 luồng; mọi nhóm chạy trên máy này (ADR-007, G13) |
