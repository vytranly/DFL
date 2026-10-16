# Sổ code_version của lưới (SPEC INT-6; ADR-008)

Mọi `code_version` có lượt thuộc lưới trong `results/` có một dòng ở đây. Khi một `code_version` bị thay thế, ghi lý do (commit sửa lỗi và test tái hiện) ở cột cuối; lượt cũ thành "stale", không xóa, không trộn vào hình.

| Ngày | code_version | Commit lưới | Nhóm | Ghi chú |
|---|---|---|---|---|
| 2026-10-16 | 00fe7f198c9a156bfac710d4d2edfcea9ea1830143a43b752a9d3a2904d46461 | nhánh `binh`, commit chứa thẻ E0a | E0a | Đăng ký trước khi chạy; chưa có lượt nào (chờ Gate G1 và nhóm duyệt `--dry-run`) |
