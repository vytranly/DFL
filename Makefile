# Kiểm tra theo chi phí (P1.8; mẫu gốc ở docs/HANDOFF_GUIDELINE.md §5.5).
# Nguồn chuẩn là CONSTRAINTS.md: Makefile chỉ chạy lại các lệnh ghi ở đó; hai nơi lệch nhau thì CONSTRAINTS.md đúng.
#   make check-fast   sau mỗi lần sửa (vài giây)
#   make check-task   trước mỗi commit (ngân sách 90 s, chỉ cảnh báo)
#   make check-full   ở mỗi checkpoint C1–C5
#   make check-inv    T-INV-1…13 có đủ và xanh; bắt buộc từ C3 và trước mọi lượt Tier 1

.PHONY: check-fast check-task check-full check-static check-floor check-inv

# Dùng công cụ trong .venv nếu có, để chạy được mà không cần activate.
VENV_BIN := .venv/bin/
PY ?= $(if $(wildcard $(VENV_BIN)python),$(VENV_BIN)python,python)
RUFF ?= $(if $(wildcard $(VENV_BIN)ruff),$(VENV_BIN)ruff,ruff)
LINT_IMPORTS ?= $(if $(wildcard $(VENV_BIN)lint-imports),$(VENV_BIN)lint-imports,lint-imports)
PY_DIRS := src analysis tests
# F5 (ratchet): số test thu thập được không được giảm. Chỉ được tăng; cập nhật ở mỗi checkpoint.
MIN_TESTS := 34
# T1: ngân sách thời gian của check-task; vượt thì chỉ cảnh báo, không chặn.
TASK_BUDGET_S := 90

check-fast: check-static
	$(PY) -m pytest -q tests/unit

check-task:
	@start=$$(date +%s); \
	$(MAKE) --no-print-directory check-static && \
	$(PY) -m pytest -q tests/unit tests/integration --cov --cov-report=term-missing:skip-covered; \
	status=$$?; elapsed=$$(( $$(date +%s) - start )); \
	echo "check-task: $${elapsed} s (ngân sách $(TASK_BUDGET_S) s)"; \
	if [ $$elapsed -gt $(TASK_BUDGET_S) ]; then echo "CẢNH BÁO T1: check-task vượt ngân sách $(TASK_BUDGET_S) s"; fi; \
	exit $$status

check-full: check-static
	$(PY) -m pytest -q --cov --cov-report=term-missing:skip-covered --durations=5
	$(PY) -m pip check

check-static: check-floor
	$(RUFF) check .
	$(RUFF) format --check .
	$(LINT_IMPORTS) --no-logo

# Sàn F2–F7 (phần ruff không kiểm được) và ratchet F5; xem CONSTRAINTS.md.
check-floor:
	@fail=0; \
	if grep -rnE --include='*.py' 'pytest\.(skip|xfail|importorskip)\(|pytest\.mark\.(skip|skipif|xfail)' $(PY_DIRS) | grep -v 'lý do:'; then \
	  echo "✗ F2: skip/xfail thiếu 'lý do:' (và phải có dòng trong bảng Ngoại lệ)"; fail=1; fi; \
	if grep -rnE --include='*.py' '#[[:space:]]*(noqa|type:[[:space:]]*ignore|pragma:[[:space:]]*no[[:space:]]*cover)' $(PY_DIRS) | grep -v 'lý do:'; then \
	  echo "✗ F3: suppression thiếu 'lý do:'"; fail=1; fi; \
	if grep -rnE --include='*.py' 'NotImplementedError' src analysis; then \
	  echo "✗ F4: còn stub NotImplementedError"; fail=1; fi; \
	if grep -rnE --include='*.py' '^[[:space:]]*\.\.\.[[:space:]]*$$' src analysis | grep -v '^src/dfl/contracts\.py:'; then \
	  echo "✗ F4: thân hàm chỉ có '...' ngoài src/dfl/contracts.py"; fail=1; fi; \
	if grep -rnE --include='*.py' 'weights_only[[:space:]]*=[[:space:]]*False' $(PY_DIRS); then \
	  echo "✗ F6: torch.load(..., weights_only=False)"; fail=1; fi; \
	if grep -rnE --include='*.py' 'torch\.manual_seed' src; then \
	  echo "✗ F7: RNG toàn cục torch.manual_seed trong src/"; fail=1; fi; \
	if grep -rnE --include='*.py' '(default_rng|SeedSequence|Generator)\(' src | grep -v '^src/dfl/seeding\.py:'; then \
	  echo "✗ F7: tạo Generator/SeedSequence ngoài src/dfl/seeding.py"; fail=1; fi; \
	n=$$($(PY) -m pytest -q --collect-only tests 2>/dev/null | tail -1 | grep -oE '^[0-9]+'); \
	if [ "$${n:-0}" -lt $(MIN_TESTS) ]; then \
	  echo "✗ F5: thu thập được $${n:-0} test, ít hơn MIN_TESTS = $(MIN_TESTS)"; fail=1; fi; \
	if [ $$fail -eq 0 ]; then echo "Sàn: đạt ($${n} test)"; fi; \
	exit $$fail

# Tên test bất biến phải có dạng test_t_inv_<k>_<hành vi> (SPEC §7).
check-inv:
	@ids=$$($(PY) -m pytest -q --collect-only tests 2>/dev/null | grep -oE 'test_t_inv_[0-9]+_' | sort -u); \
	missing=""; \
	for k in 1 2 3 4 5 6 7 8 9 10 11 12 13; do \
	  echo "$$ids" | grep -qx "test_t_inv_$${k}_" || missing="$$missing T-INV-$$k"; \
	done; \
	if [ -n "$$missing" ]; then echo "✗ Thiếu test bất biến:$$missing"; exit 1; fi; \
	$(PY) -m pytest -q -k "test_t_inv_" tests
