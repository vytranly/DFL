# Kiểm tra theo chi phí (P1.8; mẫu gốc ở docs/HANDOFF_GUIDELINE.md §5.5).
# Nguồn chuẩn là CONSTRAINTS.md: Makefile chỉ chạy lại các lệnh ghi ở đó; hai nơi lệch nhau thì CONSTRAINTS.md đúng.
#   make check-fast   sau mỗi lần sửa (vài giây)
#   make check-task   trước mỗi commit (ngân sách 90 s, chỉ cảnh báo)
#   make check-full   ở mỗi checkpoint C1–C5
#   make check-inv    T-INV-1…13 có đủ và xanh; bắt buộc từ C3 và trước mọi lượt Tier 1
# Cuối file là lối tắt cài đặt, tải dữ liệu và chạy một lượt (make help); phần đó không thuộc CONSTRAINTS.md.

.PHONY: check-fast check-task check-full check-static check-floor check-inv install-hooks
.PHONY: help venv shell download-mnist download-cifar10 smoke smoke-cifar10 fast cifar10-fast

# Dùng công cụ trong .venv nếu có, để chạy được mà không cần activate.
VENV := .venv
VENV_BIN := $(VENV)/bin/
PY ?= $(if $(wildcard $(VENV_BIN)python),$(VENV_BIN)python,python)
RUFF ?= $(if $(wildcard $(VENV_BIN)ruff),$(VENV_BIN)ruff,ruff)
LINT_IMPORTS ?= $(if $(wildcard $(VENV_BIN)lint-imports),$(VENV_BIN)lint-imports,lint-imports)
PY_DIRS := src analysis tests
# F5 (ratchet): số test thu thập được không được giảm. Chỉ được tăng; cập nhật ở mỗi checkpoint.
MIN_TESTS := 882
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

check-full: check-static check-inv
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

# Cài hook commit-msg (Conventional Commits, tiếng Anh) và mẫu thông điệp; chạy một lần sau khi clone.
# Chép vào .git/hooks thay vì đặt core.hooksPath để không che các hook có sẵn ở đó.
install-hooks:
	install -m 755 scripts/git-hooks/commit-msg "$$(git rev-parse --git-path hooks)/commit-msg"
	git config commit.template .gitmessage
	@echo "Đã cài hook commit-msg và commit.template"

# ---------------------------------------------------------------------- lối tắt: cài đặt, dữ liệu, một lượt
# Mọi lệnh gọi python của .venv qua $(PY), nên không cần activate. make không đổi được môi trường của terminal
# đang gõ (mỗi dòng lệnh chạy trong một tiến trình con), nên muốn có venv ngay trong terminal thì dùng make shell.
# Thêm cờ cho `dfl run` qua ARGS, ví dụ: make smoke ARGS="--runtime processes".
PYTHON312 ?= python3.12
ARGS ?=

help:
	@printf '%s\n' \
	  'make venv              tạo $(VENV) và cài đúng phiên bản (SPEC §4); chạy một lần' \
	  'make shell             mở shell con có $(VENV) đã activate; gõ exit để quay lại' \
	  'make download-mnist    tải MNIST vào data/' \
	  'make download-cifar10  tải CIFAR-10 bản binary vào data/cifar-10-batches-bin/ (ADR-012)' \
	  'make smoke             một lượt smoke trên dữ liệu giả cùng shape MNIST, ghi ra build/smoke' \
	  'make smoke-cifar10     một lượt smoke trên dữ liệu giả 3×32×32, ghi ra build/smoke-cifar10' \
	  'make fast              một lượt fast trên MNIST thật, ghi ra build/fast' \
	  'make cifar10-fast      một lượt fast trên CIFAR-10 thật, ghi ra build/cifar10-fast' \
	  'make check-fast | check-task | check-full | check-inv    kiểm tra theo CONSTRAINTS.md' \
	  'Thêm cờ cho dfl run: make smoke ARGS="--runtime processes"'

# Đúng lệnh cài đặt của SPEC §4: torch bản CPU trước (nếu không, pip kéo bản CUDA), rồi requirements.lock,
# rồi công cụ dev với -c để giữ phiên bản gói chạy. Gọi pip của venv nên không cần activate.
venv:
	$(PYTHON312) -m venv $(VENV)
	$(VENV_BIN)pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
	$(VENV_BIN)pip install -r requirements.lock
	$(VENV_BIN)pip install -e ".[dev]" -c requirements.lock

# Shell con dùng bash: đọc ~/.bashrc rồi activate $(VENV). Gõ exit để quay lại terminal cũ.
shell: SHELL := /bin/bash
shell:
	@test -x $(VENV_BIN)python || { echo "Chưa có $(VENV); chạy: make venv" >&2; exit 1; }
	@echo "Shell mới với $(VENV) đã activate; gõ exit để quay lại."
	@bash --rcfile <(printf '%s\n' '[ -f ~/.bashrc ] && . ~/.bashrc' '. $(VENV_BIN)activate') -i

download-mnist:
	$(PY) -m dfl download mnist

download-cifar10:
	$(PY) -m dfl download cifar10

smoke:
	$(PY) -m dfl run --config configs/profiles/smoke.yaml --out build/smoke $(ARGS)

smoke-cifar10:
	$(PY) -m dfl run --config configs/profiles/smoke_cifar10.yaml --out build/smoke-cifar10 $(ARGS)

fast:
	$(PY) -m dfl run --config configs/profiles/fast.yaml --out build/fast $(ARGS)

cifar10-fast:
	$(PY) -m dfl run --config configs/profiles/cifar10_fast.yaml --out build/cifar10-fast $(ARGS)
