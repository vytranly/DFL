# HANDOFF GUIDELINE — Đề 2: Non-IID × Decentralized Federated Learning

Quy trình prompt, context pack, và cách dùng **Skills · Agent Personas · Reference Checklists** cho 4 pha:
**Discovery, Requirements & Design → Task Breakdown → Implementation & Testing → Code Review & Thẩm định**

| Mục | Nội dung |
|---|---|
| Phiên bản | 0.1 — 2026-09-29 |
| Người dùng tài liệu | (1) Thành viên nhóm điều khiển Claude Code; (2) chính agent Claude Code, đọc theo từng mục được chỉ định |
| Tài liệu gốc | Kế hoạch & kiến trúc: `docs/KE_HOACH_DE_TAI_2.md` (ký hiệu **§x.y**) · Đề: `docs/ĐỀ_TÀI_ĐỒ_ÁN_TỔNG_HỢP_HK261.pdf` (Đề 2) · Bản nháp nhóm: `docs/đath1.docx` · Bài báo: `docs/research/` |
| Danh mục skill gốc | `/root/aidb-healthcheck/binhpt50/docs/backup/profilling/docs/skills/agent-skills.md` |
| Ký hiệu | **§x.y** = mục trong kế hoạch · **H§x.y** = mục trong tài liệu này · **P1.5** = prompt số 5 của pha 1 |

---

## 0. Cách dùng tài liệu này

1. **Mỗi pha chạy trong một (hoặc vài) phiên Claude Code riêng.** Mở phiên → dán prompt khởi động H§3.0 → dán prompt của bước cần làm.
2. **Chỉ nạp đúng các mục được chỉ định** trong prompt. Không nạp lại PDF đề tài/bài báo vào mỗi phiên; agent dùng context pack ở H§2. Theo skill `context-engineering`: mỗi task chỉ nên có khoảng < 2,000 dòng ngữ cảnh tập trung; cắt phiên ở ranh giới task khi ngữ cảnh đã dùng khoảng 75%.
3. **Gate là quyết định của người**, không để agent tự duyệt. Câu trả lời mơ hồ ("chắc được", "nghe ổn") **không** tính là duyệt.
4. **Mọi đầu ra phải thành file trong repo** (SPEC, ADR, plan, log phiên). Không để quyết định chỉ nằm trong hội thoại.
5. Luồng tổng thể:

```text
Pha 1: Discovery ─► Requirements & Design ─G1─► Pha 2: Task Breakdown ─G2─► Pha 3: Implementation & Testing (C1…C5) ─► Pha 4: Code Review & Thẩm định ─G3─► nộp
         (G0 giữa 1A và 1B)
```

---

## 1. Bản đồ công cụ

### 1.1 Nơi cài đặt và cách gọi (trên máy hiện tại)

| Loại | Vị trí | Cách gọi |
|---|---|---|
| Skills (25) | Plugin `agent-skills@addy-agent-skills` v0.6.11 (bật trong `~/.claude/settings.json`); mã nguồn ở `/root/.claude/plugins/marketplaces/addy-agent-skills/skills/` | Nói "Dùng skill `<tên>`", hoặc gọi slash command cùng tên ở `~/.claude/commands/<tên>.md`, ví dụ `/doubt-driven-development` |
| Slash command theo vòng đời | `/root/.claude/plugins/marketplaces/addy-agent-skills/.claude/commands/` | `/spec` · `/plan` · `/build` · `/build auto` · `/test` · `/review` · `/code-simplify` · `/ship` · `/constraints` (nếu trùng tên với lệnh khác: `/agent-skills:spec` …) |
| Agent personas | `~/.claude/agents/{code-reviewer,test-engineer,security-auditor,web-performance-auditor}.md` | "Dùng subagent `code-reviewer` để …" (Agent tool, `subagent_type`) |
| Reference checklists | `/root/.claude/plugins/marketplaces/addy-agent-skills/references/*.md` | Trỏ agent tới đường dẫn cụ thể trong prompt |
| Subagent có sẵn của Claude Code | — | `Explore` (tra cứu chỉ đọc) · `Plan` (thiết kế chỉ đọc) · `general-purpose` (reviewer ngữ cảnh mới) |
| Skill tài liệu | `docx`, `anthropic-skills:pdf` | Sửa `đath1.docx` (P1.4); đọc PDF khi thật cần |

Máy hiện tại **chưa có** `gemini`/`codex` CLI, nên bước cross-model của `doubt-driven-development` phải chạy thủ công hoặc ghi rõ là đã bỏ qua.

### 1.2 Skills theo pha (✅ chính · ◐ phụ · — không dùng)

| Skill | 1A Discovery | 1B Req & Design | 2 Task | 3 Impl & Test | 4 Review & Thẩm định | Dùng để làm gì trong dự án này |
|---|---|---|---|---|---|---|
| `using-agent-skills` | ✅ | ◐ | ◐ | ◐ | ◐ | Đầu mỗi phiên: chọn skill phù hợp; áp 6 hành vi cốt lõi (nêu giả định, dừng khi mâu thuẫn, phản biện, giữ đơn giản, đúng phạm vi, kiểm chứng) |
| `interview-me` | ✅ | | | | | Chốt giả định A1–A6 và câu hỏi Q1–Q7, hỏi từng câu một |
| `idea-refine` | ◐ | | | | | Chọn phần mở rộng (E5–E7, T22–T23) khi có thời gian |
| `spec-driven-development` | | ✅ | | | | Viết `SPEC.md` (6 mục lõi + capability map) |
| `constraint-driven-development` | | ✅ | | ◐ | ◐ | Viết `CONSTRAINTS.md`; `/constraints guard` trước review |
| `documentation-and-adrs` | | ✅ | | ◐ | ✅ | ADR-001…009, README, báo cáo |
| `api-and-interface-design` | | ✅ | | ◐ | | `src/dfl/contracts.py` (contract-first) |
| `source-driven-development` | ◐ | ✅ | | ✅ | | Đối chiếu API torch/torchvision/networkx/numpy trên tài liệu chính thức, có URL |
| `doubt-driven-development` | | ✅ | ◐ | ✅ (T10) | ✅ | Phản biện thiết kế, engine, và các nhận định khoa học |
| `planning-and-task-breakdown` | | | ✅ | ◐ | | `tasks/plan.md`, `tasks/todo.md` |
| `context-engineering` | ✅ | ◐ | ◐ | ✅ | ◐ | CLAUDE.md, nạp ngữ cảnh đúng lúc, ranh giới phiên có thể khởi động lại |
| `incremental-implementation` | | | | ✅ | | Lát mỏng; mỗi task một commit; hệ thống luôn ở trạng thái chạy được |
| `test-driven-development` | | | | ✅ | ◐ | RED → GREEN → REFACTOR; Prove-It; test bất biến T-INV |
| `debugging-and-error-recovery` | | | | ✅ | ◐ | Khi test hoặc sanity check thất bại (stop-the-line) |
| `git-workflow-and-versioning` | ✅ | | | ✅ | ✅ | `git init`, commit nguyên tử, tag `v1.0` |
| `observability-and-instrumentation` | | | | ✅ | | Chuyển thành *log thí nghiệm có cấu trúc*: JSONL, manifest, `run_id` trong mọi dòng log |
| `performance-optimization` | | | | ✅ (T13) | ◐ | Đo trước (MEASURE → IDENTIFY → FIX → VERIFY → GUARD); chỉ phần backend/đo đạc |
| `code-review-and-quality` | | | | ◐ | ✅ | `/review` 5 trục |
| `code-simplification` | | | | | ✅ | `/code-simplify` sau review |
| `security-and-hardening` | | | | | ◐ | Phạm vi hẹp: phụ thuộc, deserialization an toàn, secrets, thao tác xóa/ghi đường dẫn |
| `ci-cd-and-automation` | | | | ◐ | ◐ | Tùy chọn: GitHub Actions chạy `make check-full` |
| `shipping-and-launch` | | | | | ◐ | `/ship` bản điều chỉnh: tag, đóng băng kết quả, hướng dẫn tái lập (không có deploy) |
| `frontend-ui-engineering` | — | — | — | — | — | Không có UI |
| `browser-testing-with-devtools` | — | — | — | — | — | Không có web app |
| `deprecation-and-migration` | — | — | — | — | — | Không có hệ thống cũ cần gỡ |

### 1.3 Agent personas và subagent có sẵn

| Persona | Vai trò | Dùng ở | Gọi qua | Ghi chú cho dự án này |
|---|---|---|---|---|
| `code-reviewer` | Senior Staff Engineer, review 5 trục | Checkpoint pha 3; pha 4 | `/review`, `/ship` | Bổ sung bộ câu hỏi riêng ở P4.1 (bất biến P2P, snapshot, đếm byte, rò rỉ test) |
| `test-engineer` | QA, chiến lược test, Prove-It | Thiết kế T-INV (P3.3); coverage ở C3, C5 (P3.7) | `/test`, `/ship` | Ưu tiên edge case: N = 2, node chỉ có 10 mẫu, α rất nhỏ, N lẻ trong gossip |
| `security-auditor` | Security engineer | Một lượt cuối (P4.4) | `/ship` | Không có auth/web nên phần lớn OWASP không áp dụng; nêu rõ phạm vi hẹp khi gọi |
| `web-performance-auditor` | Core Web Vitals | — | `/webperf` | **Không áp dụng** (không có web app) |
| `Explore` (built-in) | Tra cứu chỉ đọc | 1A (đọc bài báo), 3 (tìm trong repo) | — | Pattern "Research isolation": chỉ trả về bản tóm tắt |
| `Plan` (built-in) | Thiết kế chỉ đọc | 1B, 2 | — | So sánh phương án trước khi viết ADR |
| `general-purpose` | Reviewer ngữ cảnh mới | P1.9, P3.3, P4.3 | — | Dán nguyên văn prompt đối kháng của `doubt-driven-development` |

### 1.4 Reference checklists

| Checklist | Dùng ở pha | Cách dùng trong dự án này |
|---|---|---|
| `definition-of-done.md` | 1B, 2, 3, 4 | Dùng nguyên văn (Correctness, Quality, Integration, Documentation, Ship-readiness) + phần bổ sung nghiên cứu ở H§4.2 |
| `testing-patterns.md` | 2, 3 | Ví dụ viết cho JS/TS → chỉ lấy nguyên tắc (Arrange-Act-Assert, tên test mô tả hành vi, hạn chế mock, anti-pattern) và áp vào **pytest**; bỏ phần React/E2E trình duyệt |
| `observability-checklist.md` | 3 | Chỉ lấy structured logging và gắn định danh vào mọi dòng log (`run_id`, `round`); bỏ alerting/tracing/runbook production |
| `performance-checklist.md` | 3 (T13) | Chỉ phần backend và nguyên tắc đo trước; bỏ Core Web Vitals |
| `security-checklist.md` | 1B (Boundaries), 4 | Chỉ: secrets, kiểm tra phụ thuộc, deserialization, thao tác xóa/ghi đường dẫn (bảo vệ `results/`) |
| `orchestration-patterns.md` | 1–4 | Dùng pattern 1 (gọi persona trực tiếp), 2 (slash command một persona), 3 (fan-out `/ship`), 4 (pipeline do người điều phối), 5 (Explore). Tránh anti-pattern A–D |
| `accessibility-checklist.md` | — | Không áp dụng (không có UI) |

### 1.5 Quy tắc điều phối (từ `docs/agents.md` và `orchestration-patterns.md`)

- **Người dùng là người điều phối.** Chuỗi `/spec → /plan → /build → /test → /review → /ship` do người chạy tuần tự, có duyệt giữa các bước.
- **Persona không gọi persona.** Nếu `code-reviewer` thấy cần kiểm tra bảo mật sâu hơn, nó ghi đề xuất trong báo cáo; người quyết định có gọi `security-auditor` hay không.
- **Chỉ fan-out song song khi các nhánh độc lập** (ví dụ `/ship`: 3 persona cùng đọc một diff). Gọi các nhánh trong **cùng một lượt** để chạy song song.
- **Không đưa `doubt-driven-development` vào persona.** Skill này chạy ở phiên chính, vì nó cần sinh subagent.

---

## 2. Context pack (agent đọc phần này thay cho PDF)

### 2.1 Yêu cầu của đề (trích Đề 2) và nơi đáp ứng

| ID | Yêu cầu (trích/diễn giải từ đề) | Thiết kế | Kiểm chứng |
|---|---|---|---|
| REQ-01 | "Xây dựng môi trường mô phỏng DFL (không có server trung tâm)"; "Mô phỏng DFL: Xây dựng hệ thống peer-to-peer" | §4, INV-1…6, ADR-001/002/009 | T-INV-2, T-INV-4, T-INV-12 |
| REQ-02 | "Tạo ra các phân phối dữ liệu với mức độ Non-IID khác nhau (từ gần IID đến cực kỳ Non-IID) … Dirichlet distribution, pathological partition, quantity-based skew, label skew"; "α Dirichlet từ 0.1 → 100" | §3.2 P0–P4 | T-INV-8, P7 |
| REQ-03 | Đo "độ chính xác mô hình (accuracy / F1-score)" | §3.6 | T-INV-10 |
| REQ-04 | Đo "tốc độ hội tụ (convergence speed)" | §3.6 ($R@x$, AUC) | test ở T15 |
| REQ-05 | Đo "chi phí giao tiếp"; "Đo communication overhead" (được **tô vàng** trong đề) | §3.6, CommLedger | T-INV-6, P6 |
| REQ-06 | Đo "độ ổn định của quá trình huấn luyện" | §3.6 (độ dao động, consensus distance) | P5 |
| REQ-07 | "So sánh hiệu năng giữa các topology khác nhau (ring, fully-connected, random gossip…) dưới cùng điều kiện Non-IID"; "ít nhất 2–3 loại: Ring, Fully-connected, Random graph" | §3.3 T1–T4, E2 | T-INV-9, P3 |
| REQ-08 | "Thuật toán tổng hợp: FedAvg (hoặc biến thể Gossip Learning), FedSGD, FedProx" | §3.4, ADR-006, E3 (quét τ, μ) | T-INV-3, T-INV-7, P9 |
| REQ-09 | "Đánh giá: accuracy theo round, communication volume, variance giữa các client" | §3.6 (acc toàn cục + acc^loc, đo trước/sau tổng hợp), §4.7 | P1, P4, P6, T-INV-11 |
| REQ-10 | "Biểu đồ so sánh rõ ràng ảnh hưởng của Non-IID" | §3.8 P1–P9 | T16 |
| REQ-11 | "Cần vẽ lại kiến trúc tổng quan của toàn hệ thống" | §4.1–§4.3 | Gate G1 |
| REQ-12 | "Xây dựng baseline: triển khai DFL với dữ liệu IID"; "cố định số client, model, learning rate…; thay đổi mức Non-IID và topology" | §3.5, §3.7 (E0) | §5.3 |
| REQ-13 | "Đưa ra nhận xét và khuyến nghị về việc lựa chọn topology / chiến lược aggregation khi dữ liệu Non-IID" | §2.1 O4, T24 | P4.3 (kiểm toán nhận định) |
| REQ-14 | Dataset: "MNIST / Fashion-MNIST (dễ quan sát); CIFAR-10 (thách thức hơn)"; nhóm đã chốt PyTorch + torchvision + MNIST | ADR-003 | E6 |

### 2.2 Tài liệu nghiên cứu — lấy gì (chi tiết ở §1.3, §12)

| Tài liệu | Lấy gì cho implementation |
|---|---|
| NIID-Bench (Li et al., ICDE 2022) | Thuật toán chia `noniid-labeldir` (có bước "balance"), `noniid-#label{k}` (nhãn đầu i mod 10), `iid-diff-quantity`; vòng lặp lấy mẫu lại theo `min_size` (dự án dùng 64) |
| Palmieri et al. 2024 | Topology và vị trí dữ liệu tương tác với nhau → gán partition vào node bằng hoán vị theo seed; ý tưởng E7 |
| Koloskova et al. 2020; Kong et al. 2021 | Spectral gap và consensus distance là đại lượng giải thích chính (H2, H3) |
| Hsieh et al. 2020 | Không dùng BatchNorm |
| Kalra et al. 2023 (ProxyFL) | Baseline Regular/Joint; mean ± std qua 5 seed; đồ thị mũ; skew kiểu $p_{major}$ |
| Zhou et al. 2023 | Chỉ làm related work; không dùng số liệu baseline |
| Lian 2017 / Sun 2022 / Li 2020 / McMahan 2017 | Định nghĩa D-PSGD, DFedAvg, FedProx, FedAvg/FedSGD |
| Xiao & Boyd 2004; Boyd et al. 2006 | Trọng số Metropolis–Hastings; randomized gossip |

### 2.3 Quyết định và số liệu đã chốt (sửa chỉ qua ADR mới)

- **Stack**: Python 3.12 · torch 2.14.0 · torchvision 0.29.0 · networkx 3.x · numpy 2.x · pandas · matplotlib · pyyaml · pytest · pytest-cov · ruff · scikit-learn (chỉ trong test).
- **Mô phỏng**: đơn tiến trình, đồng bộ, message-passing qua `Transport` (in-memory, không deep-copy) + `CommLedger`; vòng Gossip-SGD kiểu ATC tham số $(\tau, \mu, c_i)$; khởi tạo chung. Observer đo **trước** tổng hợp (drift) và **sau** tổng hợp (acc, acc^loc, macro-F1, consensus distance, acc($\bar w$)).
- **Thuật toán**: tổng số bước cục bộ τ·R = 4,000 cố định, η dùng chung cho mọi τ. D-PSGD (τ = 1, R = 4,000) · DFedAvg (τ = 20, R = 200) · DFedProx (τ = 20: μ ∈ {0.1, 1}, tức $\eta\mu\tau$ ∈ {0.1, 1}; τ = 50: μ = 1, tức $\eta\mu\tau$ = 2.5) · quét τ ∈ {1, 5, 20, 50}. Baseline: B1 centralized (10 epoch), B2 local-only ($W=I$), B3 FedAvg có server với trọng số $n_k/n$ (chính là các lượt E1c).
- **Mặc định**: N = 20; B = 32, minibatch **có hoàn lại**; SGD, η chọn ở E0a theo quy tắc "tốt nhất trong trường hợp xấu nhất" trên validation 5k (dự kiến 0.05); SmallCNN **46,730** tham số (0.187 MB/message); trọng số Metropolis–Hastings; **5 seed** cho E0b-DFedAvg/E1/E2, **3 seed** cho baseline và ablation; lịch đánh giá theo mốc bước cục bộ (20, 40, …, 1,000, rồi mỗi 100 bước; đánh giá ở round đầu tiên vượt mốc) trên tập con test 2,000 ảnh; đánh giá cuối trên 10,000 ảnh lưu riêng; số luồng CPU cố định và ghi vào manifest.
- **Chia dữ liệu**: theo thuật toán NIID-Bench (Dirichlet có bước "balance"; #C=k với nhãn đầu i mod 10, cần N ≥ 10); khác biệt có chủ đích: **min_size = 64**, lấy mẫu lại khi tổng xác suất bằng 0, dùng numpy Generator; partition gán vào vị trí node bằng hoán vị theo seed.
- **Spectral gap (N = 20)**: ring 0.033 · ER p = 0.3 trung bình 0.198 (0.085–0.310 theo seed) · gossip trên $K_N$ 0.312 (kỳ vọng) · FC 1.000. **Message/round**: 40 · ~108 · 20 · 380.
- **Thang Non-IID (JS, bit)**: IID 0.001 → α = 100: 0.002 → α = 1: 0.19 → α = 0.5: 0.28 → α = 0.1: 0.55 → α = 0.05: 0.63 ≈ #C = 2: 0.61 → #C = 1: 0.76. **Lệch mục tiêu (TV)**: α = 0.1: 0.10; α = 0.05: 0.13; #C = 2: 0.07; quantity β = 0.5: 0.009.
- **Thí nghiệm**: Tier 1 (E0a, E0b, E1, E2, E3a) = 176 lượt; Tier 2 (E1c, E2b, E3b, E4) = 87 lượt (+12 có điều kiện); E5–E7 là mở rộng (§3.7).
- **Kết quả**: `code_version` = hash nội dung `src/` + file khóa phụ thuộc; mỗi lần chạy là một attempt trong `results/<exp>/<run_id>/attempt-<k>/`; độ đo suy ra ($R@x$, AUC, độ dao động) do `analysis/` tính từ `rounds.jsonl`.

### 2.4 Bẫy đã biết (known gotchas) — agent phải kiểm tra

| # | Bẫy | Cách tránh |
|---|---|---|
| G1 | Hiểu ngược chiều α (bản nháp ghi sai) | α nhỏ = Non-IID mạnh. Kết quả "accuracy tăng khi α giảm" là dấu hiệu lỗi, không phải phát hiện |
| G2 | Cập nhật model tại chỗ khi tổng hợp | Dùng snapshot/double-buffer (INV-3, T-INV-4) |
| G3 | BatchNorm hoặc buffer số nguyên | Không dùng BatchNorm; không trộn buffer số nguyên (ví dụ `num_batches_tracked`) |
| G4 | Đồ thị không liên thông | Erdős–Rényi phải lấy mẫu lại đến khi liên thông; gossip liên thông theo hợp các round, không phải từng round |
| G5 | Ma trận trộn sai | Không âm, tổng hàng bằng 1; MH đối xứng và ngẫu nhiên kép; uniform không ngẫu nhiên kép trên đồ thị không đều (hội tụ về trung bình có trọng số theo bậc) |
| G6 | Sai anchor của FedProx | Anchor = $w_i^t$ của chính node ở đầu round, clone + detach, không có gradient |
| G7 | Đếm byte rải rác hoặc đếm cả self-loop | Chỉ đếm ở Transport; không có message node tự gửi cho mình |
| G8 | Chọn siêu tham số trên tập test | Dùng validation 5k tách từ train |
| G9 | So sánh không ghép cặp | Cùng seed → cùng partition, cùng hoán vị vị trí, cùng $w^0$ cho mọi topology/thuật toán |
| G10 | So D-PSGD với DFedAvg theo round | So theo tổng số bước cục bộ và theo byte |
| G11 | Quên rằng Dirichlet kéo theo quantity skew | Luôn báo cáo kích thước node; dùng P3 để tách riêng ảnh hưởng |
| G12 | Đánh giá 20 model tốn CPU | Lịch đánh giá theo bước cục bộ (dày lúc đầu, thưa về sau) + tập con test cố định; đánh giá cuối đủ 10,000 ảnh, lưu ở trường riêng |
| G13 | Kỳ vọng tái lập tuyệt đối giữa các máy | Tài liệu PyTorch không đảm bảo trùng khớp giữa phiên bản, nền tảng, CPU/GPU → manifest ghi máy + phiên bản; chỉ so khớp trên cùng máy |
| G14 | `torch.load(..., weights_only=False)` | Giữ mặc định `weights_only=True` (PyTorch 2.14); YAML dùng `safe_load` |
| G15 | Diễn giải sai độ đo suy biến trên FC | Với MH trên FC, $W=\frac1N\mathbf 1\mathbf 1^\top$: sau tổng hợp mọi node trùng model nên std accuracy và consensus distance bằng 0 theo cấu trúc. Dùng thêm acc^loc và drift trước tổng hợp (T-INV-11) |
| G16 | Chọn μ của FedProx quá nhỏ | Đánh giá theo cường độ hiệu dụng $\eta\mu\tau$; μ = 0.01 ở η = 0.05, τ = 20 gần như không đổi gì. Dùng μ ∈ {0.1, 1} |
| G17 | Node nhỏ hơn một batch | min_size = 64 (= 2B); minibatch lấy có hoàn lại, không dùng `drop_last` |
| G18 | Gossip có "điều phối viên ẩn" hoặc lịch có trạng thái | Matching sinh từ lịch chung (seed chung + chỉ số round), mỗi node tự tính; `graph_at` là hàm thuần của (seed, round). Cận $\rho^t$ chỉ áp dụng cho topology tĩnh; gossip chỉ kiểm tra theo kỳ vọng |
| G19 | Nhầm "lệch mục tiêu" với drift | Với Dirichlet, DFL (W ngẫu nhiên kép) tối ưu $\frac1N\sum_i F_i$: báo cáo TV lệch mục tiêu, dùng đối chứng E1c. Không dùng $W_{ij}\propto n_j$ trên đồ thị không đầy đủ (không hiệu chỉnh đúng) |
| G20 | Dùng lại kết quả do code cũ sinh ra | `code_version` = hash nội dung `src/` + file khóa (không phải HEAD commit); `run_id` gồm `code_version`; sweep từ chối khi `src/`/`configs/`/file khóa có thay đổi chưa commit; không trộn nhiều `code_version` trong một hình |
| G21 | Rò rỉ tính bất định | Thời gian ghi ở `timing.jsonl`, không ở `rounds.jsonl`; Observer dùng model đánh giá cấp phát sẵn và không tiêu thụ RNG của huấn luyện; số luồng CPU cố định và ghi vào manifest |
| G22 | Snapshot dùng chung bộ nhớ với tham số | `state_dict()` trả về view chia sẻ bộ nhớ với tham số đang sống; mỗi node phải clone + detach đúng một snapshot mỗi round, và chỉ ghi kết quả tổng hợp sau khi mọi node đã tổng hợp xong |
| G23 | #C=k với N < 10 | Không thể để mọi nhãn có chủ; báo lỗi cấu hình. Profile smoke (N = 4) chỉ dùng IID/Dirichlet |
| G24 | Độ đo ổn định và acc^loc gây ngộ nhận | Tính độ dao động/mức sụt trên từng node rồi mới lấy trung bình (trung bình của 20 model tự nó mượt hơn 1 model); acc^loc thưởng thiên lệch cục bộ nên luôn báo cáo kèm acc toàn cục và acc^loc của B2 |
| G25 | Rò rỉ test khi chọn lr | Quét lr (E0a) chỉ đánh giá trên validation; nếu các η chênh ≤ 0.2 điểm % thì xét AUC validation |

### 2.5 Muốn biết X → đọc Y

| Cần biết | Đọc |
|---|---|
| Yêu cầu đề nguyên văn | H§2.1 (chỉ mở PDF khi có tranh chấp cách hiểu) |
| Vì sao chọn PyTorch + NetworkX, không Flower | §1.1, ADR-001 (§10) |
| Lỗi của bản nháp cần sửa | §1.2 (D1–D17) |
| Thuật toán chia dữ liệu và thang Non-IID | §3.2 |
| Topology, trọng số, spectral gap | §3.3 |
| Định nghĩa thuật toán D-PSGD/DFedAvg/DFedProx, baseline | §3.4 |
| Siêu tham số, seed, profile | §3.5 |
| Công thức độ đo | §3.6 |
| Lưới thí nghiệm và ngân sách | §3.7 |
| Biểu đồ cần vẽ | §3.8 |
| Kiến trúc, module, contract, bất biến, cấu trúc thư mục, định dạng kết quả | §4.1–§4.7 |
| Test bất biến và thẩm định khoa học | §5.2, §5.3 |
| Danh sách task gốc | §8 |

---

## 3. Quy trình prompt theo pha

### 3.0 Prompt khởi động phiên (dán đầu **mọi** phiên)

```text
Bạn đang làm dự án Đề 2 (Non-IID × Decentralized Federated Learning) trong repo này.
Trước khi làm bất cứ việc gì:
1. Đọc CLAUDE.md (nếu đã có), mục cuối của tasks/SESSION_LOG.md (nếu đã có), và chạy `git status` + `git log --oneline -10`.
2. Dùng skill using-agent-skills để chọn skill cho việc của phiên này; nói rõ sẽ dùng skill nào.
3. Chỉ nạp các mục tài liệu mà prompt bên dưới chỉ định. Không đọc lại PDF đề tài/bài báo; dùng context pack ở docs/HANDOFF_GUIDELINE.md §2.
4. Với việc không tầm thường: nêu khối ASSUMPTIONS trước. Gặp mâu thuẫn giữa spec, code và tài liệu thì DỪNG và hỏi, không tự chọn một cách hiểu.
5. Kết thúc phiên bằng prompt "Kết thúc phiên" (H§3.3, P3.9).
Việc của phiên này:
<dán prompt của bước>
```

### 3.1 PHA 1 — Discovery, Requirements & Design

| | 1A Discovery | 1B Requirements & Design |
|---|---|---|
| Mục tiêu | Hiểu đúng đề; chốt giả định và câu hỏi cho GVHD; dựng repo và rules file; sửa bản nháp | SPEC, ADR, contract, ngưỡng chất lượng; phản biện thiết kế |
| Tài liệu cần nạp | H§2 (context pack); §0, §1, §2.4–§2.5, §11 | H§2; §2–§5, §10 |
| Skills | `using-agent-skills`, `context-engineering`, `git-workflow-and-versioning`, `interview-me`, `idea-refine`, `docx` | `spec-driven-development`, `documentation-and-adrs`, `api-and-interface-design`, `source-driven-development`, `constraint-driven-development`, `doubt-driven-development` |
| Persona / subagent | `Explore` | `Plan`, `general-purpose` (reviewer ngữ cảnh mới) |
| Checklists | `orchestration-patterns.md` (pattern 5) | `definition-of-done.md`, `security-checklist.md` (phần secrets/phụ thuộc cho Boundaries) |
| Đầu ra | Repo git + CLAUDE.md + khung thư mục; `docs/decisions/OPEN_QUESTIONS.md`; `docs/đath1_revised.docx`; (tùy chọn) `docs/research/DIGEST.md` | `SPEC.md`, `docs/decisions/ADR-00N-*.md`, `src/dfl/contracts.py`, `CONSTRAINTS.md`, `Makefile` |
| Gate | **G0** | **G1** |

#### 1A — Discovery

**P1.1 — Dựng repo và rules file** · skills: `context-engineering`, `git-workflow-and-versioning`

```text
Đọc: docs/HANDOFF_GUIDELINE.md §5.1 (mẫu CLAUDE.md); docs/KE_HOACH_DE_TAI_2.md §4.6, §4.8.
Làm:
1. `git init -b main`. Tạo .gitignore cho: .venv/, data/, results/, build/, __pycache__/, .pytest_cache/, .coverage, htmlcov/, *.egg-info/.
2. Tạo CLAUDE.md từ mẫu H§5.1; giữ nguyên phần Boundaries và "Bất biến P2P".
3. Tạo khung thư mục theo §4.6 (chỉ thư mục, __init__.py rỗng, README khung). Tạo pyproject.toml ghim phiên bản §4.8, có extras [dev] và cấu hình ruff/pytest.
4. Tạo tasks/SESSION_LOG.md theo mẫu H§5.2.
KHÔNG viết logic.
Kiểm chứng (báo lại lệnh + kết quả):
- `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`
- `python -c "import torch, torchvision, networkx; print(torch.__version__, torchvision.__version__, networkx.__version__)"`
- `pytest -q` (chưa có test cũng phải chạy được) và `ruff check .` sạch.
Commit: "chore: bootstrap repo, rules file and project skeleton". Dừng và báo phiên bản thực tế đã cài.
```

**P1.2 — (Tùy chọn) Tóm tắt tài liệu nghiên cứu** · pattern Research isolation · subagent `Explore` · skill `source-driven-development`

```text
Dùng subagent Explore (chỉ đọc) đọc docs/research/*.pdf và các link ở docs/KE_HOACH_DE_TAI_2.md §12 mà bạn cần để kiểm tra §1.3.
Trả về DIGEST không quá 150 dòng. Với mỗi tài liệu: (a) thiết lập thực nghiệm (dataset, số node, topology, cách chia Non-IID, số seed);
(b) kết quả định lượng liên quan tới RQ1–RQ4; (c) điều dự án tái sử dụng. Mọi con số phải kèm trang/mục nguồn; không suy đoán.
Không chép nguyên văn dài vào phiên chính. Lưu docs/research/DIGEST.md và commit.
```

**P1.3 — Chốt giả định và câu hỏi mở** · skill `interview-me` (sau đó `idea-refine` nếu cần chọn phần mở rộng)

```text
Đọc: docs/KE_HOACH_DE_TAI_2.md §2.4, §2.5, §11.
Phỏng vấn tôi TỪNG CÂU MỘT, mỗi câu kèm phỏng đoán của bạn và một giá trị mặc định, để chốt A1–A6 và những câu trong Q1–Q7 mà nhóm tự trả lời được.
Gom các câu cần GVHD trả lời thành một tin nhắn/email ngắn bằng tiếng Việt để nhóm gửi.
Dừng khi bạn đạt khoảng 95% chắc chắn VÀ tôi xác nhận rõ ràng ("đúng", "chốt"). "Tùy bạn" hay "nghe ổn" không phải xác nhận.
Ghi kết quả vào docs/decisions/OPEN_QUESTIONS.md theo mẫu H§5.4 (câu hỏi · trả lời · người trả lời · ngày · ảnh hưởng tới ADR/task nào). Commit.
```

**P1.4 — Sửa bản nháp `đath1.docx`** · skill `docx`

```text
Đọc: docs/KE_HOACH_DE_TAI_2.md §1.2 (bảng D1–D17), §3.2–§3.6, §4.1–§4.3.
Tạo docs/đath1_revised.docx; KHÔNG sửa file gốc. Áp dụng D1–D17 dưới dạng tracked changes (tác giả "Claude"), làm các lỗi 🔴 trước.
Với các mục đang trống (Kiến trúc hệ thống, 5.1 Framework, B. Dataset), chèn nội dung ngắn gọn dựa trên §4.1–§4.3, §1.1, ADR-003.
Kiểm tra bằng scripts/office/validate.py của skill docx, render sang PDF và xem lại từng trang đã sửa. Liệt kê những chỗ cần nhóm tự quyết.
```

**Gate G0 — cuối 1A** (nhóm duyệt)
- [ ] Repo git có CLAUDE.md, khung thư mục, môi trường cài được (P1.1).
- [ ] `OPEN_QUESTIONS.md` có câu trả lời của nhóm; tin nhắn cho GVHD đã gửi (Q1 là quan trọng nhất).
- [ ] Bản nháp đã sửa ít nhất D1–D7.

#### 1B — Requirements & Design

**P1.5 — SPEC.md** · skill `spec-driven-development` · lệnh `/spec`

```text
Đọc: CLAUDE.md; docs/HANDOFF_GUIDELINE.md §2; docs/KE_HOACH_DE_TAI_2.md §2, §3, §4.4–§4.7, §5; docs/decisions/OPEN_QUESTIONS.md.
Viết SPEC.md ở gốc repo, gồm đủ 6 mục lõi: Objective, Commands, Project Structure, Code Style (kèm một đoạn code mẫu thật), Testing Strategy, Boundaries (Always / Ask first / Never); cộng thêm Tech Stack, Success Criteria, Open Questions.
Yêu cầu:
- Mở đầu bằng khối ASSUMPTIONS.
- Có mục "Capability map": module id ổn định (data, network, learning, sim, metrics-io, analysis), hướng phụ thuộc một chiều, thứ tự build. Dùng MỘT file SPEC.md, không tách spec cho từng module trừ khi tôi yêu cầu.
- Success Criteria phải kiểm chứng được: ánh xạ REQ-01…REQ-14 (H§2.1), test bất biến T-INV-1…12 (§5.2), ngưỡng sanity (§5.3).
- Boundaries lấy từ CLAUDE.md và bổ sung các quy tắc toàn vẹn nghiên cứu (§4.7, §5.3).
- Không viết code; không thêm tính năng ngoài bảng phạm vi §2.4.
Đầu ra: SPEC.md + danh sách điểm cần tôi quyết. Dừng chờ duyệt.
```

**P1.6 — ADR** · skill `documentation-and-adrs`

```text
Đọc: docs/KE_HOACH_DE_TAI_2.md §1.1, §10; SPEC.md; docs/decisions/OPEN_QUESTIONS.md.
Tách ADR-001…ADR-009 thành các file docs/decisions/ADR-00N-<slug>.md theo template của skill:
Status / Date / Context / Decision / Alternatives Considered / Consequences.
Status = Proposed cho tới khi qua Gate G1. Nếu câu trả lời của GVHD (ví dụ Q1) làm đổi một quyết định, ghi rõ và đề xuất phương án.
Không xóa ADR cũ; khi thay đổi thì tạo ADR mới và đánh dấu ADR cũ "Superseded by ADR-00M". Commit.
```

**P1.7 — Hợp đồng giao diện** · skills `api-and-interface-design`, `source-driven-development`

```text
Đọc: SPEC.md (Capability map); docs/KE_HOACH_DE_TAI_2.md §4.2, §4.4, §4.5, §4.7.
Viết src/dfl/contracts.py chỉ gồm Protocol / TypedDict / dataclass, không có logic, cho: Partitioner, TopologySchedule, MixingPolicy, Transport, LocalTrainer, Evaluator, RunLogger, cùng schema bản ghi rounds.jsonl và summary.json (§4.7).
Mọi API framework xuất hiện trong chữ ký hoặc ghi chú (torchvision.datasets.MNIST, torch.nn.Module.state_dict, networkx generators, numpy.random.Generator, torch.distributed nếu có) phải được đối chiếu với tài liệu chính thức đúng phiên bản trong pyproject.toml; ghi URL vào docstring. Cái gì không kiểm được thì ghi "UNVERIFIED".
Thêm tests/unit/test_contracts_smoke.py (import được, Protocol đúng tên). Commit riêng.
```

**P1.8 — CONSTRAINTS.md** · skill `constraint-driven-development` · lệnh `/constraints`

```text
Phát hiện trước (pyproject.toml, cấu hình pytest/ruff, CI nếu có), báo lại trong 2 dòng. Hỏi tôi tối đa 4 câu, mỗi câu có mặc định. Mặc định đề xuất cho dự án:
- Floor: test xanh; không skip hay xóa test; không thêm `# noqa` hay `type: ignore` mà không có lý do; không để stub chưa làm; không hạ ngưỡng.
- Ngưỡng có số: coverage dòng của src/dfl ≥ 80%; ruff sạch; smoke e2e dưới 60 s; T-INV-1…12 phải tồn tại và xanh.
- Toàn vẹn nghiên cứu: hình chỉ sinh từ results/ bằng script; không sửa tay results/; mọi lượt chạy có manifest + git hash; không xóa lượt lỗi; siêu tham số đã chốt chỉ được đổi qua ADR mới.
- Đặt kiểm tra theo chi phí bằng Makefile (dự án Python, không có package.json):
  `make check-fast` (ruff + unit, vài giây) · `make check-task` (unit + integration + coverage, dưới 90 s) · `make check-full` (tất cả + smoke e2e). Mẫu ở H§5.5.
Thêm vào CLAUDE.md dòng: "Đọc CONSTRAINTS.md; không bao giờ làm yếu nó để qua kiểm tra." Chạy thử các ràng buộc trên nhánh hiện tại. Commit.
```

**P1.9 — Phản biện thiết kế** · skill `doubt-driven-development` (chạy ở phiên chính)

```text
Áp dụng doubt-driven-development cho thiết kế:
- CLAIM (chỉ ghi cho bạn, KHÔNG đưa cho reviewer): SPEC + ADR đáp ứng REQ-01…14 và cho kết luận khoa học hợp lệ.
- EXTRACT: ARTIFACT = SPEC.md + docs/decisions/ADR-002, ADR-005, ADR-006, ADR-007, ADR-009 (các quyết định rủi ro nhất).
  CONTRACT = docs/HANDOFF_GUIDELINE.md §2.1 (REQ) + §2.4 (bẫy đã biết) + ràng buộc: nhóm sinh viên, chủ yếu CPU, 12 tuần.
- DOUBT: gọi một subagent general-purpose ngữ cảnh mới, dán NGUYÊN VĂN prompt đối kháng của skill ("Adversarial review. Find what is wrong with this artifact. … Do NOT validate. Do NOT summarize. …") rồi ARTIFACT + CONTRACT.
- RECONCILE: phân loại từng phát hiện theo thứ tự ưu tiên: contract misread / valid + actionable / valid trade-off / noise. Sửa artifact với loại actionable; ghi trade-off vào ADR liên quan.
- STOP: tối đa 3 vòng; còn vấn đề lớn sau 3 vòng thì báo tôi.
- Cross-model: máy chưa có gemini/codex CLI. Hỏi tôi có muốn review thủ công ở công cụ khác không; nếu bỏ qua, ghi rõ "Cross-model skipped".
```

**Gate G1 — cuối 1B** (nhóm + GVHD duyệt)
- [ ] SPEC.md đủ 6 mục; Success Criteria kiểm chứng được và ánh xạ REQ-01…14.
- [ ] ADR-001…009 chuyển từ Proposed sang Accepted (hoặc được thay bằng ADR mới).
- [ ] `contracts.py` import được; API đã đối chiếu tài liệu chính thức, có URL.
- [ ] `CONSTRAINTS.md` + Makefile chạy được trên nhánh hiện tại.
- [ ] Vòng phản biện P1.9 đã reconcile (≤ 3 vòng); trạng thái cross-model được ghi rõ.
- [ ] GVHD đồng ý phạm vi (§2.4) và câu trả lời Q1.

### 3.2 PHA 2 — Task Breakdown

| | |
|---|---|
| Mục tiêu | Biến SPEC thành các task nhỏ theo lát dọc, có tiêu chí chấp nhận, có checkpoint |
| Tài liệu cần nạp | SPEC.md, `docs/decisions/*.md`, CONSTRAINTS.md, §7–§9 |
| Skills | `planning-and-task-breakdown` (chính), `doubt-driven-development` (tùy chọn) |
| Persona / subagent | `Plan` (nếu cần so sánh cách cắt task) |
| Checklists | `definition-of-done.md` (mọi task đều phải qua), `testing-patterns.md` (lên kế hoạch test) |
| Đầu ra | `tasks/plan.md`, `tasks/todo.md` |
| Gate | **G2** |

**P2.1 — Phân rã task** · skill `planning-and-task-breakdown` · lệnh `/plan`

```text
Chế độ plan: chỉ đọc, KHÔNG viết code.
Đọc: SPEC.md, docs/decisions/*.md, CONSTRAINTS.md, docs/KE_HOACH_DE_TAI_2.md §7–§9. Bảng WBS T01–T25 ở §8 là hạt giống: được tách hoặc gộp, nhưng phải giải thích.
Yêu cầu:
- Nếu tasks/plan.md hoặc tasks/todo.md đã có task chưa xong cho việc khác: DỪNG và hỏi, không ghi đè.
- Vẽ đồ thị phụ thuộc; cắt lát dọc (xong mỗi task hệ thống vẫn chạy được); đưa task rủi ro cao lên sớm (T10 engine, T07 mixing).
- Mỗi task có: mô tả; tiêu chí chấp nhận (không quá 3 gạch đầu dòng); lệnh kiểm chứng (`make check-task` + test cụ thể); phụ thuộc; file dự kiến (≤ 5); cỡ S hoặc M. Không có task L/XL.
- Checkpoint C1–C5 như §8; cột người phụ trách theo §7; đánh dấu các task làm song song được.
- Mỗi test bất biến T-INV-x thuộc đúng một task.
Đầu ra: tasks/plan.md (quyết định kiến trúc, rủi ro, câu hỏi mở) + tasks/todo.md (checklist). Trình bày để tôi duyệt.
```

**P2.2 — (Tùy chọn) Phản biện kế hoạch** · skill `doubt-driven-development`

```text
ARTIFACT = tasks/todo.md; CONTRACT = SPEC.md (Success Criteria) + docs/KE_HOACH_DE_TAI_2.md §5.2.
Reviewer ngữ cảnh mới tìm: phụ thuộc bị thiếu, task không kiểm chứng được, T-INV chưa có chủ, task quá lớn, thứ tự làm tăng rủi ro. Tối đa 2 vòng.
```

**Gate G2 — cuối pha 2**
- [ ] Mọi task có tiêu chí chấp nhận, lệnh kiểm chứng, phụ thuộc; không task nào quá 5 file.
- [ ] Có checkpoint C1–C5; có người phụ trách; T-INV-1…12 đều có task chủ.
- [ ] Nhóm duyệt rõ ràng ("duyệt"/"go").

### 3.3 PHA 3 — Implementation & Testing

| | |
|---|---|
| Mục tiêu | Hiện thực từng task bằng TDD theo lát mỏng; chạy thí nghiệm E0–E4 có tiền đăng ký |
| Tài liệu cần nạp (theo từng task) | Task hiện tại trong `tasks/todo.md`; mục SPEC và § kế hoạch mà task tham chiếu; file sẽ sửa + test liên quan; một ví dụ pattern có sẵn trong repo |
| Skills | `incremental-implementation` + `test-driven-development` (qua `/build`); `source-driven-development`; `debugging-and-error-recovery`; `git-workflow-and-versioning`; `context-engineering`; `observability-and-instrumentation` (log thí nghiệm); `performance-optimization` (T13); `doubt-driven-development` (T10) |
| Persona / subagent | `test-engineer` (P3.3, P3.7); `code-reviewer` ở checkpoint (P3.8); `Explore` để tìm trong repo |
| Checklists | `testing-patterns.md` (chuyển sang pytest), `observability-checklist.md` (phần structured logging), `performance-checklist.md` (phần backend), `definition-of-done.md` |
| Đầu ra | Code + test + commit theo từng task; `results/`, `figures/`, `docs/experiments/E*.md`, `docs/PERF.md` |
| Checkpoint | C1 (sau T05) · C2 (sau T08) · C3 (sau T11) · C4 (sau T18, xong E0–E2) · C5 (sau T20) |

Quy tắc chung: dùng **`/build` (mỗi lần một task)**. Chỉ được dùng **`/build auto`** sau C3, cho các task rủi ro thấp (phân tích, vẽ hình); lệnh này đòi SPEC.md ở gốc repo, `git status` sạch và một lần duyệt plan rõ ràng.

**P3.0 — Làm một task** · skills `incremental-implementation` + `test-driven-development` · lệnh `/build`

```text
1. Chọn task chưa làm đầu tiên trong tasks/todo.md mà các phụ thuộc đã xong (hoặc task <ID> tôi chỉ định). Nhắc lại tiêu chí chấp nhận.
2. Nạp ngữ cảnh tối thiểu: file sẽ sửa, test liên quan, mục SPEC / § kế hoạch mà task tham chiếu, một ví dụ pattern có sẵn trong repo. Không nạp thêm.
3. RED: viết test thất bại trước (ưu tiên test bất biến T-INV liên quan). Chạy và cho tôi thấy nó FAIL đúng lý do.
4. GREEN: viết code tối thiểu để test qua. REFACTOR khi test vẫn xanh.
5. Dùng API framework mới: đối chiếu tài liệu chính thức (source-driven-development), ghi URL vào docstring/comment.
6. `make check-task` phải xanh; diff không quá khoảng 300 dòng và chỉ chạm các file của task.
7. Đánh dấu task xong trong tasks/todo.md và ghi tasks/SESSION_LOG.md theo mẫu H§5.2.
8. Commit một task = một commit, gồm code + test + cập nhật tasks/ (Conventional Commits: feat|test|fix|chore(<module>): …). Chỉ `git add` các file của task, không `git add -A`. Rồi DỪNG.
Nếu test không thể xanh, build hỏng, hoặc spec mơ hồ: dừng; dùng debugging-and-error-recovery (P3.4) hoặc hỏi tôi.
```

**P3.1 — Bổ sung cho T04 (Partitioner)**

```text
Làm theo P3.0, thêm:
- Test trước (tests/unit/test_partition.py), dùng mảng nhãn tổng hợp từ số đếm lớp MNIST (55k sau khi tách validation, như trong docs/spikes/spike_topology_partition.py) — không tải dữ liệu:
  * Bất biến: rời nhau; phủ đủ; mọi node ≥ min_size (64); tất định theo seed; #C=k đúng k nhãn/node và nhãn nào cũng có chủ.
  * Golden: sao chép các hàm niid_bench_dirichlet / niid_bench_label_k / niid_bench_quantity từ docs/spikes/spike_topology_partition.py vào tests/reference/niid_bench_reference.py, sửa để nhận mảng nhãn làm tham số (không dùng biến toàn cục LABELS). Với cùng numpy Generator, partitioner phải cho kết quả trùng TUYỆT ĐỐI với bản tham chiếu. Thứ tự tiêu thụ số ngẫu nhiên phải giữ đúng:
    - Dirichlet: với mỗi lớp 0…9, permutation chỉ số của lớp rồi mới dirichlet; bước "balance"; nếu tổng xác suất bằng 0 thì bỏ và làm lại toàn bộ; làm lại toàn bộ nếu có node < min_size.
    - #C=k: với từng node lần lượt, nhãn đầu là i mod 10 rồi rút integers(10) tới khi đủ k nhãn khác nhau; sau đó với từng lớp 0…9, permutation chỉ số của lớp và array_split cho các node sở hữu.
    - Quantity: mỗi lần thử rút dirichlet(β) trước rồi permutation toàn bộ chỉ số; rút lại cả hai nếu có node < min_size.
  * #C=k với N < 10 phải báo lỗi cấu hình rõ ràng.
  * Thống kê lỏng (không dùng dải min–max ở Phụ lục A làm ngưỡng): α = 1000 → JS < 0.01; JS trung bình qua 3 seed tăng đơn điệu khi α giảm trong {100, 1, 0.5, 0.1, 0.05}; α = 0.1 → JS trung bình 3 seed ∈ [0.45, 0.65]; quantity β = 0.5 → JS < 0.02 và CV(n) > 0.5.
- data/partition.py và data/heterogeneity.py chỉ dùng numpy, không import torch; heterogeneity tính cả "lệch mục tiêu" TV((1/N)·Σ_i P_i, P).
- Gán partition vào vị trí node bằng hoán vị ngẫu nhiên theo seed (§3.2); cùng seed thì cùng hoán vị cho mọi topology.
```

**P3.2 — Bổ sung cho T06–T08 (Topology, Mixing, Transport)**

```text
Làm theo P3.0, thêm các test viết trước:
- Topology: ring bậc 2 và liên thông; FC bậc N−1; ER luôn liên thông sau khi lấy mẫu lại, tất định theo seed;
  gossip trên K_N (T3): mỗi node có tối đa 1 đối tác mỗi round, N lẻ thì đúng 1 node đứng ngoài; trung bình thực nghiệm của W qua 10^4 matching khớp ½·I + (J − I)/(2(N−1)) khi N chẵn (λ₂ = ½ khi N lẻ) trong sai số lấy mẫu;
  gossip trên cạnh ER (T3b): chỉ dùng cạnh của đồ thị ER và là maximal matching (không còn cạnh nào có cả hai đầu đều chưa được ghép);
  graph_at là hàm thuần: gọi graph_at(r) hai lần hoặc gọi thêm graph_at(r') xen giữa đều cho cùng kết quả (dùng np.random.default_rng([seed, r])).
- Mixing: MH đối xứng, ngẫu nhiên kép, không âm, bằng 0 ngoài cạnh (trừ đường chéo); uniform chỉ ngẫu nhiên theo hàng; size-weighted chỉ cho FC và khớp n_j/n;
  spectral gap ring(20) = 0.033 ± 0.001, FC = 1; ma trận của gossip đối xứng và W @ W = W.
- Transport/CommLedger: ring N = 20 → 40 message/round; FC → 380; gossip trên K_N → 20; byte = message × 4 × numel;
  không có message tự gửi; node chỉ đọc được inbox của chính nó; payload là tham chiếu tới snapshot (không deep-copy) nhưng byte vẫn được đếm đủ.
```

**P3.3 — T10 Engine (task rủi ro cao nhất)** · skills `test-driven-development` + `doubt-driven-development` · persona `test-engineer`

```text
Bước 1 — thiết kế test (dùng subagent test-engineer): đề xuất test tích hợp cho T-INV-1…4, T-INV-11, T-INV-12 (docs/KE_HOACH_DE_TAI_2.md §5.2; T-INV-5 thuộc T11 vì cần RunLogger) với MLP tí hon + dữ liệu tổng hợp 4 node, mỗi test chạy dưới 5 s. Tôi duyệt danh sách.
Bước 2 — RED: viết các test đó:
  T-INV-1: lr = 0, khởi tạo KHÁC nhau. Topology tĩnh (ring, ER, FC): Ξ_t ≤ ρ^t · Ξ_0 + 1e-6 và trung bình model được bảo toàn (MH).
           Gossip: KHÔNG dùng cận một phía cho Ξ_t (dễ chập chờn). Kiểm tra từng W (đối xứng, ngẫu nhiên kép, W² = W) và E[W] như ở P3.2; nếu thêm Monte Carlo cho Ξ_t² thì hai phía, ±4 sai số chuẩn, t ≤ 3;
  T-INV-2: W = I → trùng khớp local-only;
  T-INV-3: FC + trọng số size-weighted (n_j/n) → allclose với một hàm FedAvg(n_k/n) tham chiếu viết riêng (atol = 1e-6); kích thước bằng nhau thì FC + MH cũng vậy;
  T-INV-4: đảo thứ tự duyệt node → kết quả trùng TỪNG BIT (tổng hợp cộng theo thứ tự id người gửi tăng dần);
  T-INV-11: FC + MH → sau tổng hợp mọi node trùng model (std của acc = 0) và acc^loc của mọi node được tính từ cùng một vector recall;
  T-INV-12: tắt Observer, hoặc gọi graph_at thêm lần → quỹ đạo huấn luyện không đổi.
Bước 3 — GREEN: engine ATC đúng §3.4 + §4.3 + INV-1…6; Observer đo drift trước tổng hợp và các độ đo sau tổng hợp theo lịch §3.5.
  Lưu ý: không cập nhật tại chỗ; mỗi node clone + detach đúng một snapshot w^(t+½) mỗi round (không dùng view của state_dict()), chỉ ghi kết quả tổng hợp sau khi mọi node tổng hợp xong; chỉ trộn tensor kiểu float (tham số + buffer float); buffer số nguyên giữ nguyên của chính node;
  SGD thuần không mang trạng thái qua round (nếu sau này có momentum thì giữ cục bộ, không trộn);
  minibatch lấy có hoàn lại bằng torch.Generator riêng của từng node; hệ số mất mát c_i (mặc định 1);
  Observer dùng một model đánh giá cấp phát sẵn (chỉ load_state_dict), không khởi tạo module mới và không đụng RNG của huấn luyện.
Bước 4 — doubt-driven: ARTIFACT = src/dfl/sim/engine.py + src/dfl/sim/node.py; CONTRACT = §3.4, §4.5 và H§2.4 (G2–G7, G15, G18, G21, G22). Không đưa lời khẳng định của bạn cho reviewer. Tối đa 3 vòng.
```

**P3.4 — Khi có lỗi** · skill `debugging-and-error-recovery`

```text
Dừng mọi việc khác (stop-the-line). Làm theo thứ tự:
Reproduce (lệnh tối thiểu tái hiện) → Localize (data/partition? mixing? engine? metric? hay chính test sai?) → Reduce (ca nhỏ nhất: N = 2–4, 1–2 round)
→ Fix nguyên nhân gốc → Guard (thêm test hồi quy, xác nhận nó FAIL trước khi sửa) → Verify (`make check-full`).
Báo cáo: nguyên nhân gốc trong 1 câu + test đã thêm. Không sửa ngưỡng hay test để "cho qua".
```

**P3.5 — T13 Đo hiệu năng** · skill `performance-optimization` · checklist `performance-checklist.md` (phần backend)

```text
MEASURE → IDENTIFY → FIX → VERIFY → GUARD.
Đo trên máy thật: thời gian huấn luyện/round, thời gian mỗi lần đánh giá, tổng thời gian một lượt `fast`, ước lượng lượt `full`; thử torch.set_num_threads ∈ {1, 2, 4, 8} kết hợp số worker chạy song song.
Chỉ tối ưu điểm nghẽn đo được (ví dụ: nạp sẵn tensor thay DataLoader, đánh giá theo lô lớn, chỉnh lịch đánh giá). Mỗi thay đổi ghi số trước/sau và quyết định giữ hay revert.
Ghi bảng vào docs/PERF.md và cập nhật ước tính ngân sách §3.7 trong tasks/plan.md. Không đổi ngữ nghĩa thí nghiệm để chạy nhanh hơn khi chưa có ADR.
```

**P3.6 — Chạy một nhóm thí nghiệm (E0 … E4)** · skills `observability-and-instrumentation`, `git-workflow-and-versioning` · checklist `observability-checklist.md` (phần structured logging)

```text
1. Tiền đăng ký: tạo docs/experiments/E<k>.md theo mẫu H§5.3 (giả thuyết, lưới, siêu tham số, lệnh) và ghi code_version hiện tại (hash nội dung src/ + file khóa; commit thẻ không làm đổi giá trị này). Commit thẻ TRƯỚC khi chạy.
   E0 chạy hai giai đoạn, mỗi giai đoạn một thẻ: E0a (quét lr trên validation) → chọn η theo quy tắc §3.5, ghi vào ADR-007 → E0b (sanity, baseline, độ nhạy).
2. Chạy: `python -m dfl sweep --experiment configs/experiments/e<k>_<tên>.yaml --workers <n>`. Sweep từ chối nếu src/, configs/ hoặc file khóa có thay đổi chưa commit; bỏ qua run_id đã có attempt `ok`; chạy lại được attempt lỗi (ví dụ mất kết nối Colab).
   Không sửa code giữa chừng. Nếu buộc phải sửa code trong src/ sau khi đã chạy một phần, code_version đổi: chạy lại TOÀN BỘ các nhóm cần so sánh với nhau ở code_version mới; lượt cũ thành "stale", không trộn vào hình.
3. Kiểm tra: mọi run_id của nhóm có ít nhất một attempt `ok`; attempt `failed` giữ nguyên và ghi nguyên nhân; E0b phải đạt ngưỡng sanity §5.3 trước khi chạy E1.
4. Sinh hình bằng script. Ghi tóm tắt kết quả và đối chiếu từng giả thuyết (ủng hộ / không ủng hộ / chưa kết luận, kể cả khi trái kỳ vọng) vào docs/experiments/E<k>.md.
5. Không sửa results/ bằng tay; không chạy lại để chọn seed "đẹp".
```

**P3.7 — Rà soát kiểm thử ở C3 và C5** · persona `test-engineer` · checklist `testing-patterns.md`

```text
Dùng subagent test-engineer: phân tích coverage và khoảng trống kiểm thử của src/dfl, ưu tiên engine, mixing, partition, metrics.
Đối chiếu với T-INV-1…12 (docs/KE_HOACH_DE_TAI_2.md §5.2) và CONSTRAINTS.md.
Báo cáo: test còn thiếu cho happy path, biên (N = 2, node đúng 64 mẫu, α rất nhỏ, N lẻ trong gossip, R@x không đạt, FC suy biến), và đường lỗi; đề xuất test Prove-It cho từng khoảng trống. Không sửa code.
```

Sau đó nhóm quyết định tạo task mới trong `tasks/todo.md` cho các khoảng trống.

**P3.8 — Checkpoint C1 … C5**

```text
Checkpoint C<n>:
1. Chạy `make check-full` và báo kết quả.
2. Liệt kê các task đã xong kể từ checkpoint trước, kèm commit.
3. Demo lệnh chạy được (ví dụ C3: `python -m dfl run --config configs/profiles/fast.yaml` cho ring và FC trên dữ liệu IID; C2: in bảng spectral gap sinh từ code, so với §3.3).
4. Dùng subagent code-reviewer review diff từ checkpoint trước (xem câu hỏi riêng ở P4.1).
5. Đối chiếu Definition of Done (H§4.2). Dừng chờ nhóm duyệt; ghi kết luận vào tasks/plan.md.
```

**P3.9 — Kết thúc phiên** · skill `context-engineering` (ranh giới phiên có thể khởi động lại)

```text
Ghi vào tasks/SESSION_LOG.md theo mẫu H§5.2: task đã xong / đang dở; file đã đổi; trạng thái git (đã commit chưa); lệnh kiểm chứng và kết quả;
quyết định mới (nếu cần ADR thì ghi rõ); câu hỏi mở; task kế tiếp. Không để thay đổi chưa commit mà không có ghi chú.
```

### 3.4 PHA 4 — Code Review & Thẩm định

| | |
|---|---|
| Mục tiêu | Review 5 trục; đơn giản hóa; thẩm định khoa học (tái lập, toàn vẹn, kiểm toán nhận định); phát hành bản nộp |
| Tài liệu cần nạp | Diff/PR cần review; SPEC.md; CONSTRAINTS.md; §5; `docs/experiments/E*.md`; bản thảo báo cáo |
| Skills | `code-review-and-quality`, `code-simplification`, `security-and-hardening` (phạm vi hẹp), `doubt-driven-development`, `shipping-and-launch` (điều chỉnh), `documentation-and-adrs`, `constraint-driven-development` (`/constraints guard`) |
| Persona | `code-reviewer`, `test-engineer`, `security-auditor` (fan-out `/ship`); `general-purpose` cho kiểm toán nhận định |
| Checklists | `definition-of-done.md` (cổng cuối), `security-checklist.md` (secrets, phụ thuộc, deserialization, thao tác đường dẫn), `orchestration-patterns.md` (pattern 3) |
| Đầu ra | Báo cáo review; `docs/VALIDATION.md`; README cuối; báo cáo đồ án; tag `v1.0` |
| Gate | **G3** |

**P4.1 — Review** · skill `code-review-and-quality` · lệnh `/review` · persona `code-reviewer`

```text
Phạm vi: diff từ checkpoint/tag trước (`git diff <tag>..HEAD`) hoặc PR của một task. Chạy `/constraints guard` trước để bắt ngưỡng bị hạ, test bị skip/xóa, suppression mới.
Review 5 trục, thêm các câu hỏi riêng của dự án:
- Correctness: có vi phạm INV-1…6 không? Tổng hợp có dùng snapshot? W có đúng tính chất? Byte chỉ được đếm ở Transport? Seed có kiểm soát hết nguồn ngẫu nhiên? Có dùng tập test để chọn siêu tham số không? Có đo drift trước tổng hợp và acc^loc không, hay đang diễn giải phương sai 0 trên FC như một kết quả? Độ dao động và mức sụt có tính theo từng node không? acc^loc có luôn đi kèm acc toàn cục không? run_id có gồm code_version, và có hình nào trộn nhiều code_version không? Có bẫy nào ở H§2.4 (G1–G21)?
- Readability: tên theo ký hiệu kế hoạch (tau, mu, W, rho) nhất quán; hàm ngắn, dễ theo dõi.
- Architecture: partition/topology/mixing không phụ thuộc torch; observer không phản hồi vào huấn luyện; ba thuật toán không bị lặp logic (ADR-006).
- Security: không `torch.load(weights_only=False)`; YAML dùng `safe_load`; không ghi hay xóa ngoài results/ và figures/.
- Performance: không có vòng lặp Python theo từng mẫu; không copy state_dict thừa trong vòng lặp round.
Phân loại: Critical / Important / Suggestion, kèm file:dòng và cách sửa. Thay đổi lớn hơn khoảng 300 dòng: đề nghị tách.
```

**P4.2 — Đơn giản hóa** · skill `code-simplification` · lệnh `/code-simplify`

```text
Chỉ áp dụng cho file đã review xong và đang có test xanh. Giữ nguyên hành vi (Chesterton's Fence): hiểu vì sao code tồn tại trước khi đổi.
Mỗi thay đổi nhỏ chạy lại test; đỏ thì revert. Không được đổi kết quả số của thí nghiệm: so khớp summary.json của một lượt smoke trước và sau khi đơn giản hóa.
```

**P4.3 — Thẩm định khoa học** · skills `doubt-driven-development`, `test-driven-development`

```text
1. Tái lập: clone sạch vào thư mục tạm → cài môi trường → `make check-full` → chạy lại 2 lượt chọn ngẫu nhiên từ E1/E2 → so sánh summary.json (cùng máy).
2. Hình: sinh lại toàn bộ figures/ vào thư mục tạm bằng một lệnh → so khớp với bản trong repo.
3. Kiểm toán nhận định: lập bảng "Nhận định → hình/bảng → run IDs → giả thuyết H#".
   Với mỗi nhận định chính, gọi một reviewer ngữ cảnh mới (general-purpose, prompt đối kháng của doubt-driven): ARTIFACT = bảng số + hình liên quan; CONTRACT = câu nhận định cần bác bỏ.
   Reviewer tìm giải thích khác: nhiễu seed, lr cố định, trần của MNIST, quantity skew lẫn trong Dirichlet, cách chọn tập con test, hoán vị vị trí node.
4. Ghi docs/VALIDATION.md: đạt / không đạt cho từng mục §5.3 và từng nhận định.
```

**P4.4 — Phát hành (fan-out)** · lệnh `/ship` · skill `shipping-and-launch` (điều chỉnh cho sản phẩm nghiên cứu) · pattern 3 của `orchestration-patterns.md`

```text
Gọi 3 subagent SONG SONG trong cùng một lượt:
- code-reviewer: review 5 trục trên toàn repo (dùng câu hỏi riêng ở P4.1);
- test-engineer: coverage và khoảng trống kiểm thử;
- security-auditor: phạm vi hẹp — phụ thuộc có CVE (`pip-audit`), deserialization, secrets, thao tác ghi/xóa đường dẫn. Nói rõ dự án không có auth/web/dữ liệu cá nhân.
Gộp báo cáo ở phiên chính → quyết định go/no-go.
Thay "deploy/rollback" bằng: tag `v1.0` trên commit đã thẩm định; đóng băng results/ (nén + checksum SHA-256); README có hướng dẫn tái lập; báo cáo cuối dẫn chiếu run ID. Không có giám sát production (không áp dụng).
```

**P4.5 — Tài liệu cuối** · skill `documentation-and-adrs`

```text
Cập nhật README (Quick start, Commands, Architecture kèm link ADR, cách tái lập kết quả); cập nhật trạng thái ADR (Accepted / Superseded).
Bảo đảm không còn TODO bỏ quên hay code bị comment-out. Báo cáo đồ án dẫn nguồn theo danh mục §12 và dẫn chiếu hình/bảng + run ID cho mọi nhận định.
```

**Gate G3 — cuối pha 4** (nhóm duyệt, sau đó nộp)
- [ ] Không còn phát hiện Critical; các phát hiện Important đã sửa hoặc có ghi chú lý do.
- [ ] `docs/VALIDATION.md` đạt mọi mục §5.3; nhận định trái giả thuyết được báo cáo trung thực.
- [ ] Tái lập được từ clone sạch; toàn bộ hình sinh lại bằng một lệnh.
- [ ] README, ADR, báo cáo cập nhật; tag `v1.0`; results/ đã đóng băng kèm checksum.

---

## 4. Gates và Definition of Done

### 4.1 Tóm tắt các gate

| Gate | Khi nào | Ai duyệt | Điều kiện chính |
|---|---|---|---|
| G0 | Hết 1A | Nhóm | Repo + CLAUDE.md; câu hỏi cho GVHD đã gửi; bản nháp sửa D1–D7 |
| G1 | Hết 1B | Nhóm + GVHD | SPEC, ADR, contracts, CONSTRAINTS; phản biện đã reconcile |
| G2 | Hết pha 2 | Nhóm | plan + todo đầy đủ, không task quá lớn |
| C1–C5 | Trong pha 3 | Nhóm | `make check-full` xanh; demo chạy được; review diff |
| G3 | Hết pha 4 | Nhóm | Review, thẩm định, tái lập, tài liệu, tag |

### 4.2 Definition of Done của dự án

Nền tảng là `references/definition-of-done.md` (dùng nguyên các nhóm Correctness, Quality, Integration, Documentation, Ship-readiness). Bổ sung cho dự án nghiên cứu này:

- [ ] Hành vi mới có test FAIL trước và PASS sau; các T-INV liên quan vẫn xanh.
- [ ] `make check-task` xanh (mỗi task); `make check-full` xanh (mỗi checkpoint).
- [ ] Số liệu và hình mới đều sinh từ log bằng script; mỗi lượt có manifest đầy đủ (cấu hình, seed, git hash, phiên bản, máy).
- [ ] Không đổi siêu tham số hay ngưỡng ngoài ADR/CONSTRAINTS.
- [ ] SPEC/ADR/README phản ánh đúng trạng thái hiện tại.
- [ ] "Ship-readiness" hiểu theo nghĩa dự án: bảo mật ở phạm vi hẹp (H§1.4); "observability" là log thí nghiệm có cấu trúc; "rollback" là `git revert` hoặc quay về tag.

---

## 5. Mẫu (templates)

### 5.1 Mẫu `CLAUDE.md` (dùng ở P1.1)

````markdown
# Project: Non-IID × Decentralized Federated Learning (Đề 2 — ĐATH HK261)

## Mục tiêu
Mô phỏng DFL không server để đo ảnh hưởng của Non-IID lên accuracy/F1, tốc độ hội tụ, chi phí giao tiếp, độ ổn định, phương sai giữa node; so sánh topology và thuật toán.
Chi tiết: docs/KE_HOACH_DE_TAI_2.md · SPEC.md · docs/decisions/ · CONSTRAINTS.md · tasks/todo.md

## Tech stack
Python 3.12 · torch 2.14.0 · torchvision 0.29.0 · networkx 3.x · numpy 2.x · pandas · matplotlib · pyyaml · pytest · ruff

## Commands
- Cài đặt: `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`
- Kiểm tra nhanh / theo task / đầy đủ: `make check-fast` · `make check-task` · `make check-full`
- Chạy một lượt: `python -m dfl run --config configs/profiles/smoke.yaml`
- Chạy lưới: `python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --workers 4`
- Vẽ hình: `python -m analysis.plots --results results/ --out figures/`

## Quy ước code
- Layout src/: `src/dfl/...`; ưu tiên hàm thuần + dataclass; type hint đầy đủ; docstring ngắn giải thích *vì sao*.
- Truyền `numpy.random.Generator` / `torch.Generator` tường minh; không dùng `np.random.*` toàn cục.
- `data/partition.py`, `data/heterogeneity.py`, `network/topology.py`, `network/mixing.py` chỉ phụ thuộc numpy/networkx (không import torch); `network/transport.py` coi payload là đối tượng mờ.
- Không BatchNorm. Tổng hợp model dùng snapshot, không cập nhật tại chỗ.
- Test: pytest, Arrange-Act-Assert, tên test mô tả hành vi; `tests/unit|integration|e2e`.

## Bất biến P2P (không được vi phạm) — chi tiết docs/KE_HOACH_DE_TAI_2.md §4.5
INV-1 node chỉ đọc model của mình + inbox · INV-2 chỉ Transport đếm byte · INV-3 tổng hợp dùng snapshot ·
INV-4 W không âm, tổng hàng = 1, khớp cạnh; đồ thị tĩnh liên thông, gossip liên thông theo hợp các round · INV-5 observer chỉ đọc · INV-6 cùng cấu hình + seed + code_version + máy + số luồng → cùng rounds.jsonl

## Boundaries
- Always: viết test trước; `make check-task` trước khi commit; một task = một commit; cập nhật tasks/todo.md và tasks/SESSION_LOG.md; dẫn URL tài liệu chính thức cho API framework; chỉ chạy sweep khi cây làm việc sạch.
- Ask first: thêm phụ thuộc; sửa SPEC/ADR/CONSTRAINTS; đổi siêu tham số đã chốt; di chuyển hoặc xóa bất cứ gì trong results/; chạy lưới lớn hơn 10 lượt.
- Never: sửa tay results/ hoặc figures/; xóa lượt chạy lỗi; trộn kết quả của nhiều code_version trong một hình; chọn siêu tham số trên tập test; hạ ngưỡng trong CONSTRAINTS.md để qua kiểm tra; skip hay xóa test đang fail khi chưa được duyệt; commit dữ liệu MNIST hoặc results/; `torch.load(..., weights_only=False)`.
````

### 5.2 Mẫu một mục trong `tasks/SESSION_LOG.md`

```markdown
## 2026-10-12 — Phiên #7 — T07 Mixing (TV2)
- Trạng thái: T07 ✅ (commit abc1234) · T08 ⏳ chưa bắt đầu
- File đã đổi: src/dfl/network/mixing.py, tests/unit/test_mixing.py
- Kiểm chứng: `make check-task` → 41 passed, coverage 86%
- Quyết định mới: dùng numpy.linalg.eigvalsh cho W đối xứng (đã ghi docstring, không cần ADR)
- Câu hỏi mở: có cần hỗ trợ đồ thị có hướng không? (hiện tại: không — ADR-005)
- Việc tiếp theo: T08 Transport + CommLedger
```

### 5.3 Mẫu thẻ tiền đăng ký thí nghiệm `docs/experiments/E<k>.md`

```markdown
# E1 — Ảnh hưởng của mức Non-IID (tiền đăng ký)
- Giả thuyết: H1 (docs/KE_HOACH_DE_TAI_2.md §2.3)
- Lưới: DFedAvg × {ring, FC} × {α=100, 1, 0.5, 0.1, 0.05, #C=2, #C=1} × seed {0, 1, 2, 3, 4} = 70 lượt (đối chứng E1c: FC size-weighted × 5 mức α × 3 seed = 15 lượt)
- Cố định: N=20, τ=20, τ·R=4,000 (R=200), B=32 lấy có hoàn lại, η=<giá trị chốt ở E0>, SmallCNN, Metropolis–Hastings, min_size=64, lịch đánh giá §3.5 (tập con test 2,000; đánh giá cuối 10,000)
- code_version (hash nội dung src/ + file khóa): <hash> · Lệnh: `python -m dfl sweep --experiment configs/experiments/e1_noniid.yaml --workers 4`
- Tiêu chí hợp lệ: cả 70 run_id có attempt `ok`, cùng một code_version; α=100 ≈ IID (chênh không quá ~1 điểm %)
- Kết quả (điền sau khi chạy): …
- Đối chiếu giả thuyết: ủng hộ / không ủng hộ / chưa kết luận — kèm hình và run ID
```

### 5.4 Mẫu `docs/decisions/OPEN_QUESTIONS.md`

```markdown
| ID | Câu hỏi | Trả lời | Người trả lời | Ngày | Ảnh hưởng |
|---|---|---|---|---|---|
| Q1 | Mô phỏng đơn tiến trình có được coi là "hệ thống P2P"? | … | GVHD | … | ADR-002, T22 |
```

### 5.5 Mẫu `Makefile` cho kiểm tra theo chi phí (dùng ở P1.8)

```make
# Loại cli.py khỏi coverage trong pyproject.toml:  [tool.coverage.run] omit = ["src/dfl/cli.py"]
.PHONY: check-fast check-task check-full
check-fast:   ## lint + format + unit test (vài giây)
	ruff check . && ruff format --check . && pytest -q tests/unit
check-task:   ## unit + integration + coverage (dưới 90 s)
	pytest -q tests/unit tests/integration --cov=src/dfl --cov-report=term-missing --cov-fail-under=80
check-full:   ## toàn bộ test + smoke end-to-end trên dữ liệu tổng hợp, ghi vào build/ (không đụng results/)
	ruff check . && ruff format --check . && pytest -q --cov=src/dfl --cov-fail-under=80 && \
	rm -rf build/smoke && python -m dfl run --config configs/profiles/smoke.yaml --out build/smoke
```

---

## 6. Ma trận truy vết (REQ → thiết kế → task → test → hình)

| REQ | Thiết kế | Task | Test | Hình/Bảng |
|---|---|---|---|---|
| REQ-01 P2P, không server | §4.1–§4.5 | T08, T10 | T-INV-2, T-INV-4, T-INV-12 | Sơ đồ §4.1 |
| REQ-02 Non-IID | §3.2 | T04, T05 | T-INV-8 | P7 |
| REQ-03 Accuracy / F1 | §3.6 | T09 | T-INV-10 | P1, P2, T1 |
| REQ-04 Hội tụ | §3.6 | T15 | unit $R@x$, AUC | P1, P3b |
| REQ-05 Chi phí giao tiếp | §3.3, §3.6 | T08, T15 | T-INV-6 | P6 |
| REQ-06 Ổn định | §3.6 | T10, T15 | T-INV-1, T-INV-11 | P5, T1 |
| REQ-07 Topology | §3.3 | T06, T07, T18 | T-INV-9 | P3, P8 |
| REQ-08 Thuật toán | §3.4 | T09, T10, T19 | T-INV-3, T-INV-7 | P9 |
| REQ-09 Round / volume / variance | §3.6 (acc và acc^loc), §4.7 | T11, T15 | e2e smoke, T-INV-11 | P1, P4, P6 |
| REQ-10 Biểu đồ | §3.8 | T16 | e2e vẽ hình | P1–P9 |
| REQ-11 Kiến trúc | §4.1–§4.3 | P1.5 | Gate G1 | Sơ đồ |
| REQ-12 Baseline IID, cố định tham số | §3.5, §3.7 | T11, T14 | sanity §5.3 | E0 |
| REQ-13 Khuyến nghị | §2.1 O4 | T24 | P4.3 | Bảng khuyến nghị |
| REQ-14 Dataset | ADR-003 | T03, T21 | unit T03 | E6 |

---

## 7. Dấu hiệu đang đi sai (red flags)

- Chạy `/build auto` khi chưa có SPEC.md, trước checkpoint C3, hoặc khi `git status` chưa sạch.
- Viết engine trước khi có test bất biến T-INV-1…4, T-INV-11, T-INV-12.
- Kết quả "accuracy tăng khi α giảm" → nghi ngay lỗi đảo chiều α (G1) trước khi tin.
- Kết luận "FC không có phương sai giữa node" chỉ từ accuracy sau tổng hợp (G15), hoặc "FedProx không có tác dụng" với μ quá nhỏ (G16).
- Sửa lr hoặc τ sau khi thấy kết quả để số "đẹp" hơn → phải có ADR mới và chạy lại cả nhóm thí nghiệm.
- So D-PSGD với DFedAvg theo round thay vì theo tổng số bước cục bộ và theo byte.
- Dùng tập test để chọn siêu tham số.
- Một persona gọi persona khác, hoặc tạo một "meta-orchestrator" (anti-pattern A và B trong `orchestration-patterns.md`).
- Nạp toàn bộ PDF hoặc cả bản kế hoạch vào mỗi phiên (context flooding); để một phiên kéo dài qua nhiều task.
- Một commit gộp nhiều task; diff lớn hơn khoảng 300 dòng mà không tách.
- Agent tự đánh dấu gate là "đã duyệt".
