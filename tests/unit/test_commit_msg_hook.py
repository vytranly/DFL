"""Test cho scripts/git-hooks/commit-msg: hook giữ quy ước Conventional Commits bằng tiếng Anh
(https://www.conventionalcommits.org/en/v1.0.0/). Chạy hook như git chạy: một đối số là file thông điệp.
"""

import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "scripts" / "git-hooks" / "commit-msg"


def _run_hook(tmp_path: Path, message: str) -> subprocess.CompletedProcess[str]:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8")
    return subprocess.run(["bash", str(HOOK), str(msg_file)], capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    "message",
    [
        "feat(sim): add the gloo transport for one node process (T22a)\n",
        "docs: add SPEC.md draft 0.1 (P1.5)\n",
        "fix(sim)!: refuse an --out into results/\n\nBREAKING CHANGE: --out no longer accepts results/\n",
        "chore(deps): bump ruff\n\nGhi chú tiếng Việt trong thân vẫn được.\n",
        "# Please enter the commit message\n\ntest(learning): cover the init guard\n# comment line\n",
        "Merge branch 'vy' into binh\n",
        'Revert "feat(sim): add the idempotent sweep"\n',
        "fixup! feat(data): add NIID-Bench partitioners\n",
    ],
    ids=["scope", "no-scope", "breaking", "vi-body", "git-comments", "merge", "revert", "fixup"],
)
def test_hook_accepts_conventional_english_headers_and_git_generated_messages(
    tmp_path: Path, message: str
) -> None:
    result = _run_hook(tmp_path, message)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "message",
    [
        "add the gloo transport\n",
        "feature(sim): add the gloo transport\n",
        "Feat(sim): add the gloo transport\n",
        "feat(sim) add the gloo transport\n",
        "feat(sim):add the gloo transport\n",
        "feat(sim): \n",
        "feat(sim): thêm transport gloo cho một tiến trình node\n",
        "feat(sim): " + "x" * 90 + "\n",
        "# only comments\n\n",
    ],
    ids=[
        "no-type",
        "unknown-type",
        "uppercase-type",
        "no-colon",
        "no-space",
        "empty-subject",
        "vi-header",
        "too-long",
        "empty",
    ],
)
def test_hook_rejects_headers_outside_the_convention_and_explains_why(tmp_path: Path, message: str) -> None:
    result = _run_hook(tmp_path, message)

    assert result.returncode == 1
    assert "Conventional Commits" in result.stderr
