import shutil

import pytest

from app.agent.prompts import PROMPTS_DIR, load_prompts, render_system_prompt
from app.policy.config import get_policy


def test_v1_loads_and_renders_with_policy_values():
    prompts = load_prompts("v1")
    assert prompts.fingerprint.startswith("v1+")
    system = render_system_prompt(prompts, get_policy())
    assert "$" not in system
    assert "within 30 days of delivery" in system
    assert "5 to 7 business days" in system


def test_every_notice_renders_with_its_placeholders():
    prompts = load_prompts("v1")
    values = {"first_name": "A", "summary": "s", "amount": "₹1", "result": "{}",
              "message": "m", "read_back": "r", "reference": "TCK-000001"}  # fmt: skip
    for key in prompts.notices:
        assert "$" not in prompts.notice(key, **values)


def test_missing_variable_fails_loudly():
    with pytest.raises(KeyError):
        load_prompts("v1").turn_context.render(verification="x")


def test_unknown_version_fails():
    with pytest.raises(FileNotFoundError):
        load_prompts("v999")


def test_front_matter_version_must_match_file_name(tmp_path):
    shutil.copytree(PROMPTS_DIR, tmp_path / "prompts")
    for kind in ("system", "turn_context"):
        (tmp_path / "prompts" / kind / "v1.md").rename(tmp_path / "prompts" / kind / "v2.md")
    notices = tmp_path / "prompts" / "notices"
    (notices / "v1.yaml").rename(notices / "v2.yaml")
    with pytest.raises(ValueError, match="version"):
        load_prompts("v2", base=tmp_path / "prompts")


def test_fingerprint_changes_with_content(tmp_path):
    shutil.copytree(PROMPTS_DIR, tmp_path / "prompts")
    before = load_prompts("v1", base=tmp_path / "prompts").fingerprint
    path = tmp_path / "prompts" / "system" / "v1.md"
    path.write_text(path.read_text() + "\nOne more line.\n")
    assert load_prompts("v1", base=tmp_path / "prompts").fingerprint != before
