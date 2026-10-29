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


def test_v3_policy_section_follows_the_mode():
    prompts = load_prompts("v3")
    assert set(prompts.policy_context) == {"retrieval", "full_text"}
    bare = render_system_prompt(prompts, get_policy())
    assert "$" not in bare and "search_policies" not in bare
    assert "within 30 days" not in bare  # v3 has no fixed policy summary
    retrieval = render_system_prompt(prompts, get_policy(), retrieval=True)
    assert "call search_policies" in retrieval
    full = render_system_prompt(prompts, get_policy(), policy_documents="DOCS")
    assert "<policy_documents>\nDOCS\n</policy_documents>" in full


def test_policy_modes_need_a_prompt_with_a_policy_section():
    with pytest.raises(ValueError, match="policy_context"):
        render_system_prompt(load_prompts("v2"), get_policy(), retrieval=True)


def test_policy_context_is_part_of_the_fingerprint(tmp_path):
    shutil.copytree(PROMPTS_DIR, tmp_path / "prompts")
    before = load_prompts("v3", base=tmp_path / "prompts").fingerprint
    path = tmp_path / "prompts" / "policy_context" / "v3.yaml"
    path.write_text(path.read_text() + "\n# changed\n")
    assert load_prompts("v3", base=tmp_path / "prompts").fingerprint != before
