import copy

import pytest
import yaml
from pydantic import ValidationError

from app.policy.config import (
    Issue,
    PolicyConfig,
    get_policy,
    get_policy_bundle,
    load_policy,
)
from tests.policy_helpers import policy_dict


def _invalid(mutate) -> str:
    raw = copy.deepcopy(policy_dict())
    mutate(raw)
    with pytest.raises(ValidationError) as exc:
        PolicyConfig.model_validate(raw)
    return str(exc.value)


def test_default_policy_loads_with_versioned_hash():
    config, version = get_policy_bundle()
    assert version.startswith(f"{config.version}+") and len(version.split("+")[1]) == 8
    assert set(config.remedies) == set(Issue)


@pytest.mark.parametrize("bad", [10000.5, "10000", True, 0, -1])
def test_money_must_be_a_positive_whole_number(bad):
    _invalid(lambda raw: raw["refunds"].__setitem__("per_order_cap_inr", bad))


def test_unknown_keys_are_rejected():
    assert "extra" in _invalid(lambda raw: raw["returns"].__setitem__("window_dayz", 30)).lower()
    _invalid(lambda raw: raw.__setitem__("surprise_section", {}))


def test_every_issue_needs_remedies():
    message = _invalid(lambda raw: raw["remedies"].pop("damaged"))
    assert "remedies missing for: ['damaged']" in message


def test_unknown_remedy_rejected():
    _invalid(lambda raw: raw["remedies"].__setitem__("damaged", ["store_credit"]))


def test_escalation_flags_must_all_be_present():
    _invalid(lambda raw: raw["escalate_when"].pop("goodwill_refund_requested"))


def test_policy_file_override(tmp_path, monkeypatch):
    raw = copy.deepcopy(policy_dict())
    raw["returns"]["window_days"] = 14
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_policy(path)[0].returns.window_days == 14

    monkeypatch.setenv("POLICY_FILE", str(path))
    get_policy_bundle.cache_clear()
    try:
        assert get_policy().returns.window_days == 14
    finally:
        monkeypatch.delenv("POLICY_FILE")
        get_policy_bundle.cache_clear()
    assert get_policy().returns.window_days == 30
