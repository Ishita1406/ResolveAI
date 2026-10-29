"""Versioned prompts loaded from backend/prompts/<name>/<version>.(md|yaml).

Templates use string.Template ($name). Rendering fails on any missing variable, so a prompt
can't silently ship with a hole in it. The fingerprint (version + content hash) is logged
with each turn so transcripts can be tied to the exact prompt text.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

from app.policy.config import PolicyConfig

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    body: str

    def render(self, **values: object) -> str:
        return Template(self.body).substitute({k: str(v) for k, v in values.items()})


@dataclass(frozen=True)
class PromptSet:
    version: str
    system: Prompt
    turn_context: Prompt
    notices: dict[str, str]
    fingerprint: str
    # policy_context/<version>.yaml (v3+): how policy text reaches the model, per mode.
    policy_context: dict[str, str] | None = None

    def notice(self, key: str, **values: object) -> str:
        return Template(self.notices[key]).substitute({k: str(v) for k, v in values.items()})


def _markdown_prompt(path: Path, expected_version: str) -> Prompt:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(), re.DOTALL)
    if not match:
        raise ValueError(f"{path} has no front matter")
    meta = yaml.safe_load(match.group(1))
    if meta.get("version") != expected_version:
        raise ValueError(f"{path}: front matter version {meta.get('version')!r} != file name")
    return Prompt(name=meta["name"], version=meta["version"], body=match.group(2).strip())


def load_prompts(version: str, base: Path = PROMPTS_DIR) -> PromptSet:
    system_path = base / "system" / f"{version}.md"
    context_path = base / "turn_context" / f"{version}.md"
    notices_path = base / "notices" / f"{version}.yaml"
    for path in (system_path, context_path, notices_path):
        if not path.exists():
            raise FileNotFoundError(f"prompt file missing: {path}")
    notices = _yaml_texts(notices_path, version)
    policy_path = base / "policy_context" / f"{version}.yaml"
    policy_context = _yaml_texts(policy_path, version) if policy_path.exists() else None
    digest = hashlib.sha256()
    for path in (system_path, context_path, notices_path, policy_path):
        if path.exists():
            digest.update(path.read_bytes())
    return PromptSet(
        version=version,
        system=_markdown_prompt(system_path, version),
        turn_context=_markdown_prompt(context_path, version),
        notices=notices,
        fingerprint=f"{version}+{digest.hexdigest()[:8]}",
        policy_context=policy_context,
    )


def _yaml_texts(path: Path, version: str) -> dict[str, str]:
    data = yaml.safe_load(path.read_text())
    if data.get("version") != version:
        raise ValueError(f"{path}: version {data.get('version')!r} != {version!r}")
    return data["texts"]


def render_system_prompt(
    prompts: PromptSet,
    policy: PolicyConfig,
    *,
    retrieval: bool = False,
    policy_documents: str | None = None,
) -> str:
    """The system prompt with policy values filled in from policy.yaml (never hard-coded).

    From v3 the policy section comes from policy_context: retrieval mode (the agent calls
    search_policies) or full-text mode (`policy_documents` inlined). Asking for either with
    a prompt version that has no such section fails rather than silently dropping it.
    """
    context = ""
    if retrieval or policy_documents is not None:
        if prompts.policy_context is None:
            raise ValueError(
                f"prompt {prompts.version} has no policy_context; use v3 or later for "
                "policy retrieval or full-text policy"
            )
        if retrieval:
            context = prompts.policy_context["retrieval"]
        else:
            context = Template(prompts.policy_context["full_text"]).substitute(
                documents=policy_documents
            )
    return prompts.system.render(
        return_window_days=policy.returns.window_days,
        refund_timeline=policy.refunds.processing_time,
        missing_after_days=policy.missing_delivery.grace_days_after_eta,
        policy_context=context,
    )
