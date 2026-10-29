"""Typed loader for policy.yaml. Invalid or incomplete policy fails loudly at load time."""

import hashlib
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_POLICY_FILE = Path(__file__).with_name("policy.yaml")
ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class Issue(StrEnum):
    MISSING_DELIVERY = "missing_delivery"
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    NO_LONGER_NEEDED = "no_longer_needed"


class Remedy(StrEnum):
    RESHIP = "reship"
    REFUND = "refund"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Returns(_Section):
    window_days: PositiveInt


class Refunds(_Section):
    per_order_cap_inr: PositiveInt
    per_customer_cap_inr: PositiveInt
    per_customer_window_days: PositiveInt
    processing_business_days_min: PositiveInt
    processing_business_days_max: PositiveInt

    @model_validator(mode="after")
    def _processing_range(self) -> Self:
        if self.processing_business_days_min > self.processing_business_days_max:
            raise ValueError("processing_business_days_min is above the max")
        return self

    @property
    def processing_time(self) -> str:
        return (
            f"{self.processing_business_days_min} to "
            f"{self.processing_business_days_max} business days"
        )

    @property
    def per_order_cap_paise(self) -> int:
        return self.per_order_cap_inr * 100

    @property
    def per_customer_cap_paise(self) -> int:
        return self.per_customer_cap_inr * 100


class Reship(_Section):
    max_value_inr: PositiveInt

    @property
    def max_value_paise(self) -> int:
        return self.max_value_inr * 100


class Shipping(_Section):
    free_shipping_threshold_inr: NonNegativeInt
    standard_fee_inr: NonNegativeInt

    @property
    def free_shipping_threshold_paise(self) -> int:
        return self.free_shipping_threshold_inr * 100

    @property
    def standard_fee_paise(self) -> int:
        return self.standard_fee_inr * 100


class MissingDelivery(_Section):
    grace_days_after_eta: NonNegativeInt


class EscalateWhen(_Section):
    carrier_marked_delivered: bool
    returned_to_sender: bool
    goodwill_refund_requested: bool
    over_order_refund_cap: bool
    over_customer_refund_cap: bool
    over_reship_value_cap: bool


class Verification(_Section):
    max_attempts: PositiveInt


class Confirmation(_Section):
    expire_after_turns: PositiveInt
    expire_after_seconds: PositiveInt


class PolicyConfig(_Section):
    version: str
    returns: Returns
    refunds: Refunds
    reship: Reship
    shipping: Shipping
    missing_delivery: MissingDelivery
    remedies: dict[Issue, tuple[Remedy, ...]]
    escalate_when: EscalateWhen
    verification: Verification
    confirmation: Confirmation

    @model_validator(mode="after")
    def _every_issue_has_remedies(self) -> Self:
        missing = set(Issue) - set(self.remedies)
        if missing:
            raise ValueError(f"remedies missing for: {sorted(i.value for i in missing)}")
        return self

    def allows(self, issue: Issue, remedy: Remedy) -> bool:
        return remedy in self.remedies[issue]


def load_policy(path: Path) -> tuple[PolicyConfig, str]:
    """Parse and validate; returns the config and a short content hash for audit logs."""
    raw = path.read_bytes()
    config = PolicyConfig.model_validate(yaml.safe_load(raw))
    return config, f"{config.version}+{hashlib.sha256(raw).hexdigest()[:8]}"


class PolicySettings(BaseSettings):
    # Separate from app Settings so loading policy never needs database credentials.
    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")
    policy_file: Path = DEFAULT_POLICY_FILE


@lru_cache
def get_policy_bundle() -> tuple[PolicyConfig, str]:
    return load_policy(PolicySettings().policy_file)


def get_policy() -> PolicyConfig:
    return get_policy_bundle()[0]
