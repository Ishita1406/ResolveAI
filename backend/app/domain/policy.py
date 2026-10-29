"""Policy values shared by the seed data and, later, the policy engine.

Values only. Enforcement belongs to the deterministic policy engine (Milestone 1).
"""

RETURN_WINDOW_DAYS = 30
AUTO_REFUND_CAP_PAISE = 10_000_00  # ₹10,000
FREE_SHIPPING_THRESHOLD_PAISE = 499_00
SHIPPING_FEE_PAISE = 40_00
