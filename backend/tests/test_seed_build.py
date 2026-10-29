from app.db.seed import TOTAL_CUSTOMERS, TOTAL_ORDERS, build_dataset


def _fingerprint(data):
    return [
        (
            c.email,
            c.phone,
            [(o.order_number, o.status, o.total_paise, o.placed_at) for o in c.orders],
            [s.tracking_number for o in c.orders for s in o.shipments],
        )
        for c in data.customers
    ]


def test_build_is_deterministic():
    assert _fingerprint(build_dataset()) == _fingerprint(build_dataset())


def test_seed_value_changes_output():
    assert _fingerprint(build_dataset(seed=1)) != _fingerprint(build_dataset(seed=2))


def test_target_sizes():
    data = build_dataset()
    assert len(data.customers) == TOTAL_CUSTOMERS
    assert len(data.orders) == TOTAL_ORDERS
