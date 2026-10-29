import pytest

from app.domain.order_ids import spoken_confusables


@pytest.mark.parametrize(
    ("number", "expected_member"),
    [
        ("VX-1508", "VX-1580"),
        ("VX-1580", "VX-1508"),
        ("VX-1315", "VX-1350"),
        ("VX-1350", "VX-1315"),
    ],
)
def test_known_confusable_pairs_are_symmetric(number, expected_member):
    assert expected_member in spoken_confusables(number)


def test_unrelated_numbers_are_not_confusable():
    assert "VX-2201" not in spoken_confusables("VX-7302")


def test_rejects_malformed_numbers():
    with pytest.raises(ValueError, match="not an order number"):
        spoken_confusables("1508")
