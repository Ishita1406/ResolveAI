import pytest

from app.domain.order_ids import match_order_numbers, parse_spoken_order_number


@pytest.mark.parametrize(
    ("heard", "digits"),
    [
        ("VX-1508", "1508"),
        ("vx 1508", "1508"),
        ("fifteen oh eight", "1508"),
        ("fifteen eighty", "1580"),
        ("one five oh eight", "1508"),
        ("thirteen fifty", "1350"),
        ("um, order number vx fifteen eighty please", "1580"),
        ("double five one three", "5513"),
        ("seventy three oh two", "7302"),
        ("hello there", None),
    ],
)
def test_parse_spoken_order_number(heard, digits):
    assert parse_spoken_order_number(heard) == digits


def _reasons(heard, candidates):
    return [(m.order_number, m.reason) for m in match_order_numbers(heard, candidates)]


def test_exact_ranks_above_sound_alike():
    assert _reasons("fifteen eighty", ["VX-1508", "VX-1580"]) == [
        ("VX-1580", "exact"),
        ("VX-1508", "sounds_alike"),
    ]


def test_sound_alike_only_within_given_candidates():
    # Rohan owns VX-1315; VX-1350 (someone else's) is never a candidate.
    assert _reasons("thirteen fifty", ["VX-1315"]) == [("VX-1315", "sounds_alike")]


def test_transposed_and_off_by_one_and_partial():
    assert _reasons("1058", ["VX-1508"]) == [("VX-1508", "transposed")]
    assert _reasons("1509", ["VX-1508"]) == [("VX-1508", "one_digit_off")]
    assert _reasons("150", ["VX-1508", "VX-9150"]) == [
        ("VX-1508", "one_digit_off"),
        ("VX-9150", "one_digit_off"),
    ]
    assert _reasons("302", ["VX-7302"]) == [("VX-7302", "one_digit_off")]


def test_no_plausible_match():
    assert match_order_numbers("4242", ["VX-1508", "VX-1580"]) == []
    assert match_order_numbers("no digits", ["VX-1508"]) == []
