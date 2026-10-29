import pytest

from app.agent.grounding import CustomerTagGuard, ungrounded_identity


def _run(chunks):
    guard = CustomerTagGuard()
    shown = "".join(guard.feed(c) for c in chunks)
    return shown + guard.flush(), guard.tripped


def test_plain_text_passes_through_even_with_angle_brackets():
    assert _run(["Your total is < ₹500 ", "and <b>fine</b>."]) == (
        "Your total is < ₹500 and <b>fine</b>.",
        False,
    )


@pytest.mark.parametrize(
    "chunks",
    [
        ["Thanks.", "<customer_message>560001</customer_message>"],
        ["Thanks.<cust", "omer_mes", "sage>560001"],
        ["Thanks.<CUSTOMER_MESSAGE>x"],
        ["Thanks.</customer_message>"],
    ],
)
def test_customer_tag_is_cut_even_when_split_across_chunks(chunks):
    assert _run(chunks) == ("Thanks.", True)


def test_held_prefix_is_released_when_it_turns_out_harmless():
    guard = CustomerTagGuard()
    assert guard.feed("a <cust") == "a "
    assert guard.feed("ard") == "<custard"


def test_after_tripping_nothing_more_is_shown():
    guard = CustomerTagGuard()
    guard.feed("<customer_message>")
    assert guard.feed("more") == "" and guard.flush() == ""


TEXTS = ["where is my order?", "it's meera.pillai@example.com", "PIN is 302 647"]


@pytest.mark.parametrize(
    ("args", "missing"),
    [
        ({"email": "meera.pillai@example.com", "postal_code": "302647"}, None),
        ({"email": "Meera.Pillai@example.com", "postal_code": "302647"}, None),
        ({"email": "meera.pillai@example.com", "postal_code": "560001"}, "PIN code"),
        ({"email": "someone@example.com", "postal_code": "302647"}, "email address"),
        ({"phone": "+919876543210", "postal_code": "302647"}, "phone number"),
    ],
)
def test_identity_details_must_come_from_the_customer(args, missing):
    assert ungrounded_identity(args, TEXTS) == missing


def test_spoken_forms_count():
    texts = ["meera dot pillai at example dot com", "three zero two six four seven",
             "my number is 98765 43210"]  # fmt: skip
    args = {"email": "meera.pillai@example.com", "postal_code": "302647",
            "phone": "+91 98765 43210"}  # fmt: skip
    assert ungrounded_identity(args, texts) is None


def test_nothing_said_means_nothing_grounded():
    assert ungrounded_identity({"postal_code": "302647"}, []) == "PIN code"
