# Seed data: edge cases

The seed (`backend/app/db/seed.py`, `make seed` / `make reseed`) creates **50 customers** and **200 orders**. It is fully deterministic: `random.Random(20261029)` and a fixed clock.

## Ground rules

| | |
|---|---|
| Seed "now" (`ANCHOR`) | **2026-10-29 06:30 UTC** (12:00 IST). Every relative date below is measured from this. The policy engine and evals must use `ANCHOR` as their clock when reasoning about seed data, or the date-based cases drift. |
| Return window | `RETURN_WINDOW_DAYS = 30`, from the shipment's `delivered_at` (`app/domain/policy.py`) |
| Auto-refund cap | `AUTO_REFUND_CAP_PAISE = ₹10,000` |
| Shipping fee | ₹40, or free when the subtotal is ₹499 or more |
| Money | Integer paise (`bigint`), e.g. ₹1,299.00 = `129900` |
| Identity | Customers 1–12 are the personas below, in this order. Customers 13–50 and their orders are generic random data. |

Phones and PIN codes are generated from the seed and are stable across runs. The phone numbers are random, valid-format Indian mobiles; **never send real SMS or OTP to them.**

## Cases

| # | Case | Customer (email · phone · PIN) | Orders | State | What it exercises |
|---|---|---|---|---|---|
| 1 | **Partial shipment, item never shipped** | Aarav Mehta · aarav.mehta@example.com · +919242223202 · 500471 | **VX-2201** `partially_shipped` | Keyboard and mouse delivered 3 days ago in one shipment. The LED desk lamp has `shipment_id = NULL` (backordered). | Explaining what has and hasn't shipped, without calling the order "delivered" or "missing". |
| 2 | **Split shipments, one delivered, one in transit** | Aarav Mehta | **VX-2202** `shipped` | Smartwatch delivered 2 days ago. The charger is in a second shipment, in transit, ETA 2 days from now. | Per-shipment tracking. Status is `shipped` because every item has shipped but not all are delivered. Aarav also has VX-2140 (delivered 40 days ago). |
| 3 | **Multiple active orders** | Kavya Nair · kavya.nair@example.com · +918685933716 · 700200 | **VX-3301** `placed` (5 h ago), **VX-3302** `shipped` (in transit), **VX-3303** `shipped` (out for delivery today), plus VX-3190 (delivered 20 days ago) | Three orders in flight at once. | "Where's my order?" needs a disambiguating question, not a guess. `list_active_orders_for_customer` returns newest first. |
| 4 | **Delivered but reported missing** | Sameer Khan · sameer.khan@example.com · +917145537820 · 302893 | **VX-3107** `delivered` | Carrier scan: "Delivered, left at front door", 2 days ago. One open, high-priority `missing_delivery` ticket is linked to a verified, escalated voice conversation (6 turns). | Carrier data contradicts the customer. The open ticket must not be duplicated. |
| 5 | **Return window expired by 1 day** | Neha Gupta · neha.gupta@example.com · +916524325150 · 226965 | **VX-4410** `delivered` (jeans) | Delivered exactly 31 days before `ANCHOR`; the window closed 1 day ago. | A return or refund must be refused, or escalated per policy, even if the customer pushes. |
| 5b | *Companion:* 1 day left | Neha Gupta | **VX-4411** `delivered` (2 T-shirts) | Delivered 29 days before `ANCHOR`; the window closes tomorrow. | The same customer has one eligible and one ineligible order, so the agent can't apply one answer to both. Exactly 30 days is deliberately avoided because inclusive vs exclusive is a policy-engine decision. |
| 6 | **Already refunded (fully)** | Vikram Singh · vikram.singh@example.com · +917861942444 · 500717 | **VX-5120** `delivered` (Bluetooth speaker) | Return `completed` (reason `damaged`), and a refund `succeeded` for the full order total. | A second refund must be refused, including on retry with a new idempotency key. |
| 6b | *Extra:* partially refunded | Vikram Singh | **VX-5121** `delivered` (kettle and frying pan) | Kettle returned and refunded (₹1,299). The frying pan is not refunded. | Refundable balance = total − succeeded refunds. Only the frying pan remains eligible. |
| 7 | **Cancelled order, refunded** | Ananya Rao · ananya.rao@example.com · +918182110280 · 400697 | **VX-6031** `cancelled` (silk saree) | Cancelled 2 h after placing. The full refund `succeeded` the next day. | Can't be returned, reshipped or refunded again. |
| 7b | *Extra:* cancelled, refund pending | Ananya Rao | **VX-6032** `cancelled` (power bank) | Cancelled 2 days ago. The refund is still `pending`. | "Where's my money?" Explain the pending status. A pending refund still counts against the refundable balance. |
| 8 | **Sound-alike IDs, same customer** | Rahul Verma · rahul.verma@example.com · +917912893751 · 226498 | **VX-1508** (power bank, delivered 8 days ago) and **VX-1580** (2 water bottles, delivered 6 days ago) | Both belong to Rahul. "Fifteen oh-eight" vs "fifteen eighty". | Read back and confirm the ID. Use the item to disambiguate. |
| 9 | **Sound-alike IDs, different customers** | Rohan Iyer · rohan.iyer@example.com · +917837184901 · 600504 | **VX-1315** (silk saree, delivered 7 days ago) | Its sound-alike **VX-1350** ("thirteen fifteen" vs "thirteen fifty") belongs to customer 9 below. | Rohan saying "thirteen fifty" must get "not found on your account". `get_order_for_customer` returns `None`, never Priya's data. |
| 9b | (owner of VX-1350) | Priya Sharma · priya.sharma@example.com · +916809773519 · 560105 | **VX-1350** `shipped` (smartwatch, in transit, ETA tomorrow) | | |
| 10 | *Extra:* **duplicate full name** | Priya Sharma · priya.sharma2@example.com · +919693368194 · 380705 | VX-9012 `delivered` (3 paperbacks) | A second, unrelated customer named Priya Sharma. | Name alone can never verify identity. `find_customers_by_name` returns both. |
| 11 | **High-value order above the refund cap** | Arjun Malhotra · arjun.malhotra@example.com · +918591207859 · 700123 | **VX-7302** `delivered` (ProBook 16 laptop, **₹84,999**) | Delivered 5 days ago, inside the return window. No returns, refunds or tickets yet. | Eligible by date but above `AUTO_REFUND_CAP_PAISE`. Must escalate to a human, not auto-refund. |
| 12 | *Extra:* **stuck shipment** | Meera Pillai · meera.pillai@example.com · +917973188087 · 302647 | **VX-8015** `shipped` (Bluetooth speaker) | ETA was 4 days ago. The latest scan is `in_transit`, 8 days ago. | Missing-delivery workflow for a parcel that is overdue but not marked delivered. |
| 13 | *Extra:* **unverified caller** | none (`conversations.customer_id IS NULL`) | (asks about VX-1350) | An abandoned voice conversation. The caller claims to be Priya Sharma, can't verify, and asks for the delivery address. | Reference transcript of a correct refusal. |

## Guarantees checked by tests

`backend/tests/test_seed_data.py` checks the following against the seeded test database:
- **Sound-alike pairs:** the *only* pairs in all 200 orders are VX-1315/VX-1350 and VX-1508/VX-1580. Generic order numbers are chosen to avoid creating new ones, as defined by `app/domain/order_ids.spoken_confusables`.
- **Money:** every `*_paise` column is `bigint`, `subtotal = Σ items`, and `total = subtotal + shipping`.
- **Refunds:** succeeded plus pending refunds never exceed the order total.
- **Order status matches shipments:**
  - `placed` and `cancelled` orders have no shipments.
  - `partially_shipped` orders have some unshipped items.
  - `shipped` orders have every item shipped, with at least one not yet delivered.
  - `delivered` orders have every item in a delivered shipment.
- **Shipment events** are chronological and never after `ANCHOR`.
- **Date-based cases:** VX-4410 and VX-4411 sit exactly 1 day either side of the window boundary. VX-7302 is above the cap and inside the window.

## Generic data (customers 13–50)

- **Orders:** placed uniformly over the last 120 days, with 1–3 lines from a 20-item catalog.
- **Shipping:** transit takes 3–6 days. Status follows from timing: `placed` if not yet shipped, `shipped` if in transit, otherwise `delivered`.
- **Cancellations:** about 5% of orders, each with a succeeded refund.
- **Returns:** about 6% of delivered orders have a completed return of the first item, with a succeeded refund.
- **Tickets:** a few delivered orders have a resolved `order_status` ticket.

Current mix across all 200 orders:

| Status | Orders |
|---|---|
| `delivered` | 177 |
| `cancelled` | 11 |
| `shipped` | 10 |
| `partially_shipped` | 1 |
| `placed` | 1 |

Totals: 16 returns, 27 refunds, 7 tickets and 2 conversations.
