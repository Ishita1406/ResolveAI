---
policy: refunds
title: Refunds policy
source: backend/app/policy/policy.yaml
values:
  refunds.per_order_cap_inr: 10000
  refunds.per_customer_cap_inr: 15000
  refunds.per_customer_window_days: 30
  refunds.processing_business_days_min: 5
  refunds.processing_business_days_max: 7
  escalate_when.over_order_refund_cap: true
  escalate_when.over_customer_refund_cap: true
  escalate_when.goodwill_refund_requested: true
---

# Refunds policy

## When can I get a refund?

We refund:

- an item you have returned, once the return has been opened (see the returns policy);
- a parcel that never arrived, when it qualifies as a missing delivery (see the shipping policy).

You can't be refunded twice for the same thing, and the total refunded on an order can never be more than you paid for it.

## How will I be refunded, and when?

Refunds go back to your original payment method. Once a refund has been issued, it usually takes **5 to 7 business days** to reach you, depending on your bank.

## Are there limits on refunds?

Our support assistant can approve refunds automatically up to these limits:

- **₹10,000 per order**, counting every refund already made or in progress on that order;
- **₹15,000 per customer** across all orders in any rolling **30 days**. Refunds for cancelled orders don't count towards this limit.

Above either limit, a specialist reviews the refund instead. This doesn't mean the refund is refused; it means a person checks it before it goes ahead.

## Goodwill refunds

A goodwill refund is money back that isn't owed under these policies, for example as an apology. Goodwill refunds are always reviewed by a specialist.

## Cancelled orders

When an order is cancelled before it ships, the full amount is refunded automatically. You don't need to ask for it, and the support assistant can't issue it a second time.
