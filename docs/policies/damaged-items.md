---
policy: damaged-items
title: Damaged and wrong items policy
source: backend/app/policy/policy.yaml
values:
  returns.window_days: 30
  remedies.damaged: [reship, refund]
  remedies.wrong_item: [reship, refund]
  reship.max_value_inr: 10000
  refunds.per_order_cap_inr: 10000
---

# Damaged and wrong items policy

## What should I do if an item arrives damaged, or isn't what I ordered?

Contact us within **30 days** of delivery. We'll ask which item on the order is affected and what's wrong with it.

## What can I get?

For a damaged item, or an item sent in error, you can choose either:

- a free replacement of that item, sent as a new order at no charge; or
- a refund of that item, after a return has been opened for it.

A replacement or refund is for the affected item only, not the whole order.

## Is there a time limit?

Yes. We can replace or refund a damaged or wrong item only within 30 days of its delivery. After that, the item is outside the returns window.

## Are there limits?

A replacement worth more than **₹10,000** is reviewed by a specialist before it's sent. A refund that would take the total refunded on the order above **₹10,000** is also reviewed by a specialist. See the refunds policy for the other refund limits.

## What if the item has already been refunded?

An item that has already been refunded can't also be replaced.
