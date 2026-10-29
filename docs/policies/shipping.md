---
policy: shipping
title: Shipping and delivery policy
source: backend/app/policy/policy.yaml
values:
  shipping.free_shipping_threshold_inr: 499
  shipping.standard_fee_inr: 40
  missing_delivery.grace_days_after_eta: 3
  remedies.missing_delivery: [reship, refund]
  reship.max_value_inr: 10000
  escalate_when.carrier_marked_delivered: true
  escalate_when.returned_to_sender: true
  escalate_when.over_reship_value_cap: true
---

# Shipping and delivery policy

## How much does shipping cost?

Shipping is free on orders of **₹499** or more. Smaller orders have a standard shipping fee of **₹40**.

## Why did my order arrive in more than one parcel?

Items can ship separately. Each parcel has its own tracking number and its own expected delivery date, and some items may ship later than others.

## My parcel is late. When does it count as missing?

A parcel counts as missing if it still hasn't been delivered **3 days** after its expected delivery date. Until then, it is treated as on its way, and we'll tell you the expected date.

If a parcel has no expected delivery date, it isn't treated as missing automatically; ask us and we'll look into it.

## What can I get if my parcel is missing?

For a missing parcel you can choose either:

- a free replacement, sent as a new order at no charge; or
- a refund.

You can have one or the other for the same order, not both.

A free replacement worth more than **₹10,000** is reviewed by a specialist before it's sent.

## The carrier says it was delivered, but I didn't get it

If the carrier recorded the parcel as delivered, a specialist investigates before we offer a replacement or refund. We'll open a case and follow up with you.

## The parcel is being returned to us

If the carrier is returning the parcel to us (for example, after failed delivery attempts), a specialist will help you decide what happens next.
