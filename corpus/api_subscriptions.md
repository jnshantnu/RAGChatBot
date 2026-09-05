---
doc_id: api_subscriptions
acl: [public]
category: api
---

# Subscriptions API

## Change Subscription

`POST /v1/subscriptions/{subscription_id}/change`

Used for upgrade, cross-grade, and downgrade requests (see the Policy
Switch Guide for the business rules each operation follows). The body
must specify a `target_sku` and a `change_type` of `upgrade`,
`cross_grade`, or `downgrade`.

```json
{
  "target_sku": "AUTO-CAD-PRO-2026",
  "change_type": "upgrade",
  "effective": "immediate"
}
```

`effective` may be `immediate` or `next_term`. Immediate changes trigger
a proration calculation (see Order Detail and Proration doc).

## Cancel Subscription

`POST /v1/subscriptions/{subscription_id}/cancel`

Accepts an optional `reason_code`. Cancellation is effective at the end
of the current term unless `immediate: true` is passed, which forfeits
the remaining term without refund.

## Webhooks

Subscribe to `subscription.changed` and `subscription.cancelled` events
via `POST /v1/webhooks`. Payloads are signed with HMAC-SHA256; verify
using the `X-Signature` header before trusting the body.
