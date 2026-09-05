---
doc_id: business_models
acl: [public]
category: policy
---

# Business Models: Agency vs. Buy-Sell

## Agency Model

The partner acts as an agent: the end customer is invoiced directly by
the vendor, and the partner earns a commission on the sale. The partner
never takes title to the license. Order records show the partner as
`agent_of_record`, not as the billing party.

Agency orders are always placed with `business_model: "agency"` in the
Order Management API. Commission accrual for agency orders is described
in the Partner Incentives doc, which is restricted to principal users.

## Buy-Sell Model

The partner purchases the license from the vendor at a discounted rate
and resells it to the end customer at a price of the partner's choosing.
The partner takes title to the license and is the billing party of
record for the vendor invoice. The partner's margin is the difference
between vendor cost and resale price, not a paid commission.

Buy-Sell orders use `business_model: "buy_sell"`.

## Choosing a Model

A partner may operate under either model depending on region and
product line; the model is fixed at order creation time and cannot be
changed after the order is provisioned. To switch models for a
customer relationship going forward, a new order must be created under
the new model when the current term ends.
