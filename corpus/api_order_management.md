---
doc_id: api_order_management
acl: [public]
category: api
---

# Order Management API

## Create Order

`POST /v1/orders`

Creates a new order for a partner-managed subscription. Requires an
`Idempotency-Key` header to guard against duplicate submissions on retry.

Request body:
```json
{
  "customer_id": "cust_123",
  "product_sku": "AUTO-CAD-2026",
  "quantity": 5,
  "business_model": "agency"
}
```

Returns a `202 Accepted` with an `order_id`. Orders are processed
asynchronously; poll `GET /v1/orders/{order_id}` for status.

## Get Order

`GET /v1/orders/{order_id}`

Returns order status, line items, and current proration snapshot.
Status values: `pending`, `provisioned`, `failed`, `cancelled`.

## Rate Limits

100 requests/minute per API key, burst of 20. A `429` response includes a
`Retry-After` header in seconds.

## Authentication

All API calls require a bearer token obtained via the partner OAuth2
client-credentials flow. Tokens expire after 1 hour.
