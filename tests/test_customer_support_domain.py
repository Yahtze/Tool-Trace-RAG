from tool_trace_rag.tools.customer_support import (
    InMemoryCustomerSupportAdapter,
    JsonCustomerSupportAdapter,
    RefundPolicyModule,
)


def test_refund_policy_module_works_with_in_memory_adapter():
    adapter = InMemoryCustomerSupportAdapter(
        customers=[{"customer_id": "cust_001", "name": "Maya Chen", "email": "maya@example.com"}],
        orders=[
            {
                "order_id": "ord_1001",
                "customer_id": "cust_001",
                "item": "headphones",
                "category": "electronics",
                "status": "delivered",
                "delivered_days_ago": 12,
                "opened": False,
            }
        ],
        policies={
            "standard_refund_window_days": 30,
            "opened_electronics_refundable": False,
            "final_sale_categories": ["clearance"],
            "not_delivered_statuses": ["processing", "shipped"],
        },
    )

    module = RefundPolicyModule(adapter)

    assert module.find_customer("maya@example.com")["status"] == "found"
    assert module.check_refund_eligibility("ord_1001") == {
        "status": "ok",
        "order_id": "ord_1001",
        "eligible": True,
        "reason": "Order is within the 30-day refund window.",
    }


def test_json_adapter_exposes_customer_order_and_policy_data():
    adapter = JsonCustomerSupportAdapter("data/mock_customer_support.json")

    assert len(adapter.customers()) > 0
    assert len(adapter.orders()) > 0
    assert "standard_refund_window_days" in adapter.policies()
