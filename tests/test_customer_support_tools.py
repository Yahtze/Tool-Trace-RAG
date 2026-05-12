from tool_trace_rag.tools.customer_support import build_customer_support_registry


def test_find_customer_by_name():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("find_customer", {"query": "Maya Chen"})

    assert result["status"] == "found"
    assert result["customer"]["customer_id"] == "cust_001"


def test_find_customer_returns_not_found():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("find_customer", {"query": "Unknown Person"})

    assert result == {
        "status": "not_found",
        "error_code": "CUSTOMER_NOT_FOUND",
        "message": "No customer matched query.",
    }


def test_get_customer_orders_returns_compact_summaries():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("get_customer_orders", {"customer_id": "cust_001"})

    assert result["status"] == "found"
    assert result["customer_id"] == "cust_001"
    assert len(result["orders"]) == 3
    assert result["orders"][0] == {
        "order_id": "ord_1001",
        "item": "headphones",
        "category": "electronics",
        "status": "delivered",
        "delivered_days_ago": 12,
    }


def test_get_order_returns_full_record():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("get_order", {"order_id": "ord_1005"})

    assert result["status"] == "found"
    assert result["order"]["item"] == "wireless mouse"
    assert result["order"]["opened"] is True


def test_refund_eligibility_allows_unopened_recent_electronics():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("check_refund_eligibility", {"order_id": "ord_1001"})

    assert result == {
        "status": "ok",
        "order_id": "ord_1001",
        "eligible": True,
        "reason": "Order is within the 30-day refund window.",
    }


def test_refund_eligibility_blocks_opened_electronics():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("check_refund_eligibility", {"order_id": "ord_1005"})

    assert result == {
        "status": "ok",
        "order_id": "ord_1005",
        "eligible": False,
        "reason": "Opened electronics are not refundable.",
    }


def test_refund_eligibility_blocks_not_delivered_order():
    registry = build_customer_support_registry("data/mock_customer_support.json")

    result = registry.execute("check_refund_eligibility", {"order_id": "ord_1003"})

    assert result == {
        "status": "ok",
        "order_id": "ord_1003",
        "eligible": False,
        "reason": "Order has not been delivered yet.",
    }
