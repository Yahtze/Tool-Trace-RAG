from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tool_trace_rag.tools.registry import ToolDefinition, ToolRegistry


class CustomerSupportTools:
    def __init__(self, data_path: str | Path) -> None:
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.customers: list[dict[str, Any]] = data["customers"]
        self.orders: list[dict[str, Any]] = data["orders"]
        self.policies: dict[str, Any] = data["policies"]

    def find_customer(self, query: str) -> dict[str, Any]:
        normalized_query = query.strip().lower()
        for customer in self.customers:
            if normalized_query in {
                customer["customer_id"].lower(),
                customer["name"].lower(),
                customer["email"].lower(),
            }:
                return {"status": "found", "customer": customer}

        return {
            "status": "not_found",
            "error_code": "CUSTOMER_NOT_FOUND",
            "message": "No customer matched query.",
        }

    def get_customer_orders(self, customer_id: str) -> dict[str, Any]:
        customer_orders = [order for order in self.orders if order["customer_id"] == customer_id]
        if not customer_orders:
            return {
                "status": "not_found",
                "error_code": "CUSTOMER_ORDERS_NOT_FOUND",
                "message": "No orders matched customer_id.",
            }

        summaries = [
            {
                "order_id": order["order_id"],
                "item": order["item"],
                "category": order["category"],
                "status": order["status"],
                "delivered_days_ago": order["delivered_days_ago"],
            }
            for order in customer_orders
        ]
        return {"status": "found", "customer_id": customer_id, "orders": summaries}

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._find_order(order_id)
        if order is None:
            return {
                "status": "not_found",
                "error_code": "ORDER_NOT_FOUND",
                "message": "No order matched order_id.",
            }
        return {"status": "found", "order": order}

    def check_refund_eligibility(self, order_id: str) -> dict[str, Any]:
        order = self._find_order(order_id)
        if order is None:
            return {
                "status": "not_found",
                "error_code": "ORDER_NOT_FOUND",
                "message": "No order matched order_id.",
            }

        if order["status"] in self.policies["not_delivered_statuses"]:
            return self._refund_result(order_id, False, "Order has not been delivered yet.")
        if order["status"] == "cancelled":
            return self._refund_result(order_id, False, "Cancelled orders are not refundable.")
        if order["category"] in self.policies["final_sale_categories"]:
            return self._refund_result(order_id, False, "Final sale items are not refundable.")
        if order["category"] == "electronics" and order["opened"] and not self.policies["opened_electronics_refundable"]:
            return self._refund_result(order_id, False, "Opened electronics are not refundable.")
        if order["delivered_days_ago"] is None:
            return self._refund_result(order_id, False, "Order has not been delivered yet.")
        if order["delivered_days_ago"] > self.policies["standard_refund_window_days"]:
            return self._refund_result(order_id, False, "Order is outside the 30-day refund window.")
        return self._refund_result(order_id, True, "Order is within the 30-day refund window.")

    def _find_order(self, order_id: str) -> dict[str, Any] | None:
        return next((order for order in self.orders if order["order_id"] == order_id), None)

    @staticmethod
    def _refund_result(order_id: str, eligible: bool, reason: str) -> dict[str, Any]:
        return {"status": "ok", "order_id": order_id, "eligible": eligible, "reason": reason}


def build_customer_support_registry(data_path: str | Path = "data/mock_customer_support.json") -> ToolRegistry:
    tools = CustomerSupportTools(data_path)
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="find_customer",
            description="Find a customer by customer ID, exact name, or email address.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Customer ID, exact name, or email."}},
                "required": ["query"],
                "additionalProperties": False,
            },
            function=tools.find_customer,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_customer_orders",
            description="Return compact order summaries for a customer.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
            function=tools.get_customer_orders,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_order",
            description="Return the full record for an order.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            function=tools.get_order,
        )
    )
    registry.register(
        ToolDefinition(
            name="check_refund_eligibility",
            description="Check whether an order is eligible for refund under the store policy.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            function=tools.check_refund_eligibility,
        )
    )
    return registry
